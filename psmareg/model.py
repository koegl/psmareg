"""The LapIRN pyramid: three levels, each predicting a stationary velocity field.

Level *k* takes the velocity of the coarser level, upsamples it, predicts a
residual, and adds the two. Exponentiating the sum by scaling and squaring gives
a diffeomorphic deformation. Levels run at quarter, half and full resolution and
are trained in that order, each initialised from the previous one.

Module and submodule names are load-bearing: they are the keys of the published
checkpoint. In particular every level calls its encoder ``input_encoder_lvl1``
and its trunk ``resblock_group_lvl1`` regardless of which level it is — an
inherited quirk, kept so the weights load.
"""

from pathlib import Path
from typing import NamedTuple, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .transforms import identity_grid_tensor, integrate_velocity, warp


class LevelOutput(NamedTuple):
    """What one pyramid level returns.

    ``flow`` is the exponentiated field of *this* level (channel-first, this
    level's resolution); ``velocity`` is the pre-exponential sum that the next
    level up continues from. ``embedding`` is the trunk feature map, added into
    the next level's encoder output as a skip connection.
    """

    flow: torch.Tensor
    warped_moving: torch.Tensor
    velocity: torch.Tensor
    embedding: torch.Tensor


class PreActBlock(nn.Module):
    """Pre-activation residual block.

    ``expansion`` makes it an inverted bottleneck — conv1 lifts to
    ``channels * expansion`` and conv2 projects back — so the block's input and
    output width are unchanged and only its interior widens. ``expansion=1`` is
    the original block, parameter shapes and state_dict keys included.
    """

    def __init__(self, channels: int, bias: bool = False, expansion: int = 1):
        super().__init__()
        hidden = channels * expansion
        self.conv1 = nn.Conv3d(channels, hidden, 3, stride=1, padding=1, bias=bias)
        self.conv2 = nn.Conv3d(hidden, channels, 3, stride=1, padding=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.leaky_relu(x, negative_slope=0.2)
        out = self.conv1(out)
        out = self.conv2(F.leaky_relu(out, negative_slope=0.2))
        return out + x


def _resblock_trunk(channels: int, n_blocks: int, expansion: int) -> nn.Sequential:
    """The residual trunk shared by every level.

    Depth also sets the receptive field (roughly +4 voxels per block at the
    trunk's own resolution), which caps how much misalignment a level can see;
    widening the blocks does not.
    """
    layers: list = []
    for _ in range(n_blocks):
        layers.append(PreActBlock(channels, bias=False, expansion=expansion))
        layers.append(nn.LeakyReLU(0.2))
    return nn.Sequential(*layers)


def _encoder(in_channels: int, out_channels: int) -> nn.Sequential:
    """Two 3x3x3 convolutions; the shallow path into the trunk."""
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, 3, stride=1, padding=1, bias=False),
        nn.LeakyReLU(0.2),
        nn.Conv3d(out_channels, out_channels, 3, stride=1, padding=1, bias=False),
    )


def _velocity_head(in_channels: int, out_channels: int) -> nn.Sequential:
    """Readout producing the velocity field.

    The closing ``Softsign`` bounds the output to (-1, 1); the caller scales it
    by ``range_flow``, which is what keeps the field small enough for scaling
    and squaring to stay invertible.
    """
    return nn.Sequential(
        nn.Conv3d(in_channels, in_channels // 2, 3, stride=1, padding=1, bias=False),
        nn.LeakyReLU(0.2),
        nn.Conv3d(in_channels // 2, out_channels, 3, stride=1, padding=1, bias=False),
        nn.Softsign(),
    )


class _PyramidLevel(nn.Module):
    """Shared machinery of the three levels.

    The levels differ only in what they feed the encoder — level 1 has no
    coarser level, levels 2 and 3 warp the moving image by the coarser flow and
    concatenate the incoming velocity — so everything else lives here.
    """

    def __init__(
        self,
        in_channel: int,
        n_classes: int,
        start_channel: int,
        img_shape: Sequence[int],
        range_flow: float,
        n_resblocks: int,
        resblock_expansion: int,
        integration_steps: int,
        encoder_extra_channels: int,
        device: torch.device,
    ):
        super().__init__()
        self.range_flow = range_flow
        self.integration_steps = integration_steps
        self.img_shape = tuple(img_shape)

        width = start_channel * 4
        self.input_encoder_lvl1 = _encoder(in_channel + encoder_extra_channels, width)
        self.down_conv = nn.Conv3d(width, width, 3, stride=2, padding=1, bias=False)
        self.resblock_group_lvl1 = _resblock_trunk(
            width, n_resblocks, resblock_expansion
        )
        self.up = nn.ConvTranspose3d(width, width, 2, stride=2, bias=False)
        self.output_lvl1 = _velocity_head(width * 2, n_classes)

        self.down_avg = nn.AvgPool3d(3, stride=2, padding=1, count_include_pad=False)
        self.up_tri = nn.Upsample(scale_factor=2, mode="trilinear")

        # persistent=False keeps it out of the state_dict — the published
        # checkpoint has no such key — while still following the module across
        # `.to(device)`.
        self.register_buffer(
            "grid", identity_grid_tensor(self.img_shape, device), persistent=False
        )

    def _trunk(
        self, cat_input: torch.Tensor, embedding: Optional[torch.Tensor]
    ) -> tuple:
        """Encoder, trunk and velocity head, in bf16 where the device allows it.

        Only the convolutions are autocast: the flow leaves this function in
        fp32, so every composition, ``grid_sample`` and loss downstream runs at
        full precision. bf16 here saves roughly 40% of the activation memory.
        """
        device_type = cat_input.device.type
        with torch.autocast(
            device_type=device_type,
            dtype=torch.bfloat16,
            enabled=device_type == "cuda",
        ):
            features = self.input_encoder_lvl1(cat_input)
            trunk = self.down_conv(features)
            if embedding is not None:
                trunk = trunk + embedding
            trunk = self.resblock_group_lvl1(trunk)
            trunk = self.up(trunk)
            velocity = (
                self.output_lvl1(torch.cat([trunk, features], dim=1)) * self.range_flow
            )
        return velocity.float(), trunk.float()

    def _finish(
        self, moving: torch.Tensor, velocity: torch.Tensor, embedding: torch.Tensor
    ) -> LevelOutput:
        flow = integrate_velocity(velocity, self.grid, self.integration_steps)
        warped = warp(moving, flow.permute(0, 2, 3, 4, 1), self.grid)
        return LevelOutput(flow, warped, velocity, embedding)


class Level1(_PyramidLevel):
    """Coarsest level, at quarter resolution. Sees no incoming velocity."""

    def __init__(self, cfg: ModelConfig, device: torch.device):
        super().__init__(
            in_channel=cfg.in_channel,
            n_classes=cfg.n_classes,
            start_channel=cfg.start_channel,
            img_shape=cfg.img_shape_4,
            range_flow=cfg.range_flow,
            n_resblocks=cfg.n_resblocks,
            resblock_expansion=cfg.resblock_expansion,
            integration_steps=cfg.integration_steps,
            encoder_extra_channels=0,
            device=device,
        )

    def forward(self, moving: torch.Tensor, fixed: torch.Tensor) -> LevelOutput:
        # Both images are pooled twice to reach quarter resolution. Pooling the
        # concatenation rather than each image keeps this to two calls.
        cat_input = self.down_avg(self.down_avg(torch.cat((moving, fixed), 1)))
        velocity, embedding = self._trunk(cat_input, None)
        return self._finish(moving, velocity, embedding)


class Level2(_PyramidLevel):
    """Half resolution, continuing from level 1."""

    def __init__(self, cfg: ModelConfig, device: torch.device, level1: Level1):
        super().__init__(
            in_channel=cfg.in_channel,
            n_classes=cfg.n_classes,
            start_channel=cfg.start_channel,
            img_shape=cfg.img_shape_2,
            range_flow=cfg.range_flow,
            n_resblocks=cfg.n_resblocks,
            resblock_expansion=cfg.resblock_expansion,
            integration_steps=cfg.integration_steps,
            # the 3 velocity channels handed up by level 1
            encoder_extra_channels=3,
            device=device,
        )
        self.model_lvl1 = level1

    def forward(self, moving: torch.Tensor, fixed: torch.Tensor) -> LevelOutput:
        coarse = self.model_lvl1(moving, fixed)
        flow_up = self.up_tri(coarse.flow)
        velocity_up = self.up_tri(coarse.velocity)

        down_moving = self.down_avg(moving)
        down_fixed = self.down_avg(fixed)
        warped = warp(down_moving, flow_up.permute(0, 2, 3, 4, 1), self.grid)
        cat_input = torch.cat((warped, down_fixed, velocity_up), 1)

        residual, embedding = self._trunk(cat_input, coarse.embedding)
        # The velocities add; it is their sum that gets exponentiated, which is
        # what makes the pyramid one transform rather than a chain of them.
        return self._finish(moving, residual + velocity_up.float(), embedding)


class Level3(_PyramidLevel):
    """Full resolution, continuing from level 2. The deployed model."""

    def __init__(self, cfg: ModelConfig, device: torch.device, level2: Level2):
        super().__init__(
            in_channel=cfg.in_channel,
            n_classes=cfg.n_classes,
            start_channel=cfg.start_channel,
            img_shape=cfg.img_shape,
            range_flow=cfg.range_flow,
            n_resblocks=cfg.n_resblocks,
            resblock_expansion=cfg.resblock_expansion,
            integration_steps=cfg.integration_steps,
            encoder_extra_channels=3,
            device=device,
        )
        self.model_lvl2 = level2

    def forward(self, moving: torch.Tensor, fixed: torch.Tensor) -> LevelOutput:
        coarse = self.model_lvl2(moving, fixed)
        flow_up = self.up_tri(coarse.flow)
        velocity_up = self.up_tri(coarse.velocity)

        warped = warp(moving, flow_up.permute(0, 2, 3, 4, 1), self.grid)
        cat_input = torch.cat((warped, fixed, velocity_up), 1)

        residual, embedding = self._trunk(cat_input, coarse.embedding)
        return self._finish(moving, residual + velocity_up.float(), embedding)


def build_model(
    cfg: ModelConfig, device: torch.device, weights: Optional[Path] = None
) -> Level3:
    """Assemble the three levels and, if given, load a checkpoint into them.

    The checkpoint is a level-3 state_dict with the two coarser levels nested
    inside it, so one load covers the whole pyramid.
    """
    model = Level3(cfg, device, Level2(cfg, device, Level1(cfg, device))).to(device)
    if weights is not None:
        model.load_state_dict(torch.load(weights, map_location=device))
    return model
