"""Grids, warping, and the scaling-and-squaring exponential.

Every flow in this codebase is a *unit* flow: displacements in the [-1, 1]
coordinates ``grid_sample`` uses, not voxels. Conversion to voxel units happens
once, on the way out, in :func:`unit_flow_to_voxel`.

Flows are laid out channel-first ``(1, 3, H, W, D)`` where they come out of a
convolution, and channel-last ``(1, H, W, D, 3)`` where they are added to a
sampling grid. The two are one ``permute`` apart; each function below says
which it takes.
"""

from typing import Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def identity_grid(img_shape: Sequence[int]) -> np.ndarray:
    """Sampling grid of the identity transform, shape ``(H, W, D, 3)``.

    ``align_corners=False`` throughout, so voxel *i* sits at
    ``((i + 0.5) / dim) * 2 - 1``. The last axis is ordered (z, y, x) — the
    order ``grid_sample`` expects, which is the reverse of the spatial axes.
    """
    axes = [(np.arange(n) + 0.5) / n * 2 - 1 for n in img_shape]
    grid = np.rollaxis(np.array(np.meshgrid(axes[2], axes[1], axes[0])), 0, 4)
    grid = np.swapaxes(grid, 0, 2)
    return np.swapaxes(grid, 1, 2)


def identity_grid_tensor(
    img_shape: Sequence[int], device: torch.device
) -> torch.Tensor:
    """:func:`identity_grid` as a batched tensor of shape ``(1, H, W, D, 3)``."""
    grid = identity_grid(img_shape)
    return torch.from_numpy(grid[None]).to(device).float()


def warp(
    volume: torch.Tensor, flow: torch.Tensor, grid: torch.Tensor, nearest: bool = False
) -> torch.Tensor:
    """Resample ``volume`` at ``grid + flow``.

    ``volume`` is ``(1, C, H, W, D)`` and ``flow`` channel-last
    ``(1, H, W, D, 3)``. Both are upcast to fp32: a bf16 flow arriving from an
    autocast conv trunk would otherwise quantise the sampling positions.

    ``nearest=True`` for label maps, which must not be interpolated.
    """
    return F.grid_sample(
        volume.float(),
        grid + flow.float(),
        mode="nearest" if nearest else "bilinear",
        padding_mode="border",
        align_corners=False,
    )


def integrate_velocity(
    velocity: torch.Tensor, grid: torch.Tensor, steps: int = 7
) -> torch.Tensor:
    """Exponentiate a stationary velocity field by scaling and squaring.

    Takes and returns a channel-first ``(1, 3, H, W, D)`` field. The result is
    a diffeomorphism as long as the scaled velocity is small enough that each
    squaring step stays invertible, which is what ``range_flow`` bounds.

    Kept in fp32 unconditionally — bf16 rounding compounds over the squarings.
    """
    velocity = velocity.float()
    grid = grid.float()
    flow = velocity / (2.0**steps)
    for _ in range(steps):
        sample_at = grid + flow.permute(0, 2, 3, 4, 1)
        flow = flow + F.grid_sample(
            flow, sample_at, mode="bilinear", padding_mode="border", align_corners=False
        )
    return flow


def compose(
    outer: torch.Tensor, inner: torch.Tensor, grid: torch.Tensor
) -> torch.Tensor:
    """Compose two channel-last unit flows into one, ``outer ∘ inner``.

    ``outer`` is applied *after* ``inner``. In this pipeline the affine is the
    outer field and the network prediction the inner one, so the result maps
    the original (un-prereg'd) moving image onto the fixed grid.

    Returns the composed flow channel-first, ``(1, 3, H, W, D)``.
    """
    outer_grid = (grid + outer).permute(0, 4, 1, 2, 3)
    composed = F.grid_sample(
        outer_grid,
        grid + inner,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    ).permute(0, 2, 3, 4, 1)
    return (composed - grid).permute(0, 4, 1, 2, 3)


def unit_flow_to_voxel(flow: torch.Tensor) -> torch.Tensor:
    """Convert a channel-last unit flow ``(1, H, W, D, 3)`` to voxel units.

    The unit grid spans [-1, 1] across each axis, so one unit is ``(n - 1) / 2``
    voxels. The channel order is (z, y, x), hence the reversed pairing with the
    spatial dimensions.
    """
    _, h, w, d, _ = flow.shape
    scale = torch.tensor(
        [(d - 1) / 2.0, (w - 1) / 2.0, (h - 1) / 2.0],
        dtype=flow.dtype,
        device=flow.device,
    )
    return flow * scale


def voxel_disp_to_unit_flow(
    disp: torch.Tensor, img_shape: Tuple[int, int, int]
) -> torch.Tensor:
    """Inverse of :func:`unit_flow_to_voxel`, for a channel-first ``(1, 3, ...)``
    voxel displacement field; returns a channel-last unit flow.

    Used for the affine stage, which ANTs hands back in voxels. Note the axis
    reversal: channel *i* of the input is the displacement along spatial axis
    *i*, whereas a unit flow orders its channels (z, y, x).

    The scale here is ``n / 2`` where :func:`unit_flow_to_voxel` uses
    ``(n - 1) / 2``. That half-voxel inconsistency is inherited from the trained
    pipeline: the network learned to correct whatever the affine stage left
    behind, so "fixing" it here would shift the input the checkpoint expects.
    """
    h, w, d = img_shape
    return torch.stack(
        [disp[:, 2] / (d / 2.0), disp[:, 1] / (w / 2.0), disp[:, 0] / (h / 2.0)],
        dim=1,
    ).permute(0, 2, 3, 4, 1)
