"""Training one pyramid level.

Each level is trained on its own, initialised from the level below and running
at its own resolution. The two coarser levels are supervised against the
affinely pre-registered moving image, because that is what the network sees.
The full-resolution level instead composes its prediction with the affine and is
supervised on the *total* transform applied to the original moving scan: the
biomarker terms have to see the same volume change the scorer does, including
the affine's own determinant.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils import data as torch_data

from . import losses, metrics
from .affine import augment_displacement, cached_affine_displacement, displacement_to_flow
from .config import TrainConfig
from .dataset import RegistrationPairs, patient_split
from .jacobian import jacobian_matrix, non_diff_volume_loss
from .model import build_level, set_nested_trainable
from .transforms import compose, identity_grid_tensor, unit_flow_to_voxel, warp


def downsample(volume: torch.Tensor, factor: int, label: bool = False) -> torch.Tensor:
    """Bring a volume to a coarser pyramid level.

    Labels go through nearest sampling — averaging label *ids* would invent
    structures that are not there.
    """
    if factor == 1:
        return volume
    if label:
        size = tuple(s // factor for s in volume.shape[2:])
        return F.interpolate(volume, size=size, mode="nearest")
    for _ in range(int(np.log2(factor))):
        volume = F.avg_pool3d(volume, 3, stride=2, padding=1, count_include_pad=False)
    return volume


def velocity_in_voxels(velocity: torch.Tensor) -> torch.Tensor:
    """Scale a unit velocity field to voxels, for the smoothness term.

    Without this the penalty would be anisotropic: one unit of flow is a
    different number of voxels along each axis. The factor is ``n - 1`` rather
    than ``(n - 1) / 2`` purely to match the scale ``w_smooth`` was tuned at.
    """
    _, _, d, h, w = velocity.shape
    scale = torch.tensor([w - 1, h - 1, d - 1], device=velocity.device)
    return velocity * scale.view(1, 3, 1, 1, 1)


@dataclass
class Batch:
    """One pair on the device, with the affine already replayed through the
    augmentation that was applied to the images."""

    moving: torch.Tensor
    fixed: torch.Tensor
    moving_label_ct: torch.Tensor
    moving_label_pet: torch.Tensor
    fixed_label_ct: torch.Tensor
    fixed_body_mask: torch.Tensor
    affine_flow: torch.Tensor


def prepare_batch(
    item: dict, data_dir: Path, cfg: TrainConfig, cache_dir: Path, device: torch.device
) -> Batch:
    """Move one item to the device and attach its affine pre-registration."""
    case, timepoint = item["case"][0], item["timepoint"][0]
    images = data_dir / "imagesTr"
    displacement = cached_affine_displacement(
        images / f"PSMARegPSMA_{case}_0000_00.nii.gz",
        images / f"PSMARegPSMA_{case}_0000_{timepoint}.nii.gz",
        cfg.model,
        cache_dir,
    )
    displacement = augment_displacement(
        displacement,
        flipped=bool(item["flipped"][0]),
        crop_head=int(item["crop_head"][0]),
        crop_feet=int(item["crop_feet"][0]),
        crop_head_fixed=int(item["crop_head_fixed"][0]),
        crop_feet_fixed=int(item["crop_feet_fixed"][0]),
    )
    to_device = lambda key: item[key].to(device).float()
    return Batch(
        moving=to_device("moving"),
        fixed=to_device("fixed"),
        moving_label_ct=to_device("moving_label_ct"),
        moving_label_pet=to_device("moving_label_pet"),
        fixed_label_ct=to_device("fixed_label_ct"),
        fixed_body_mask=to_device("fixed_body_mask"),
        affine_flow=displacement_to_flow(displacement, cfg.model, device),
    )


def objective(
    level: int,
    batch: Batch,
    output,
    grid: torch.Tensor,
    level_grid: torch.Tensor,
    cfg: TrainConfig,
    ncc: losses.MultiResolutionNCC,
    bone_values: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """The weighted objective for one level, and its terms for logging."""
    weights = cfg.level_weights(level)
    factor = 2 ** (3 - level)

    if level == 3:
        # Compose so every term sees the transform that will actually be
        # scored: affine outside, network inside, applied to the ORIGINAL
        # moving image in a single interpolation.
        flow = compose(batch.affine_flow, output.flow.permute(0, 2, 3, 4, 1), grid)
        moving, moving_label_ct = batch.moving, batch.moving_label_ct
        moving_label_pet = batch.moving_label_pet
        sample_grid = grid
    else:
        # The coarser levels never see the affine: they are trained on the
        # pre-registered pair, which is their input.
        flow = output.flow
        moving = warp(batch.moving, batch.affine_flow, grid)
        moving_label_ct = warp(batch.moving_label_ct, batch.affine_flow, grid, nearest=True)
        moving_label_pet = warp(batch.moving_label_pet, batch.affine_flow, grid, nearest=True)
        moving = downsample(moving, factor)
        moving_label_ct = downsample(moving_label_ct, factor, label=True)
        moving_label_pet = downsample(moving_label_pet, factor, label=True)
        sample_grid = level_grid

    fixed = downsample(batch.fixed, factor)
    fixed_label_ct = downsample(batch.fixed_label_ct, factor, label=True)
    body_mask = downsample(batch.fixed_body_mask, factor, label=True)

    flow_channel_last = flow.permute(0, 2, 3, 4, 1)
    flow_voxel = unit_flow_to_voxel(flow_channel_last)
    warped = warp(moving, flow_channel_last, sample_grid)

    loss = (
        weights["ncc"] * ncc(warped[:, 0:1], fixed[:, 0:1])
        + weights["non_diff"] * non_diff_volume_loss(flow_voxel, mask=body_mask)
        + weights["smooth"] * losses.smooth_loss(velocity_in_voxels(output.velocity))
    )
    logs: Dict[str, float] = {}

    dice = losses.dice_loss(moving_label_ct, fixed_label_ct, flow, sample_grid)
    if dice is not None:
        loss = loss + weights["dice"] * dice
        logs["dice"] = dice.item()

    if weights["mtv"] or weights["tlg"] or weights["jacobian_tumor"]:
        lesion = (moving_label_pet == 1).float()
        warped_lesion = warp(lesion, flow_channel_last, sample_grid)
        jac_det = jacobian_matrix(flow_voxel)[0]

        mtv = losses.mtv_bias(warped_lesion, lesion)
        tlg = losses.tlg_bias(warped[:, 1:2], warped_lesion, moving[:, 1:2], lesion)
        # det(J) is defined on the fixed grid, so the lesion has to be there too
        # — detached, so the gradient improves the determinant rather than
        # sliding the mask somewhere it is already 1.
        jac_tumour = losses.masked_jacobian_bias(jac_det, warped_lesion.detach())
        mtv_mean = losses.mean_jacobian_bias(jac_det, warped_lesion.detach())

        loss = loss + (
            weights["mtv"] * mtv**2
            + weights["mtv_mean"] * mtv_mean
            + weights["tlg"] * tlg
            + weights["jacobian_tumor"] * jac_tumour
        )
        logs.update(mtv=mtv.item(), tlg=tlg.item(), jac_tumour=jac_tumour.item())

    if weights["rigidity"]:
        # Fitted over the FIXED-frame bones: no label resampling is needed, and
        # the network is pushed to undo any non-rigid bone motion the affine
        # introduced.
        rigidity, _ = losses.per_label_rigid_loss(
            flow_voxel,
            batch.fixed_label_ct,
            bone_values,
            min_voxels=cfg.rigidity_min_voxels,
        )
        loss = loss + weights["rigidity"] * rigidity
        logs["rigidity"] = rigidity.item()

    logs["loss"] = loss.item()
    return loss, logs


@torch.no_grad()
def validate(
    level: int,
    model,
    loader: torch_data.DataLoader,
    data_dir: Path,
    cfg: TrainConfig,
    cache_dir: Path,
    grid: torch.Tensor,
    device: torch.device,
) -> Dict[str, float]:
    """Score the validation split on the quantities the challenge reports.

    Always on the total transform at full resolution, whatever level is being
    trained: a coarse level's own resolution is not what it will be judged at,
    and comparing levels on different grids would make the numbers meaningless.
    """
    model.eval()
    totals: Dict[str, list] = {k: [] for k in ("dice", "hd95", "mtv", "tlg", "ndv")}

    for item in loader:
        batch = prepare_batch(item, data_dir, cfg, cache_dir, device)
        moving_affine = warp(batch.moving, batch.affine_flow, grid)
        output = model(moving_affine, batch.fixed)
        # A coarse level predicts a smaller field; upsample before composing, the
        # same thing the scorer does with a sub-resolution submission.
        level_flow = output.flow
        if level_flow.shape[2:] != tuple(cfg.model.img_shape):
            level_flow = F.interpolate(
                level_flow, size=cfg.model.img_shape, mode="trilinear", align_corners=False
            )
        flow = compose(batch.affine_flow, level_flow.permute(0, 2, 3, 4, 1), grid)
        flow_channel_last = flow.permute(0, 2, 3, 4, 1)

        warped_label = warp(batch.moving_label_ct, flow_channel_last, grid, nearest=True)
        totals["dice"].append(
            1.0
            - metrics.multilabel_dice(
                warped_label[0, 0].round().long(), batch.fixed_label_ct[0, 0].round().long()
            )
        )
        totals["hd95"].append(
            metrics.hd95(
                warped_label, batch.fixed_label_ct, batch.moving_label_ct, cfg.model.spacing
            )
        )

        lesion = (batch.moving_label_pet == 1).float()
        warped_lesion = warp(lesion, flow_channel_last, grid, nearest=True)
        warped_pet = warp(batch.moving[:, 1:2], flow_channel_last, grid)
        totals["mtv"].append(losses.mtv_bias(warped_lesion, lesion).item())
        totals["tlg"].append(
            losses.tlg_bias(warped_pet, warped_lesion, batch.moving[:, 1:2], lesion).item()
        )
        totals["ndv"].append(
            non_diff_volume_loss(
                unit_flow_to_voxel(flow_channel_last), mask=batch.fixed_body_mask
            ).item()
        )

    model.train()
    return {key: float(np.nanmean(values)) for key, values in totals.items()}


def cycle(loader: torch_data.DataLoader) -> Iterator[dict]:
    """Endless iteration, so the loop is driven by steps rather than epochs."""
    while True:
        yield from loader


def train_level(
    level: int,
    data_dir: Path,
    out_dir: Path,
    cfg: TrainConfig,
    device: torch.device,
    init: Optional[Path] = None,
    steps: Optional[int] = None,
    cache_dir: Optional[Path] = None,
) -> Path:
    """Train one pyramid level and return the path of its best checkpoint."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = cache_dir or out_dir / "affine_cache"
    total_steps = steps or cfg.steps[level]

    train_cases, val_cases = patient_split(data_dir, out_dir / "split.json")
    train_set = RegistrationPairs(
        data_dir,
        train_cases,
        cfg.model,
        augment=cfg.augment,
        flip_prob=cfg.flip_prob,
        ct_shift_range=cfg.ct_shift_range,
        ct_scale_range=cfg.ct_scale_range,
        pet_scale_range=cfg.pet_scale_range,
        max_crop_z=cfg.max_crop_z,
        max_crop_z_asymmetric=cfg.max_crop_z_asymmetric,
    )
    val_set = RegistrationPairs(data_dir, val_cases, cfg.model, augment=False)
    print(
        f"level {level}: {len(train_set)} training pairs ({len(train_cases)} patients), "
        f"{len(val_set)} validation pairs ({len(val_cases)} patients)"
    )

    train_loader = torch_data.DataLoader(
        train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
    )
    val_loader = torch_data.DataLoader(val_set, batch_size=1, shuffle=False)

    model = build_level(cfg.model, device, level, init)
    set_nested_trainable(model, False)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr[level])

    grid = identity_grid_tensor(cfg.model.img_shape, device)
    level_shape = [cfg.model.img_shape_4, cfg.model.img_shape_2, cfg.model.img_shape][
        level - 1
    ]
    level_grid = identity_grid_tensor(level_shape, device)
    ncc = losses.MultiResolutionNCC(cfg.ncc_window[level], scales=level).to(device)
    bone_values = losses.bone_label_tensor(device)

    steps_per_epoch = max(len(train_loader), 1)
    warmup_steps = int(round(cfg.warmup_epochs * steps_per_epoch))
    unfreeze_step = cfg.unfreeze_epoch * steps_per_epoch
    val_interval = cfg.val_interval * steps_per_epoch

    best_score = float("-inf")
    best_path = out_dir / f"level{level}_best.pth"
    final_path = out_dir / f"level{level}_final.pth"

    batches = cycle(train_loader)
    model.train()
    start = time.time()

    for step in range(total_steps):
        if step < warmup_steps:
            # A fresh head emits noise; at full learning rate its first
            # gradients would disturb the level underneath before it says
            # anything useful.
            for group in optimizer.param_groups:
                group["lr"] = cfg.lr[level] * (step + 1) / warmup_steps
        if step == unfreeze_step:
            set_nested_trainable(model, True)
            print(f"step {step}: unfroze level {level - 1}")

        batch = prepare_batch(next(batches), data_dir, cfg, cache_dir, device)
        moving_affine = warp(batch.moving, batch.affine_flow, grid)
        output = model(moving_affine, batch.fixed)

        loss, logs = objective(
            level, batch, output, grid, level_grid, cfg, ncc, bone_values
        )
        (loss / cfg.accumulation_steps).backward()

        if (step + 1) % cfg.accumulation_steps == 0 or step + 1 == total_steps:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if step == 0 or (step + 1) % 100 == 0:
            terms = " ".join(f"{k}={v:.4f}" for k, v in logs.items())
            elapsed = time.time() - start
            print(
                f"step {step + 1}/{total_steps} {terms} "
                f"({elapsed / (step + 1):.2f}s/step)",
                flush=True,
            )

        if (step + 1) % val_interval == 0 or step + 1 == total_steps:
            scores = validate(
                level, model, val_loader, data_dir, cfg, cache_dir, grid, device
            )
            selection = metrics.challenge_score(
                cfg,
                dice_loss=scores["dice"],
                hd95_mm=scores["hd95"],
                mtv_bias=scores["mtv"],
                tlg_bias=scores["tlg"],
                ndv_percent=scores["ndv"],
            )
            print(
                f"  validation: dice_loss={scores['dice']:.4f} hd95={scores['hd95']:.2f}mm "
                f"mtv={scores['mtv']:.4f} tlg={scores['tlg']:.4f} ndv={scores['ndv']:.4f} "
                f"-> score {selection['final']:.3f}",
                flush=True,
            )
            # Selected on the composite score, not on Dice: alignment keeps
            # improving after the biomarker errors have bottomed out, so the
            # best-aligned checkpoint is not the best submission.
            if selection["final"] > best_score:
                best_score = selection["final"]
                torch.save(model.state_dict(), best_path)
                print(f"  new best ({best_score:.3f}) -> {best_path}", flush=True)

    torch.save(model.state_dict(), final_path)
    print(f"level {level} done in {(time.time() - start) / 3600:.1f}h; best {best_score:.3f}")
    return best_path
