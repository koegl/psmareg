"""CT-only affine pre-registration with ANTs.

The pyramid can only recover what its receptive field can see, and longitudinal
whole-body scans routinely differ by more than that — different table position,
arms up versus down, a different axial field of view. A rigid-plus-scale
alignment first brings the two within the network's range.

Only CT drives this stage. PET uptake changes with therapy, so matching it would
align disease rather than anatomy.
"""

import os
from pathlib import Path
from typing import Optional, Tuple

import ants
import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import zoom

from .config import CT_AIR_HU, ModelConfig
from .data import body_mask
from .transforms import voxel_disp_to_unit_flow

# Mattes mutual information with 32 sampling points, the ANTs defaults for a
# multi-modal-capable affine. Note this stage is not seeded from Python: ITK
# uses its own RNG, so repeated runs differ at the sub-voxel level.
ANTS_TRANSFORM = "Affine"
ANTS_METRIC = "mattes"
ANTS_SAMPLING = 32


def _to_ants(
    volume: np.ndarray,
    spacing: Tuple[float, float, float],
    downsample: int,
    window: Tuple[float, float],
) -> ants.ANTsImage:
    """Window to [0, 1], optionally downsample, and wrap as an ANTs image.

    The geometry is set explicitly rather than carried over from the NIfTI: both
    images are placed at the origin in a fixed LPS-style frame, so the affine
    ANTs returns is a pure image-to-image transform with no patient-position
    difference folded into it.
    """
    lo, hi = window
    windowed = (np.clip(volume, lo, hi) - lo) / (hi - lo)
    windowed[~np.isfinite(volume)] = 0.0

    if downsample != 1:
        windowed = zoom(windowed, 1.0 / downsample, order=1)

    image = ants.from_numpy(windowed.astype(np.float32))
    image.set_spacing(tuple(float(s) * downsample for s in spacing))
    image.set_origin((0.0, 0.0, 0.0))
    image.set_direction(np.diag([-1.0, -1.0, 1.0]))
    return image


def _affine_to_voxel_displacement(
    transform, reference: ants.ANTsImage, spacing: Tuple[float, float, float]
) -> np.ndarray:
    """Densify an ANTs affine into a ``(H, W, D, 3)`` voxel displacement field.

    ANTs gives a 3x3 matrix, a translation and a centre of rotation, all in
    physical space. Each voxel of ``reference`` is mapped to physical
    coordinates, transformed, and the difference converted back to voxels.
    """
    params = np.asarray(transform.parameters, dtype=np.float32)
    fixed_params = np.asarray(transform.fixed_parameters, dtype=np.float32)
    if params.size != 12 or fixed_params.size < 3:
        raise ValueError(
            f"expected a 3D affine (12 parameters + centre), got {params.size}"
        )

    matrix = params[:9].reshape(3, 3)
    translation = params[9:12]
    centre = fixed_params[:3]

    shape = tuple(int(v) for v in reference.shape)
    direction = np.asarray(reference.direction, dtype=np.float32)
    origin = np.asarray(reference.origin, dtype=np.float32)
    reference_spacing = np.asarray(reference.spacing, dtype=np.float32)

    index = np.stack(
        np.meshgrid(*[np.arange(n, dtype=np.float32) for n in shape], indexing="ij"),
        axis=-1,
    ).reshape(-1, 3)
    physical = origin + (index * reference_spacing).dot(direction.T)
    moved = (physical - centre).dot(matrix.T) + centre + translation

    delta = (moved - physical).reshape(shape + (3,))
    # Back to voxels: undo the direction cosines, then divide by the spacing of
    # the FULL-resolution grid the field will be applied on.
    voxel_delta = delta.reshape(-1, 3).dot(direction) / np.asarray(
        spacing, dtype=np.float32
    )
    return voxel_delta.reshape(shape + (3,)).astype(np.float32)


def affine_displacement(
    fixed_ct: Path, moving_ct: Path, cfg: ModelConfig
) -> np.ndarray:
    """Estimate the affine and return it as a full-resolution voxel field.

    Registration runs at ``cfg.affine_downsample`` — half resolution is ample
    for a 12-parameter fit and several times faster — but the field is
    densified on the full-resolution grid.
    """
    fixed_nii = nib.load(str(fixed_ct))
    moving_nii = nib.load(str(moving_ct))
    spacing = tuple(float(s) for s in fixed_nii.header.get_zooms()[:3])

    fixed = fixed_nii.get_fdata(dtype=np.float32)
    moving = moving_nii.get_fdata(dtype=np.float32)
    # Same bed removal as the network input: the bed does not move between
    # timepoints, so leaving it in biases the fit towards the table.
    fixed = np.where(body_mask(fixed), fixed, CT_AIR_HU)
    moving = np.where(body_mask(moving), moving, CT_AIR_HU)

    window, factor = cfg.affine_ct_window, cfg.affine_downsample
    fixed_lowres = _to_ants(fixed, spacing, factor, window)
    moving_lowres = _to_ants(moving, spacing, factor, window)
    fixed_fullres = _to_ants(fixed, spacing, 1, window)

    result = ants.registration(
        fixed=fixed_lowres,
        moving=moving_lowres,
        type_of_transform=ANTS_TRANSFORM,
        aff_metric=ANTS_METRIC,
        aff_sampling=ANTS_SAMPLING,
        verbose=False,
    )
    transform = ants.read_transform(result["fwdtransforms"][0])
    return _affine_to_voxel_displacement(transform, fixed_fullres, spacing)


def affine_flow(
    fixed_ct: Path, moving_ct: Path, cfg: ModelConfig, device: torch.device
) -> torch.Tensor:
    """:func:`affine_displacement` as a unit flow, ``(1, H, W, D, 3)``."""
    displacement = affine_displacement(fixed_ct, moving_ct, cfg)
    tensor = torch.from_numpy(displacement).permute(3, 0, 1, 2)[None].to(device).float()
    return voxel_disp_to_unit_flow(tensor, cfg.img_shape)


# ---------------------------------------------------------------------------
# Training-time reuse
#
# The affine is deterministic given a pair, and ANTs takes ~15 s of CPU per
# call, which would dominate a training step. It is therefore computed once per
# pair and cached on disk; augmentation is applied to the cached field
# afterwards, so the field stays consistent with the images it will be composed
# with.
# ---------------------------------------------------------------------------


def cached_affine_displacement(
    fixed_ct: Path, moving_ct: Path, cfg: ModelConfig, cache_dir: Optional[Path]
) -> np.ndarray:
    """:func:`affine_displacement`, memoised on disk under ``cache_dir``."""
    if cache_dir is None:
        return affine_displacement(fixed_ct, moving_ct, cfg)

    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{moving_ct.stem}__{fixed_ct.stem}.npy"
    if path.exists():
        return np.load(path).astype(np.float32)

    displacement = affine_displacement(fixed_ct, moving_ct, cfg)
    # Write via a temporary file: several dataloader workers may race on the
    # same pair, and a half-written .npy would poison the cache permanently.
    temporary = path.with_suffix(f".{os.getpid()}.tmp.npy")
    np.save(temporary, displacement)
    temporary.replace(path)
    return displacement


def flip_displacement(displacement: np.ndarray) -> np.ndarray:
    """Mirror a displacement field along the left-right axis.

    Two things change: the field is reversed along that axis, and its component
    along it flips sign. Reversing alone would describe the same motion in a
    mirrored body, which is not what a mirrored image needs.
    """
    flipped = displacement[::-1].copy()
    flipped[..., 0] = -flipped[..., 0]
    return flipped


def crop_displacement(
    displacement: np.ndarray, crop_head: int, crop_feet: int
) -> np.ndarray:
    """Apply the same axial crop-and-zero-pad the images get.

    Removed slices become zero displacement rather than being dropped, so the
    field keeps the original shape and the padded region is the identity.
    """
    if crop_head == 0 and crop_feet == 0:
        return displacement

    depth = displacement.shape[2]
    cropped = displacement[:, :, crop_head : depth - crop_feet if crop_feet else depth]
    pad = lambda n: np.zeros(displacement.shape[:2] + (n, 3), dtype=displacement.dtype)
    return np.concatenate([pad(crop_head), cropped, pad(crop_feet)], axis=2)


def augment_displacement(
    displacement: np.ndarray,
    flipped: bool,
    crop_head: int,
    crop_feet: int,
    crop_head_fixed: int = 0,
    crop_feet_fixed: int = 0,
) -> np.ndarray:
    """Replay the image augmentation on a cached affine field.

    The field is indexed in the *fixed* frame — warping samples the moving image
    at ``grid + flow`` — so it has to follow the fixed image's crops, in the
    order they were applied: the shared crop first, then the fixed-only one. The
    moving-only crop is deliberately absent: that one is already baked into the
    moving image this field will sample from.
    """
    if flipped:
        displacement = flip_displacement(displacement)
    displacement = crop_displacement(displacement, crop_head, crop_feet)
    return crop_displacement(displacement, crop_head_fixed, crop_feet_fixed)


def displacement_to_flow(
    displacement: np.ndarray, cfg: ModelConfig, device: torch.device
) -> torch.Tensor:
    """A ``(H, W, D, 3)`` voxel field as a unit flow ``(1, H, W, D, 3)``."""
    tensor = torch.from_numpy(np.ascontiguousarray(displacement))
    tensor = tensor.permute(3, 0, 1, 2)[None].to(device).float()
    return voxel_disp_to_unit_flow(tensor, cfg.img_shape)
