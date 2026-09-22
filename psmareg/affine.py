"""CT-only affine pre-registration with ANTs.

The pyramid can only recover what its receptive field can see, and longitudinal
whole-body scans routinely differ by more than that — different table position,
arms up versus down, a different axial field of view. A rigid-plus-scale
alignment first brings the two within the network's range.

Only CT drives this stage. PET uptake changes with therapy, so matching it would
align disease rather than anatomy.
"""

from pathlib import Path
from typing import Tuple

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
