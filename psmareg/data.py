"""Reading a pair, normalising it, and writing the displacement field.

The body mask is the only non-obvious step: the scans include the scanner bed
and, in PET, whatever activity sits outside the patient. Both are stationary
between timepoints and neither belongs to the anatomy being registered, so they
are removed before anything else runs.
"""

from pathlib import Path
from typing import Tuple

import nibabel as nib
import numpy as np
import torch
from scipy import ndimage

from .config import CT_AIR_HU, ModelConfig
from .transforms import unit_flow_to_voxel

# Everything at or above this HU is a candidate for "inside the patient".
BODY_THRESHOLD_HU = -700.0


def norm_ct(volume: np.ndarray, window: Tuple[float, float]) -> np.ndarray:
    """Clip to a fixed HU window and scale to [0, 1]."""
    lo, hi = window
    return np.clip((volume - lo) / (hi - lo), 0.0, 1.0)


def norm_pet(volume: np.ndarray, suv_max: float) -> np.ndarray:
    """Clip SUV at ``suv_max`` and scale to [0, 1]."""
    return np.clip(volume, 0.0, suv_max) / suv_max


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labels, n = ndimage.label(mask)
    if n == 0:
        return np.zeros_like(mask, dtype=bool)
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    return labels == int(np.argmax(counts))


def _central_region(shape: Tuple[int, int]) -> np.ndarray:
    """The middle 60% of a slice — where the patient is and the bed is not."""
    region = np.zeros(shape, dtype=bool)
    x0, x1 = round(shape[0] * 0.2), round(shape[0] * 0.8)
    y0, y1 = round(shape[1] * 0.2), round(shape[1] * 0.8)
    region[x0:x1, y0:y1] = True
    return region


def _select_body_components(
    slice_mask: np.ndarray, central: np.ndarray, previous: np.ndarray | None
) -> np.ndarray:
    """Pick the connected components of one axial slice that are the patient.

    Three signals separate body from bed and from table padding: overlap with
    the previous slice's selection (anatomy is continuous in z), presence in the
    central region, and how much of the component touches the slice border (the
    bed runs off the edge; a torso does not). A component that overlaps the
    previous slice is taken outright — that is the strongest evidence available
    and it is what keeps the arms attached as they enter the field of view.

    If nothing qualifies, the best-scoring candidate is used rather than
    returning an empty slice, so a mask always exists.
    """
    labels, n = ndimage.label(slice_mask)
    if n == 0:
        return np.zeros_like(slice_mask, dtype=bool)

    support = (
        None
        if previous is None
        else ndimage.binary_dilation(previous, structure=np.ones((9, 9), dtype=bool))
    )

    selected = np.zeros_like(slice_mask, dtype=bool)
    fallback = np.zeros_like(slice_mask, dtype=bool)
    fallback_score = -np.inf

    for index in range(1, n + 1):
        component = labels == index
        coords = np.argwhere(component)
        area = int(coords.shape[0])
        if area < 64:
            continue
        extent = coords.max(axis=0) - coords.min(axis=0) + 1
        # A component thinner than 6 voxels in either direction is a table rail
        # or a cable, not anatomy.
        if int(extent.min()) < 6:
            continue

        central_hits = int(np.logical_and(component, central).sum())
        border_hits = int(
            component[0, :].sum()
            + component[-1, :].sum()
            + component[:, 0].sum()
            + component[:, -1].sum()
        )
        overlap = 0 if support is None else int(np.logical_and(component, support).sum())

        if support is not None and overlap > 0:
            selected |= component
            continue
        if central_hits > 0 and border_hits < 0.35 * max(area, 1):
            selected |= component
            continue

        score = float(
            area + 4 * central_hits + 8 * extent.min() + 6 * overlap - 3 * border_hits
        )
        if score > fallback_score:
            fallback_score, fallback = score, component

    return selected if selected.any() else fallback


def body_mask(ct_hu: np.ndarray) -> np.ndarray:
    """Boolean mask of the patient, excluding the scanner bed.

    Slices are visited outwards from the middle of the volume in both
    directions, each one seeded by the previous slice's result. Starting in the
    middle matters: the torso there is unambiguous, whereas the first and last
    slices may hold only arms or a table edge.
    """
    candidate = ct_hu >= BODY_THRESHOLD_HU
    candidate = ndimage.binary_opening(
        candidate, structure=np.ones((3, 3, 3), dtype=bool)
    )

    mask = np.zeros_like(candidate, dtype=bool)
    central = _central_region(candidate.shape[:2])
    middle = candidate.shape[2] // 2

    for z_range in (range(middle, -1, -1), range(middle + 1, candidate.shape[2])):
        previous = None
        for z in z_range:
            current = _select_body_components(candidate[:, :, z], central, previous)
            mask[:, :, z] = current
            if current.any():
                previous = current

    if not mask.any():
        mask = _largest_component(candidate)
    else:
        mask = _largest_component(ndimage.binary_fill_holes(mask))

    # Close across the slice direction with a flatter structure: the axial
    # spacing is coarser than in-plane, so a symmetric element would bridge
    # further in millimetres along z than it does in x and y.
    mask = ndimage.binary_closing(mask, structure=np.ones((5, 5, 3), dtype=bool))
    mask = ndimage.binary_fill_holes(mask)
    mask = ndimage.binary_dilation(mask, structure=np.ones((3, 3, 3), dtype=bool))
    return mask.astype(bool)


def _load(path: Path) -> np.ndarray:
    return nib.load(str(path)).get_fdata().astype(np.float32)


def load_scan(ct_path: Path, pet_path: Path, cfg: ModelConfig) -> torch.Tensor:
    """One timepoint as a normalised ``(1, 2, H, W, D)`` tensor, CT then PET.

    The CT's body mask is applied to both channels — the PET has no reliable
    body outline of its own, and the two are already on the same grid.
    """
    ct = _load(ct_path)
    mask = body_mask(ct)

    ct = np.where(mask, ct, CT_AIR_HU)
    pet = np.where(mask, _load(pet_path), 0.0)

    channels = np.stack(
        [norm_ct(ct, cfg.ct_window), norm_pet(pet, cfg.pet_suv_max)], axis=0
    )
    return torch.from_numpy(channels[None]).float()


def load_pair(
    fixed_ct: Path,
    fixed_pet: Path,
    moving_ct: Path,
    moving_pet: Path,
    cfg: ModelConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(moving, fixed)``, each ``(1, 2, H, W, D)``.

    Moving is the follow-up scan, fixed the baseline — the direction the
    challenge asks for.
    """
    return (
        load_scan(moving_ct, moving_pet, cfg),
        load_scan(fixed_ct, fixed_pet, cfg),
    )


def save_displacement(flow: torch.Tensor, path: Path) -> None:
    """Write a channel-first unit flow as the submission's displacement field.

    The scorer expects ``(3, H, W, D)`` float32 voxel displacements on the fixed
    grid, with channel *i* belonging to spatial axis *i*. A unit flow orders its
    channels the other way round, hence the reversal. The NIfTI affine is the
    identity because the field is indexed in voxels, not millimetres.
    """
    voxel = unit_flow_to_voxel(flow.permute(0, 2, 3, 4, 1))
    array = voxel[0].permute(3, 0, 1, 2).detach().cpu().numpy()
    array = array.astype(np.float32)[::-1].copy()

    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(array, np.eye(4)), str(path))


def load_displacement(
    path: Path, img_shape: Tuple[int, int, int], device: torch.device
) -> torch.Tensor:
    """Read a saved displacement field back as a channel-last unit flow.

    The inverse of :func:`save_displacement`. A field stored at lower resolution
    than ``img_shape`` is upsampled first, which is what the challenge's scorer
    does with a sub-resolution submission — so a half-resolution field is scored
    on the same grid as a full one.
    """
    array = nib.load(str(path)).get_fdata().astype(np.float32)
    # undo the channel reversal save_displacement applies
    voxel = torch.from_numpy(array[::-1].copy())[None].to(device)

    if tuple(voxel.shape[2:]) != tuple(img_shape):
        voxel = torch.nn.functional.interpolate(
            voxel, size=img_shape, mode="trilinear", align_corners=False
        )

    h, w, d = img_shape
    scale = torch.tensor(
        [(d - 1) / 2.0, (w - 1) / 2.0, (h - 1) / 2.0], device=device
    )
    return voxel.permute(0, 2, 3, 4, 1) / scale
