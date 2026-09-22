"""Validation metrics and the score checkpoints are selected on.

None of this is differentiable and none of it is a loss. These are the
quantities the challenge reports — Dice, HD95, MTV and TLG bias, NDV — measured
on hard labels so they mean the same thing they mean on the leaderboard.
"""

from typing import Dict, Sequence, Tuple

import numpy as np
import torch
from scipy import ndimage

# The composite ranking weights registration accuracy, biomarker preservation
# and deformation regularity in this proportion.
W_ACCURACY, W_BIOMARKER, W_REGULARITY = 0.4, 0.4, 0.2
# Percent NDV at or below this counts as practically diffeomorphic, so it cannot
# separate two otherwise equal fields.
NDV_EQUIVALENCE = 0.01
NDV_REFERENCE = 0.1

CT_LABELS = range(1, 118)


def multilabel_dice(
    predicted: torch.Tensor, target: torch.Tensor, labels: Sequence[int] = CT_LABELS
) -> float:
    """Mean Dice over a fixed label set, counting a missing label as 0.

    Fixed rather than "labels present in this case": the scorer averages over
    the same list for everyone, and skipping absent labels would quietly reward
    cases with less anatomy in view.
    """
    scores = []
    for label in labels:
        p, t = predicted == label, target == label
        total = p.sum() + t.sum()
        scores.append(0.0 if total == 0 else (2.0 * (p & t).sum() / total).item())
    return float(np.mean(scores)) if scores else float("nan")


def _surface_distances(mask: np.ndarray, other: np.ndarray, spacing) -> np.ndarray:
    """Distances from each surface voxel of ``mask`` to the surface of ``other``."""
    surface = mask ^ ndimage.binary_erosion(mask)
    other_surface = other ^ ndimage.binary_erosion(other)
    if not surface.any() or not other_surface.any():
        return np.array([])
    distance = ndimage.distance_transform_edt(~other_surface, sampling=spacing)
    return distance[surface]


def hd95(
    warped: torch.Tensor,
    fixed: torch.Tensor,
    moving: torch.Tensor,
    spacing: Tuple[float, float, float],
    labels: Sequence[int] = CT_LABELS,
) -> float:
    """Mean 95th-percentile Hausdorff distance in mm over the CT labels.

    A label is scored only if it is present in both the fixed and the *original*
    moving map, matching the scorer: a structure the warp pushed out of view
    should count against the result, but one that was never in the moving scan
    should not.

    This is a distance-transform implementation, not the surfel-area one the
    challenge server runs, so absolute values differ slightly from the
    leaderboard. It is used for checkpoint selection and logging, where only the
    ordering between checkpoints matters.
    """
    warped_np = warped[0, 0].round().long().cpu().numpy()
    fixed_np = fixed[0, 0].round().long().cpu().numpy()
    moving_np = moving[0, 0].round().long().cpu().numpy()

    scores = []
    for label in labels:
        fixed_mask = fixed_np == label
        if not fixed_mask.any() or not (moving_np == label).any():
            continue
        warped_mask = warped_np == label
        if not warped_mask.any():
            continue
        both = np.concatenate(
            [
                _surface_distances(fixed_mask, warped_mask, spacing),
                _surface_distances(warped_mask, fixed_mask, spacing),
            ]
        )
        if both.size:
            scores.append(float(np.percentile(both, 95)))
    return float(np.mean(scores)) if scores else float("nan")


def _quality(value: float, reference: float, scale: float) -> float:
    """Map a metric onto (0, 1), reading 0.5 at its reference.

    Without this the arithmetic mean inside a group would be dominated by
    whichever metric happens to be numerically larger — HD95 in millimetres
    against a Dice in [0, 1].
    """
    if not np.isfinite(value):
        return float("nan")
    return float(0.5 * (1.0 + np.tanh(0.5 * (reference - float(value)) / scale)))


def challenge_score(
    cfg,
    dice_loss: float,
    hd95_mm: float,
    mtv_bias: float,
    tlg_bias: float,
    ndv_percent: float,
) -> Dict[str, float]:
    """Surrogate of the official ranking, for checkpoint selection.

    Mirrors its structure — a weighted geometric mean of the three groups —
    with the per-metric significance tests the server runs replaced by the
    qualities above, which a single run can actually compute.

    Higher is better, unlike every loss in this codebase. A group with no finite
    term yields NaN, which never wins a ``>`` comparison, so a failed evaluation
    round cannot promote a checkpoint.
    """

    def group(*qualities: float) -> float:
        finite = [q for q in qualities if np.isfinite(q)]
        return float(np.mean(finite)) if finite else float("nan")

    accuracy = group(
        _quality(dice_loss, cfg.sel_ref_dice, cfg.sel_scale_dice),
        _quality(hd95_mm, cfg.sel_ref_hd95, cfg.sel_scale_hd95),
    )
    biomarker = group(
        _quality(mtv_bias, cfg.sel_ref_mtv, cfg.sel_scale_mtv),
        _quality(tlg_bias, cfg.sel_ref_tlg, cfg.sel_scale_tlg),
    )
    regularity = group(
        _quality(
            max(float(ndv_percent) - NDV_EQUIVALENCE, 0.0),
            NDV_REFERENCE,
            cfg.sel_scale_ndv,
        )
    )

    final = 100.0 * (
        accuracy**W_ACCURACY * biomarker**W_BIOMARKER * regularity**W_REGULARITY
    )
    return {
        "accuracy": accuracy,
        "biomarker": biomarker,
        "regularity": regularity,
        "final": final,
    }
