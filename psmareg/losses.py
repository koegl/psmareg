"""The loss terms, in the three groups of the paper.

Registration accuracy (NCC, Dice), PET quantification (MTV, TLG, Jacobian) and
deformation regularization (rigidity, smoothness, NDV). Training and instance
optimization both build their objective from these, with different weights.

Conventions: unit flows are ``(B, 3, D, H, W)`` channel-first; voxel flows are
``(B, D, H, W, 3)``; masks and label maps are ``(B, 1, D, H, W)``.
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from .jacobian import voxel_identity_grid

# The 61 skeletal TotalSegmentator labels the rigidity term constrains:
# vertebrae, sacrum and hips, then ribs and sternum.
BONE_LABEL_VALUES: Tuple[int, ...] = (
    *range(26, 51),
    *range(69, 79),
    *range(91, 117),
)


# --------------------------------------------------------------------------
# Registration accuracy
# --------------------------------------------------------------------------


class NCC(torch.nn.Module):
    """Local normalised cross-correlation over a cubic window.

    Returns the *negative* squared correlation, so it is a loss in [-1, 0].
    Local rather than global because whole-body CT has no single intensity
    relationship: the correlation has to be evaluated per neighbourhood.
    """

    def __init__(self, win: int = 7, eps: float = 1e-5):
        super().__init__()
        self.win = win
        self.eps = eps

    def forward(self, moving: torch.Tensor, fixed: torch.Tensor) -> torch.Tensor:
        window = torch.ones(
            (1, 1) + (self.win,) * 3, device=moving.device, dtype=moving.dtype
        )
        pad = self.win // 2
        n = float(self.win**3)

        def local_sum(x: torch.Tensor) -> torch.Tensor:
            return F.conv3d(x, window, padding=pad)

        sum_m, sum_f = local_sum(moving), local_sum(fixed)
        sum_mm, sum_ff = local_sum(moving * moving), local_sum(fixed * fixed)
        sum_mf = local_sum(moving * fixed)

        mean_m, mean_f = sum_m / n, sum_f / n
        cross = sum_mf - mean_f * sum_m - mean_m * sum_f + mean_m * mean_f * n
        var_m = (sum_mm - 2 * mean_m * sum_m + mean_m * mean_m * n).clamp_min(0.0)
        var_f = (sum_ff - 2 * mean_f * sum_f + mean_f * mean_f * n).clamp_min(0.0)

        cc = (cross * cross) / (var_m * var_f + self.eps)
        return -torch.mean(cc.clamp(max=1.0))


def dice_loss(
    moving_label: torch.Tensor,
    fixed_label: torch.Tensor,
    flow: torch.Tensor,
    grid: torch.Tensor,
    eps: float = 1e-5,
    class_weights: Optional[torch.Tensor] = None,
    chunk_size: int = 16,
) -> Optional[torch.Tensor]:
    """Soft multi-class Dice, differentiable through ``flow``.

    Only classes present in *both* label maps are scored: a structure cropped
    out of one of them gives Dice ≈ 0 with no usable gradient and would only
    drag the mean down.

    The sampling grid is identical for every class, so it is built once and
    shared, and classes are warped in chunks. Warping one class at a time would
    keep a full-resolution tensor alive per class for the backward pass — with
    roughly 117 labels at 192x192x288 that is several GB.

    Returns ``None`` when no class is usable.
    """
    classes = fixed_label.unique()
    classes = classes[classes != 0]
    if classes.numel() == 0:
        return None
    classes = classes[torch.isin(classes, moving_label.unique())]
    if classes.numel() == 0:
        return None

    sample_grid = grid + flow.permute(0, 2, 3, 4, 1)
    dims = (0, 2, 3, 4)

    scores = []
    for start in range(0, classes.numel(), chunk_size):
        chunk = classes[start : start + chunk_size].view(1, -1, 1, 1, 1)
        moving_hot = (moving_label == chunk).float()
        fixed_hot = (fixed_label == chunk).float()
        warped = F.grid_sample(
            moving_hot,
            sample_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        intersection = (warped * fixed_hot).sum(dim=dims)
        cardinality = warped.sum(dim=dims) + fixed_hot.sum(dim=dims)
        scores.append((2.0 * intersection + eps) / (cardinality + eps))

    dice = torch.cat(scores)
    if class_weights is not None:
        weights = class_weights[classes.round().long()]
        weights = weights / (weights.mean() + eps)
        return 1.0 - (weights * dice).sum() / weights.sum()
    return 1.0 - dice.mean()


# --------------------------------------------------------------------------
# PET quantification
# --------------------------------------------------------------------------


def mtv_bias(warped_mask: torch.Tensor, moving_mask: torch.Tensor, eps: float = 1e-5):
    """Relative change in metabolic tumor volume under the transform."""
    volume = moving_mask.sum()
    return torch.abs(warped_mask.sum() - volume) / (volume + eps)


def tlg_bias(
    warped_pet: torch.Tensor,
    warped_mask: torch.Tensor,
    moving_pet: torch.Tensor,
    moving_mask: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Relative change in lesion PET intensity mass (total lesion glycolysis).

    Both the image and the mask are warped, so this sees lesion volume change
    and intensity interpolation together — which is what the scorer measures.
    """
    moving_tlg = (moving_pet * moving_mask).sum()
    warped_tlg = (warped_pet * warped_mask).sum()
    return torch.abs(warped_tlg - moving_tlg) / (moving_tlg + eps)


def masked_jacobian_bias(
    jac_det: torch.Tensor, mask: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    """Mean squared deviation of det(J) from 1 inside ``mask``.

    Per voxel, so unlike :func:`mean_jacobian_bias` it also forbids a lesion
    from compressing on one side while expanding on the other.
    """
    return ((jac_det * mask - mask) ** 2).sum() / (mask.sum() + eps)


def mean_jacobian_bias(
    jac_det: torch.Tensor, mask: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    """Squared deviation of the *mean* det(J) over the mask from 1.

    The net volume ratio of the region. Squared rather than absolute so the
    gradient is smooth through zero.
    """
    mean_det = (jac_det * mask).sum() / (mask.sum() + eps)
    return (mean_det - 1.0) ** 2


def lesion_components(
    mask: torch.Tensor, max_components: int, min_voxels: int
) -> Optional[torch.Tensor]:
    """One-hot stack ``(1, C, D, H, W)`` of the largest lesions, or ``None``.

    6-connectivity: 26-connectivity would merge lesions touching at a single
    corner, hiding exactly the per-lesion error the component terms exist to
    measure. Components below ``min_voxels`` are dropped — their relative bias
    is dominated by interpolation noise — and remain covered by the global
    terms.
    """
    labels, n = ndimage.label(mask[0, 0].detach().cpu().numpy() > 0.5)
    if n == 0:
        return None

    sizes = np.bincount(labels.ravel(), minlength=n + 1)
    sizes[0] = 0
    keep = [i for i in np.argsort(sizes)[::-1] if sizes[i] >= min_voxels][
        :max_components
    ]
    if not keep:
        return None

    stack = np.stack([labels == i for i in keep]).astype(np.float32)
    return torch.from_numpy(stack)[None].to(mask.device)


def mtv_bias_per_component(
    warped: torch.Tensor, moving: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    """Size-weighted mean of the squared per-lesion volume bias.

    The global MTV term is a sum over the whole mask, so a lesion that expands
    cancels one that contracts — on our validation data the per-lesion bias is
    roughly seven times the global one. Squaring per component before
    aggregating removes that cancellation; weighting by size keeps a 30-voxel
    lesion from counting as much as a 3000-voxel one.
    """
    n_moving = moving.sum(dim=(2, 3, 4))
    bias = (warped.sum(dim=(2, 3, 4)) - n_moving) / (n_moving + eps)
    weight = n_moving / (n_moving.sum() + eps)
    return (weight * bias**2).sum()


def tlg_bias_per_component(
    warped_pet: torch.Tensor,
    warped: torch.Tensor,
    moving_pet: torch.Tensor,
    moving: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Size-weighted mean of the absolute per-lesion TLG bias.

    Absolute rather than squared, mirroring the global TLG term: its gradient
    does not vanish as the residual approaches zero, where the squared MTV
    terms go flat.
    """
    moving_tlg = (moving_pet * moving).sum(dim=(2, 3, 4))
    warped_tlg = (warped_pet * warped).sum(dim=(2, 3, 4))
    bias = (warped_tlg - moving_tlg).abs() / (moving_tlg + eps)
    weight = moving_tlg / (moving_tlg.sum() + eps)
    return (weight * bias).sum()


def mean_jacobian_bias_per_component(
    jac_det: torch.Tensor, components: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    """Per-lesion version of :func:`mean_jacobian_bias`."""
    sizes = components.sum(dim=(2, 3, 4))
    mean_det = (jac_det * components).sum(dim=(2, 3, 4)) / (sizes + eps)
    weight = sizes / (sizes.sum() + eps)
    return (weight * (mean_det - 1.0) ** 2).sum()


# --------------------------------------------------------------------------
# Deformation regularization
# --------------------------------------------------------------------------


def smooth_loss(flow: torch.Tensor) -> torch.Tensor:
    """Diffusion regularizer: mean squared spatial gradient of the flow."""
    dd = flow[:, :, 1:] - flow[:, :, :-1]
    dh = flow[:, :, :, 1:] - flow[:, :, :, :-1]
    dw = flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1]
    return (dd.pow(2).mean() + dh.pow(2).mean() + dw.pow(2).mean()) / 3.0


def per_label_rigid_loss(
    flow_voxel: torch.Tensor,
    labels: torch.Tensor,
    label_values: torch.Tensor,
    min_voxels: int = 50,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, dict]:
    """Residual of each structure's own best rigid fit, in voxel².

    For every label independently, the rigid transform (R, t) minimising the
    displacement residual is found in closed form (Kabsch), and each voxel's
    squared distance from that fit is penalised::

        loss_l = mean_i ‖ (p_i + u_i) − (R_l p_i + t_l) ‖²

    Fitting each structure separately is the point: it constrains deformation
    *within* a bone while leaving neighbouring bones free to move relative to
    each other. It also reads no neighbourhood at all, so unlike a
    finite-difference rigidity penalty no stencil can reach into the soft
    tissue around a thin rib.

    (R, t) are detached, which is exact rather than an approximation: they
    minimise the residual, so by the envelope theorem the derivative with
    respect to the displacement is unchanged. Detaching only avoids
    differentiating an SVD.

    Labels with fewer than ``min_voxels`` voxels are skipped as ill-posed. Each
    surviving label contributes its own mean, so a rib counts as much as the
    pelvis.
    """
    work = torch.float32  # the SVD is not safe in reduced precision
    flow_voxel = flow_voxel.to(work)
    b, d, h, w, _ = flow_voxel.shape
    device = flow_voxel.device
    zero = torch.zeros((), device=device, dtype=work)
    empty = {"n_labels": zero, "worst": zero}

    flat_labels = labels.reshape(b, -1).long()
    values = label_values.reshape(-1).long()
    if values.numel() == 0 or flat_labels.numel() == 0:
        return zero, empty

    # Map label value -> compact column index; -1 for anything unconstrained.
    size = int(max(int(flat_labels.max()), int(values.max()))) + 1
    lookup = torch.full((size,), -1, device=device, dtype=torch.long)
    lookup[values] = torch.arange(values.numel(), device=device)
    label_index = lookup[flat_labels.clamp_min(0)]

    n_labels = values.numel()
    batch_offset = torch.arange(b, device=device).view(b, 1).expand_as(label_index)
    keep = (label_index >= 0).reshape(-1)
    if not bool(keep.any()):
        return zero, empty
    group = (label_index + batch_offset * n_labels).reshape(-1)[keep]

    grid = voxel_identity_grid((d, h, w), device, work)
    source = grid.reshape(1, -1, 3).expand(b, -1, -1).reshape(-1, 3)[keep]
    target = source + flow_voxel.reshape(-1, 3)[keep]

    n_groups = b * n_labels
    counts = torch.zeros(n_groups, device=device, dtype=work).index_add_(
        0, group, torch.ones_like(group, dtype=work)
    )
    safe_counts = counts.clamp_min(1.0).unsqueeze(-1)
    source_mean = (
        torch.zeros(n_groups, 3, device=device, dtype=work).index_add_(0, group, source)
        / safe_counts
    )
    target_mean = (
        torch.zeros(n_groups, 3, device=device, dtype=work).index_add_(0, group, target)
        / safe_counts
    )

    # Centre before the outer products. The one-pass identity
    # sum(p qᵀ) − n p̄ q̄ᵀ is algebraically equal but numerically wrong here:
    # p holds absolute voxel indices, so for a small bone far from the origin
    # (centroid ~200, radius ~10) the two terms are ~400x larger than their
    # difference and cancel away most of the mantissa — the fitted rotation
    # then comes out visibly wrong for exactly-rigid input.
    source_centred = source - source_mean[group]
    target_centred = target - target_mean[group]
    covariance = torch.zeros(n_groups, 3, 3, device=device, dtype=work).index_add_(
        0, group, source_centred.unsqueeze(2) * target_centred.unsqueeze(1)
    )

    # Kabsch: for cov = U S Vᵀ the minimiser is R = V diag(1, 1, s) Uᵀ, where
    # s = sign(det(V Uᵀ)) forbids a reflection.
    u, _, vh = torch.linalg.svd(covariance.double())
    v = vh.transpose(-2, -1)
    sign = torch.sign(torch.det(v @ u.transpose(-2, -1)))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    flip = torch.eye(3, device=device, dtype=torch.float64).repeat(n_groups, 1, 1)
    flip[:, 2, 2] = sign
    rotation = (v @ flip @ u.transpose(-2, -1)).to(work).detach()

    # Residual in centred coordinates: with t = q̄ − R p̄ this equals
    # q − (R p + t) exactly, but stays O(the structure's radius).
    source_mean, target_mean = source_mean.detach(), target_mean.detach()
    residual = (target - target_mean[group]) - torch.einsum(
        "gij,gj->gi", rotation[group], source - source_mean[group]
    )
    summed = torch.zeros(n_groups, device=device, dtype=work).index_add_(
        0, group, (residual**2).sum(-1)
    )
    per_label = summed / counts.clamp_min(eps)

    usable = counts >= min_voxels
    if not bool(usable.any()):
        return zero, empty
    kept = per_label[usable]
    return kept.mean(), {"n_labels": usable.sum().to(work), "worst": kept.max()}


def bone_label_tensor(device: torch.device) -> torch.Tensor:
    """:data:`BONE_LABEL_VALUES` as a tensor, for :func:`per_label_rigid_loss`."""
    return torch.tensor(BONE_LABEL_VALUES, dtype=torch.float32, device=device)
