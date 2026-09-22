"""Jacobian and folding quantities of a deformation field.

Everything here takes a displacement field in *voxel units* with shape
``(B, D, H, W, 3)``, where component ``c`` is the displacement along spatial
axis ``c`` — the same convention as the challenge scorer.

Two different views of folding live here and they are not interchangeable:

* :func:`non_diff_volume_loss` reproduces the challenge's non-diffeomorphic
  *volume* metric differentiably, by decomposing each voxel into tetrahedra and
  accumulating the negative part of their determinants. Minimising it minimises
  the scored NDV.
* :func:`jacobian_matrix` is the plain central-difference Jacobian, which the
  volume-preservation and rigidity terms need.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

# Per-axis forward ('+') or backward ('-') differences. The eight combinations
# are the eight corners of the voxel, matching the scorer's '+x+y+z' … '-x-y-z'.
_CORNER_SCHEMES = [
    (d, h, w) for d in "+-" for h in "+-" for w in "+-"
]

_INNER = slice(1, -1)


def voxel_identity_grid(
    shape: Tuple[int, int, int], device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Identity coordinates in voxels, ``(1, D, H, W, 3)``, component c along axis c."""
    axes = [torch.arange(n, device=device, dtype=dtype) for n in shape]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).unsqueeze(0)


def _diff_along(transform: torch.Tensor, dim: int, kind: str) -> torch.Tensor:
    """Finite difference along one spatial axis, evaluated on the interior.

    ``kind`` is '0' for central, '+' forward, '-' backward. Every output is
    cropped to ``[1:-1]`` on all three spatial axes so the schemes can be
    combined voxel-for-voxel.
    """
    slices_plus = [slice(None), _INNER, _INNER, _INNER, slice(None)]
    slices_minus = list(slices_plus)
    slices_plus[dim] = slice(2, None)
    slices_minus[dim] = slice(None, -2)

    plus = transform[tuple(slices_plus)]
    minus = transform[tuple(slices_minus)]
    if kind == "0":
        return (plus - minus) / 2
    middle = transform[:, _INNER, _INNER, _INNER, :]
    return plus - middle if kind == "+" else middle - minus


def _det3(row0: torch.Tensor, row1: torch.Tensor, row2: torch.Tensor) -> torch.Tensor:
    """Determinant of the 3x3 matrix with these rows; the scalar triple product."""
    a0, a1, a2 = row0.unbind(-1)
    b0, b1, b2 = row1.unbind(-1)
    c0, c1, c2 = row2.unbind(-1)
    return (
        a0 * (b1 * c2 - b2 * c1)
        - a1 * (b0 * c2 - b2 * c0)
        + a2 * (b0 * c1 - b1 * c0)
    )


def _scheme_determinants(transform: torch.Tensor) -> List[torch.Tensor]:
    """Per-voxel determinants of the ten tetrahedral schemes.

    Eight corner schemes plus two "Jstar" schemes built from diagonal
    differences. Ten rather than one because a voxel can fold in a way a single
    difference scheme cannot see: the scorer counts the negative volume of every
    tetrahedron, so the loss has to as well.
    """
    dets = [
        _det3(
            _diff_along(transform, 1, kd),
            _diff_along(transform, 2, kh),
            _diff_along(transform, 3, kw),
        )
        for kd, kh, kw in _CORNER_SCHEMES
    ]

    centre = transform[:, _INNER, _INNER, _INNER, :]
    # Jstar_1, backward diagonals: the (D,H), (D,W) and (H,W) planes.
    dets.append(
        _det3(
            transform[:, :-2, :-2, _INNER, :] - centre,
            transform[:, :-2, _INNER, :-2, :] - centre,
            transform[:, _INNER, :-2, :-2, :] - centre,
        )
    )
    # Jstar_2, forward diagonals. The second and third planes are swapped
    # relative to Jstar_1 — that ordering is the scorer's, not a typo.
    dets.append(
        _det3(
            transform[:, 2:, 2:, _INNER, :] - centre,
            transform[:, _INNER, 2:, 2:, :] - centre,
            transform[:, 2:, _INNER, 2:, :] - centre,
        )
    )
    return dets


def _interior(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Crop a mask to the interior the difference schemes are defined on."""
    if mask is None:
        return None
    if mask.dim() == 5:
        mask = mask[:, 0]
    return mask[:, _INNER, _INNER, _INNER]


def non_diff_volume_loss(
    flow_voxel: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Non-diffeomorphic volume as a percentage, differentiably.

    ``min(det, 0)`` is written ``-relu(-det)`` so the gradient reaches exactly
    the folded tetrahedra and nothing else. With a body mask, folding is both
    counted and normalised inside it, matching the scorer.
    """
    transform = flow_voxel + voxel_identity_grid(
        flow_voxel.shape[1:4], flow_voxel.device, flow_voxel.dtype
    )
    interior = _interior(mask)

    volume = flow_voxel.new_zeros(())
    for det in _scheme_determinants(transform):
        negative = torch.relu(-det)
        if interior is not None:
            negative = negative * interior
        volume = volume + negative.sum()
    # Each voxel is six tetrahedra, each counted with weight 1/2.
    volume = volume * (0.5 / 6.0)

    if interior is not None:
        total = interior.sum()
    else:
        b, d, h, w = flow_voxel.shape[0], *flow_voxel.shape[1:4]
        total = flow_voxel.new_tensor(b * (d - 2) * (h - 2) * (w - 2))
    return volume / (total + eps) * 100.0


def jacobian_matrix(flow_voxel: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Central-difference Jacobian of the deformation.

    Returns ``(det, jac)``: the determinant as ``(B, 1, D, H, W)`` with the
    boundary padded with 1.0 (an unfolded value, so padding cannot be mistaken
    for a defect), and ``J = I + du/dx`` on the interior as
    ``(B, 3, 3, Di, Hi, Wi)`` indexed ``[:, axis, component]``.
    """
    transform = flow_voxel + voxel_identity_grid(
        flow_voxel.shape[1:4], flow_voxel.device, flow_voxel.dtype
    )
    rows = [_diff_along(transform, dim, "0") for dim in (1, 2, 3)]

    det = _det3(*rows)
    det = F.pad(det, (1, 1, 1, 1, 1, 1), value=1.0).unsqueeze(1)
    jac = torch.stack(rows, dim=1).permute(0, 1, 5, 2, 3, 4)
    return det, jac
