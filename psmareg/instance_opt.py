"""Test-time instance optimization.

The network gives one field per pair in a single forward pass. Instance
optimization then spends a few dozen gradient steps on *that pair alone*,
refining the field against the same objective the network was trained on.

The refinement is parametrised as a residual stationary velocity field,
initialised at zero and exponentiated by scaling and squaring, then added to the
network's flow. So the starting point is exactly the network's answer, every
iterate stays diffeomorphic, and stopping early is always safe.

Which iterate to keep is decided on the metrics the challenge scores, not on the
objective being descended: the two differ because the optimizer needs smooth,
interpolated proxies where the scorer counts whole voxels.
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from . import losses
from .jacobian import jacobian_matrix, non_diff_volume_loss, voxel_identity_grid
from .transforms import unit_flow_to_voxel, warp


@dataclass
class IOConfig:
    """Step count, learning rate and term weights for instance optimization.

    These are separate from the training weights: IO descends the same terms but
    from a much better starting point and for far fewer steps, so the balance
    that works during training is not the one that works here.
    """

    # Fixed step count, used when run_io gets no deadline.
    steps: int = 9
    lr: float = 0.025
    # Under a deadline: the assumed cost of the first step (the slowest, it pays
    # cuDNN autotuning), the margin each predicted step cost is inflated by, and
    # a ceiling that exists only so a broken clock cannot spin forever.
    min_step_seconds: float = 4.0
    step_safety: float = 1.3
    max_steps: int = 1_000_000
    integration_steps: int = 7
    ncc_window: int = 7

    w_ncc: float = 5.0
    w_dice: float = 5.0
    w_non_diff: float = 10.0
    w_smooth: float = 0.0

    w_mtv: float = 400.0
    w_mtv_mean: float = 0.0
    w_tlg: float = 160.0
    w_jacobian_tumor: float = 40.0

    # Per-lesion counterparts of the three terms above. They sit alongside the
    # global ones rather than replacing them: the global term is the metric, and
    # its free cancellation is worth keeping where it is available.
    w_mtv_cc: float = 80.0
    w_tlg_cc: float = 80.0
    w_mtv_mean_cc: float = 150.0
    max_components: int = 8
    min_component_voxels: int = 20

    w_rigidity: float = 1.0
    rigidity_min_voxels: int = 50

    def uses_components(self) -> bool:
        return max(self.w_mtv_cc, self.w_tlg_cc, self.w_mtv_mean_cc) > 0.0


@dataclass
class IOInputs:
    """What IO needs besides the images and the field.

    Every label is optional and each one gates exactly one group of terms, so a
    pair with no labels at all still refines against NCC, smoothness and the
    folding barrier.

    All of these live in the frame the field maps *from* — the original moving
    image, not the affinely aligned one — so the volume terms include the
    affine's own det(A), which is what the scorer sees.
    """

    moving_lesion: Optional[torch.Tensor] = None
    moving_ct_labels: Optional[torch.Tensor] = None
    fixed_ct_labels: Optional[torch.Tensor] = None

    def has_pet(self) -> bool:
        return self.moving_lesion is not None

    def has_ct(self) -> bool:
        return self.moving_ct_labels is not None and self.fixed_ct_labels is not None


def _integrate_voxel_svf(
    velocity: torch.Tensor, identity: torch.Tensor, steps: int
) -> torch.Tensor:
    """Scaling and squaring for a velocity field in *voxel* units.

    The refinement is parametrised in voxels rather than unit coordinates so one
    learning rate means the same thing along every axis, despite the anisotropic
    grid.
    """
    disp = velocity / (2**steps)
    for _ in range(steps):
        shape = disp.shape[2:]
        coords = identity + disp
        # align_corners=True here: these are absolute voxel coordinates, so the
        # endpoints must map to the corner voxel centres exactly.
        normalised = torch.stack(
            [
                2.0 * coords[:, 2] / (shape[2] - 1) - 1.0,
                2.0 * coords[:, 1] / (shape[1] - 1) - 1.0,
                2.0 * coords[:, 0] / (shape[0] - 1) - 1.0,
            ],
            dim=-1,
        )
        disp = disp + torch.nn.functional.grid_sample(
            disp, normalised, mode="bilinear", padding_mode="border", align_corners=True
        )
    return disp


def _apply_refinement(
    base: torch.Tensor,
    velocity: torch.Tensor,
    identity: torch.Tensor,
    shape: Tuple[int, int, int],
    steps: int,
) -> torch.Tensor:
    """``base`` plus the refinement decoded from ``velocity``, as a unit flow."""
    voxel = _integrate_voxel_svf(velocity, identity, steps)
    unit = torch.stack(
        [
            voxel[:, 0] * 2.0 / (shape[0] - 1),
            voxel[:, 1] * 2.0 / (shape[1] - 1),
            voxel[:, 2] * 2.0 / (shape[2] - 1),
        ],
        dim=1,
    )
    # flip(1): the voxel field is ordered (d, h, w), a unit flow (w, h, d).
    return base + unit.flip(1)


def io_objective(
    flow: torch.Tensor,
    moving: torch.Tensor,
    fixed: torch.Tensor,
    inputs: IOInputs,
    grid: torch.Tensor,
    cfg: IOConfig,
    ncc: losses.NCC,
    bone_values: torch.Tensor,
    components: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """The objective IO descends, plus the scored quantities for step selection.

    Returns ``(loss, logs)``. ``logs`` carries the *hard* MTV and TLG — computed
    with a nearest-warped binary mask, the way the scorer counts them — which
    the loss itself cannot use because nearest sampling has no gradient.
    """
    flow_channel_last = flow.permute(0, 2, 3, 4, 1)
    flow_voxel = unit_flow_to_voxel(flow_channel_last)

    warped = warp(moving, flow_channel_last, grid)
    warped_ct, warped_pet = warped[:, 0:1], warped[:, 1:2]

    loss = (
        cfg.w_ncc * ncc(warped_ct, fixed[:, 0:1])
        + cfg.w_non_diff * non_diff_volume_loss(flow_voxel)
        + cfg.w_smooth * losses.smooth_loss(flow)
    )
    logs: Dict[str, float] = {}

    if inputs.has_ct():
        dice = losses.dice_loss(
            inputs.moving_ct_labels, inputs.fixed_ct_labels, flow, grid
        )
        if dice is not None:
            loss = loss + cfg.w_dice * dice
            logs["dice"] = dice.item()

    # Only the PET volume terms read it, and it is a full-resolution 3x3 field.
    jac_det = jacobian_matrix(flow_voxel)[0] if inputs.has_pet() else None

    logs["hard_mtv"] = float("nan")
    logs["hard_tlg"] = float("nan")

    if inputs.has_pet():
        moving_mask = (inputs.moving_lesion == 1).float()
        moving_pet = moving[:, 1:2]
        warped_mask = warp(moving_mask, flow_channel_last, grid)

        mtv = losses.mtv_bias(warped_mask, moving_mask)
        tlg = losses.tlg_bias(warped_pet, warped_mask, moving_pet, moving_mask)

        # det(J) lives on the FIXED grid, so the Jacobian-based volume terms are
        # masked with the lesion in that frame — the warped mask, not the moving
        # one. Under the total field those two sit a whole affine apart.
        # Detached, so the gradient flows through det(J) rather than by sliding
        # the mask onto a region that already has det(J) = 1.
        fixed_mask = warped_mask.detach()
        masked_jac = losses.masked_jacobian_bias(jac_det, fixed_mask)
        mtv_mean = losses.mean_jacobian_bias(jac_det, fixed_mask)

        loss = loss + (
            cfg.w_mtv * mtv**2
            + cfg.w_mtv_mean * mtv_mean
            + cfg.w_jacobian_tumor * masked_jac
            + cfg.w_tlg * tlg
        )
        logs.update(mtv=mtv.item(), tlg=tlg.item(), masked_jac=masked_jac.item())

        if components is not None:
            # The components are disjoint, so warping the whole stack in one
            # grid_sample is identical to warping each mask separately.
            warped_components = warp(components, flow_channel_last, grid)
            loss = loss + (
                cfg.w_mtv_cc * losses.mtv_bias_per_component(warped_components, components)
                + cfg.w_tlg_cc
                * losses.tlg_bias_per_component(
                    warped_pet, warped_components, moving_pet, components
                )
                + cfg.w_mtv_mean_cc
                * losses.mean_jacobian_bias_per_component(
                    jac_det, warped_components.detach()
                )
            )

        with torch.no_grad():
            # MTV as the scorer counts it: nearest-warped, whole voxels. The
            # bilinear mask the optimizer descends sits between the scorer's
            # integer levels, so it is not a quantity that is ever reported.
            hard_mask = (warp(moving_mask, flow_channel_last, grid, nearest=True) > 0.5).float()
            n_moving = moving_mask.sum()
            logs["hard_mtv"] = (
                (hard_mask.sum() - n_moving).abs() / n_moving.clamp_min(1)
            ).item()
            # TLG likewise: nearest for the mask, bilinear for the image.
            moving_tlg = (moving_pet * moving_mask).sum()
            logs["hard_tlg"] = (
                ((warped_pet * hard_mask).sum() - moving_tlg).abs()
                / moving_tlg.clamp_min(1e-5)
            ).item()

    if inputs.has_ct() and cfg.w_rigidity > 0.0:
        rigidity, _ = losses.per_label_rigid_loss(
            flow_voxel,
            inputs.moving_ct_labels,
            bone_values,
            min_voxels=cfg.rigidity_min_voxels,
        )
        loss = loss + cfg.w_rigidity * rigidity
        logs["rigidity"] = rigidity.item()

    return loss, logs


def run_io(
    flow: torch.Tensor,
    moving: torch.Tensor,
    fixed: torch.Tensor,
    inputs: IOInputs,
    grid: torch.Tensor,
    cfg: IOConfig,
    device: torch.device,
    verbose: bool = True,
    deadline: Optional[float] = None,
) -> torch.Tensor:
    """Refine ``flow`` for this pair and return the best iterate.

    ``deadline`` is an absolute ``time.time()`` by which the loop must end. With
    one, steps continue until the next would not fit; without, ``cfg.steps`` run.

    ``flow`` is the total (affine-composed) unit flow. The returned field is the
    best-scoring iterate, which may be the input itself — ``best`` starts as the
    unrefined field, so zero useful steps costs nothing.
    """
    shape = tuple(moving.shape[2:])
    base = flow.detach()

    identity = voxel_identity_grid(shape, device, torch.float32).permute(0, 4, 1, 2, 3)
    velocity = torch.zeros((1, 3) + shape, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([velocity], lr=cfg.lr)

    ncc = losses.NCC(cfg.ncc_window)
    bone_values = losses.bone_label_tensor(device)

    # Computed once: the moving mask does not change during IO, only the field.
    components = None
    if inputs.has_pet() and cfg.uses_components():
        components = losses.lesion_components(
            (inputs.moving_lesion == 1).float(),
            max_components=cfg.max_components,
            min_voxels=cfg.min_component_voxels,
        )

    best_score = float("inf")
    best_flow = base.clone()
    best_step = -1
    history: List[Dict[str, float]] = []

    max_steps = cfg.steps if deadline is None else cfg.max_steps
    next_step_estimate = cfg.min_step_seconds

    for step in range(max_steps):
        if deadline is not None:
            remaining = deadline - time.time()
            if remaining < next_step_estimate * cfg.step_safety:
                if verbose:
                    print(f"  IO stopping after {step} step(s): {remaining:.1f}s left", flush=True)
                break
        step_start = time.time()

        optimizer.zero_grad()
        current = _apply_refinement(
            base, velocity, identity, shape, cfg.integration_steps
        )
        loss, logs = io_objective(
            current, moving, fixed, inputs, grid, cfg, ncc, bone_values, components
        )
        loss.backward()
        optimizer.step()
        # CUDA is asynchronous: without the sync this would time the queueing.
        if velocity.is_cuda:
            torch.cuda.synchronize()
        next_step_estimate = time.time() - step_start

        # Select on what the challenge scores: substitute the hard (whole-voxel)
        # MTV and TLG for the soft proxies inside the objective value. Without
        # this the best step is chosen on quantities nobody reports.
        score = loss.item()
        if not np.isnan(logs["hard_mtv"]):
            score += cfg.w_mtv * (logs["hard_mtv"] ** 2 - logs["mtv"] ** 2)
        if not np.isnan(logs["hard_tlg"]):
            score += cfg.w_tlg * (logs["hard_tlg"] - logs["tlg"])

        if score < best_score:
            best_score, best_flow, best_step = score, current.detach().clone(), step

        history.append({"step": step, "loss": loss.item(), "score": score, **logs})
        if verbose:
            reported = " ".join(
                f"{k}={v:.4f}" for k, v in logs.items() if not np.isnan(v)
            )
            print(f"  IO {step + 1} loss={loss.item():.4f} {reported}", flush=True)

    if verbose:
        print(f"  IO kept step {best_step + 1} (score {best_score:.4f})", flush=True)
    return best_flow
