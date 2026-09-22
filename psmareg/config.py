"""Configuration shared by training and inference.

The fields below are the ones that define the *network* and the *preprocessing*,
so they must match between the two. Anything that only training needs — loss
weights, augmentation, schedules — lives in the training config instead.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class ModelConfig:
    """Everything needed to rebuild the network and preprocess a pair.

    The defaults are the submitted configuration; they have to match the
    checkpoint or ``load_state_dict`` rejects it.
    """

    # --- geometry ---------------------------------------------------------
    # The challenge ships every scan on this grid, so there is no resampling
    # anywhere in the pipeline.
    img_shape: Tuple[int, int, int] = (192, 192, 288)
    spacing: Tuple[float, float, float] = (2.7344, 2.7344, 3.27)

    # --- network ----------------------------------------------------------
    # 4 = fixed CT, fixed PET, moving CT, moving PET.
    in_channel: int = 4
    # 3 velocity components.
    n_classes: int = 3
    start_channel: int = 7
    n_resblocks: int = 5
    resblock_expansion: int = 1
    # Caps the velocity the output layer (a Softsign) can emit, in unit-grid
    # coordinates, before scaling and squaring.
    range_flow: float = 0.4
    # Scaling-and-squaring steps used to exponentiate the velocity field.
    integration_steps: int = 7

    # --- intensity normalisation -----------------------------------------
    # CT_AIR_HU doubles as the window's lower bound and as the body-mask fill,
    # so masked-out voxels land on exactly 0 after normalisation.
    ct_window: Tuple[float, float] = (-1000.0, 1500.0)
    pet_suv_max: float = 20.0

    # --- affine pre-registration -----------------------------------------
    # ANTs runs on half-resolution CT windowed to this range; wider than the
    # network's window because the affine stage only needs bone and body
    # outline, and the soft-tissue detail above 1000 HU is noise to it.
    affine_ct_window: Tuple[float, float] = (-1000.0, 1000.0)
    affine_downsample: int = 2

    @property
    def img_shape_2(self) -> Tuple[int, int, int]:
        return tuple(s // 2 for s in self.img_shape)

    @property
    def img_shape_4(self) -> Tuple[int, int, int]:
        return tuple(s // 4 for s in self.img_shape)


CT_AIR_HU = -1000.0


@dataclass
class TrainConfig:
    """Schedule, loss weights and augmentation for training.

    Inference does not read any of this; it only needs :class:`ModelConfig`,
    which this one carries so the two cannot drift apart.
    """

    model: ModelConfig = field(default_factory=ModelConfig)

    # --- schedule ---------------------------------------------------------
    # Levels are trained in order, each initialised from the previous one. The
    # full-resolution level gets the most steps: it is the only one carrying the
    # complete objective.
    steps: Dict[int, int] = field(
        default_factory=lambda: {1: 60_000, 2: 60_000, 3: 120_000}
    )
    lr: Dict[int, float] = field(
        default_factory=lambda: {1: 3e-4, 2: 2e-4, 3: 2.5e-4}
    )
    batch_size: int = 1
    # One pair per step is all that fits at full resolution, so the effective
    # batch comes from accumulation instead.
    accumulation_steps: int = 4
    # Linear 0 -> lr ramp at the start of each level, in epochs. Kept below
    # unfreeze_epoch so the fresh head is warm before the level below it starts
    # moving.
    warmup_epochs: float = 5.0
    # The preceding level is frozen for this long, then fine-tuned jointly.
    unfreeze_epoch: int = 10
    val_interval: int = 2
    num_workers: int = 8

    # --- loss weights -----------------------------------------------------
    # Similarity and regularization apply at every level.
    w_ncc: float = 5.0
    w_smooth: float = 10.0
    w_non_diff: float = 10_000.0
    # Dice grows with resolution: at quarter resolution most structures are a
    # few voxels across, so the term is mostly interpolation.
    w_dice: Dict[int, float] = field(default_factory=lambda: {1: 3.0, 2: 4.0, 3: 5.0})

    # PET quantification and rigidity, full resolution only — see
    # `level_weights`.
    w_mtv: float = 20.0
    w_mtv_mean: float = 0.5
    w_tlg: float = 5.0
    w_jacobian_tumor: float = 5.0
    w_rigidity: float = 0.2
    rigidity_min_voxels: int = 50

    # NCC window per level, and how many scales the multi-resolution sum has.
    ncc_window: Dict[int, int] = field(default_factory=lambda: {1: 5, 2: 7, 3: 7})

    # --- augmentation -----------------------------------------------------
    # Chosen to mimic longitudinal variability rather than generic robustness.
    augment: bool = True
    flip_prob: float = 0.5
    # In normalised [0, 1] CT space, so about +-50 HU.
    ct_shift_range: Tuple[float, float] = (-0.02, 0.02)
    ct_scale_range: Tuple[float, float] = (0.9, 1.1)
    # Wider than CT and with no shift: PET uptake scales with dose and uptake
    # time, it does not acquire an offset.
    pet_scale_range: Tuple[float, float] = (0.85, 1.15)
    # Field-of-view mismatch: a shared crop of both scans, then a smaller
    # independent crop of each, which is what differs between two sessions.
    max_crop_z: int = 40
    max_crop_z_asymmetric: int = 10

    # --- checkpoint selection --------------------------------------------
    # Reference values and scales of the composite score (see
    # psmareg.metrics.challenge_score). Each metric reads 0.5 at its reference,
    # so no single one dominates the mean inside its group.
    sel_ref_dice: float = 0.278
    sel_ref_hd95: float = 8.3319
    sel_ref_mtv: float = 0.044138
    sel_ref_tlg: float = 0.045699
    sel_scale_dice: float = 0.0090181
    sel_scale_hd95: float = 0.13427
    sel_scale_mtv: float = 0.0051981
    sel_scale_tlg: float = 0.0050091
    sel_scale_ndv: float = 0.005

    def level_weights(self, level: int) -> Dict[str, float]:
        """Loss weights for one pyramid level.

        The two coarser levels drop the PET and bone terms entirely: at half and
        quarter resolution a lesion or a rib spans only a few voxels, so volume
        ratios and per-structure rigid fits measure interpolation rather than
        anatomy.
        """
        weights = {
            "ncc": self.w_ncc,
            "dice": self.w_dice[level],
            "smooth": self.w_smooth,
            "non_diff": self.w_non_diff,
            "mtv": 0.0,
            "mtv_mean": 0.0,
            "tlg": 0.0,
            "jacobian_tumor": 0.0,
            "rigidity": 0.0,
        }
        if level == 3:
            weights.update(
                mtv=self.w_mtv,
                mtv_mean=self.w_mtv_mean,
                tlg=self.w_tlg,
                jacobian_tumor=self.w_jacobian_tumor,
                rigidity=self.w_rigidity,
            )
        return weights
