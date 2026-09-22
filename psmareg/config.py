"""Configuration shared by training and inference.

The fields below are the ones that define the *network* and the *preprocessing*,
so they must match between the two. Anything that only training needs — loss
weights, augmentation, schedules — lives in the training config instead.
"""

from dataclasses import dataclass
from typing import Tuple


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
