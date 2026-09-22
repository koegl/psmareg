"""Longitudinal pairs, with the augmentation the training schedule uses.

Each item is one (moving, fixed) pair: a follow-up scan registered onto the
patient's baseline. Augmentation is applied here, and the parameters it chose
travel with the item — the affine field is cached per pair and has to be
replayed through the same flip and crops before it can be composed with the
network's output.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import torch
from torch.utils import data as torch_data

from .config import CT_AIR_HU, ModelConfig
from .data import body_mask, norm_ct, norm_pet

# The moving image is a follow-up; the fixed image is always the baseline.
BASELINE = "00"


@dataclass
class Augmentation:
    """What the augmenter chose for one item, so the affine can follow it."""

    flipped: bool = False
    crop_head: int = 0
    crop_feet: int = 0
    crop_head_moving: int = 0
    crop_feet_moving: int = 0
    crop_head_fixed: int = 0
    crop_feet_fixed: int = 0

    def as_dict(self) -> Dict[str, int]:
        return {
            "flipped": int(self.flipped),
            "crop_head": self.crop_head,
            "crop_feet": self.crop_feet,
            "crop_head_fixed": self.crop_head_fixed,
            "crop_feet_fixed": self.crop_feet_fixed,
        }


def list_timepoints(data_dir: Path) -> Dict[str, List[str]]:
    """Map each case to its sorted timepoints, discovered from the CT files."""
    timepoints: Dict[str, List[str]] = {}
    for path in (data_dir / "imagesTr").glob("PSMARegPSMA_*_0000_*.nii.gz"):
        case, _, timepoint = path.name.removesuffix(".nii.gz").split("_")[1:4]
        timepoints.setdefault(case, []).append(timepoint)
    return {case: sorted(tps) for case, tps in sorted(timepoints.items())}


def build_pairs(
    timepoints: Dict[str, List[str]], cases: Optional[Sequence[str]] = None
) -> List[Tuple[str, str]]:
    """Every (case, follow-up) pair, each registered against that case's baseline."""
    selected = sorted(cases) if cases is not None else sorted(timepoints)
    return [
        (case, tp)
        for case in selected
        for tp in timepoints.get(case, [])
        if tp != BASELINE and BASELINE in timepoints.get(case, [])
    ]


def patient_split(
    data_dir: Path, split_path: Path, val_fraction: float = 0.2, seed: int = 0
) -> Tuple[List[str], List[str]]:
    """Train/validation case ids, split by *patient* and cached to disk.

    Patient-wise, not pair-wise: two timepoints of the same patient share
    anatomy, so splitting on pairs would leak the validation anatomy into
    training. The split is written out so every level of the pyramid — and every
    later run — trains and validates on the same cases.
    """
    if split_path.exists():
        split = json.loads(split_path.read_text())
        return split["train"], split["val"]

    eligible = sorted(
        case for case, tps in list_timepoints(data_dir).items() if len(tps) >= 2
    )
    shuffled = list(eligible)
    np.random.RandomState(seed).shuffle(shuffled)
    n_val = int(round(len(shuffled) * val_fraction))
    val, train = sorted(shuffled[:n_val]), sorted(shuffled[n_val:])

    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps({"train": train, "val": val}, indent=2))
    return train, val


def _flip(volumes: Dict[str, torch.Tensor], keys: Sequence[str]) -> None:
    for key in keys:
        volumes[key] = torch.flip(volumes[key], dims=[1])


def _crop_z(
    volumes: Dict[str, torch.Tensor], keys: Sequence[str], head: int, feet: int
) -> None:
    """Remove axial slices from either end and pad the volume back to size.

    Padded rather than resized: the network's input shape is fixed, and a scan
    that simply does not cover those slices is exactly the field-of-view
    mismatch this is imitating.
    """
    if head == 0 and feet == 0:
        return
    for key in keys:
        volume = volumes[key]
        depth = volume.shape[-1]
        cropped = volume[..., head : depth - feet if feet else depth]
        pad = lambda n: torch.zeros(
            *volume.shape[:-1], n, dtype=volume.dtype
        )
        volumes[key] = torch.cat([pad(head), cropped, pad(feet)], dim=-1)


class RegistrationPairs(torch_data.Dataset):
    """Longitudinal PSMA PET/CT pairs with their labels.

    Yields a dict of ``(1, H, W, D)`` tensors: the two images stacked as
    ``moving``/``fixed`` with CT in channel 0 and PET in channel 1, the CT organ
    and PET lesion labels of both, the fixed body mask (which bounds the NDV
    term), and the augmentation parameters.

    Volumes are loaded per item rather than cached in RAM: one pair is about
    180 MB in float32 and the training set is over a hundred of them.
    """

    def __init__(
        self,
        data_dir: Path,
        cases: Sequence[str],
        cfg: ModelConfig,
        augment: bool = False,
        flip_prob: float = 0.5,
        ct_shift_range: Tuple[float, float] = (-0.02, 0.02),
        ct_scale_range: Tuple[float, float] = (0.9, 1.1),
        pet_scale_range: Tuple[float, float] = (0.85, 1.15),
        max_crop_z: int = 40,
        max_crop_z_asymmetric: int = 10,
    ):
        self.data_dir = data_dir
        self.cfg = cfg
        self.augment = augment
        self.flip_prob = flip_prob
        self.ct_shift_range = ct_shift_range
        self.ct_scale_range = ct_scale_range
        self.pet_scale_range = pet_scale_range
        self.max_crop_z = max_crop_z
        self.max_crop_z_asymmetric = max_crop_z_asymmetric
        self.pairs = build_pairs(list_timepoints(data_dir), cases)

    def __len__(self) -> int:
        return len(self.pairs)

    def image_path(self, case: str, channel: str, timepoint: str) -> Path:
        return self.data_dir / "imagesTr" / f"PSMARegPSMA_{case}_{channel}_{timepoint}.nii.gz"

    def label_path(self, case: str, channel: str, timepoint: str) -> Path:
        return self.data_dir / "labelsTr" / f"PSMARegPSMA_{case}_{channel}_{timepoint}.nii.gz"

    def _load_side(self, case: str, timepoint: str) -> Dict[str, torch.Tensor]:
        """One timepoint: masked CT and PET, plus its two label maps."""
        ct = nib.load(str(self.image_path(case, "0000", timepoint))).get_fdata()
        pet = nib.load(str(self.image_path(case, "0001", timepoint))).get_fdata()
        mask = body_mask(ct.astype(np.float32))

        as_tensor = lambda a: torch.from_numpy(np.asarray(a)).unsqueeze(0)
        return {
            "ct": as_tensor(norm_ct(np.where(mask, ct, CT_AIR_HU), self.cfg.ct_window)).float(),
            "pet": as_tensor(norm_pet(np.where(mask, pet, 0.0), self.cfg.pet_suv_max)).float(),
            "label_ct": as_tensor(
                nib.load(str(self.label_path(case, "0000", timepoint))).get_fdata()
            ).float(),
            "label_pet": as_tensor(
                nib.load(str(self.label_path(case, "0001", timepoint))).get_fdata()
            ).float(),
            "body_mask": as_tensor(mask.astype(np.float32)).float(),
        }

    def _augment(self, volumes: Dict[str, torch.Tensor]) -> Augmentation:
        """Flip, crop and jitter intensities; report what was chosen."""
        chosen = Augmentation()
        moving_keys = [k for k in volumes if k.startswith("moving")]
        fixed_keys = [k for k in volumes if k.startswith("fixed")]

        if np.random.random() < self.flip_prob:
            _flip(volumes, list(volumes))
            chosen.flipped = True

        # A shared crop stands for the two scans covering a different part of
        # the body than the training grid assumes...
        chosen.crop_head = int(np.random.randint(0, self.max_crop_z + 1))
        chosen.crop_feet = int(np.random.randint(0, self.max_crop_z + 1))
        _crop_z(volumes, list(volumes), chosen.crop_head, chosen.crop_feet)

        # ...and a smaller independent crop for the two sessions differing from
        # each other, which is the mismatch the network actually has to absorb.
        limit = self.max_crop_z_asymmetric + 1
        chosen.crop_head_moving = int(np.random.randint(0, limit))
        chosen.crop_feet_moving = int(np.random.randint(0, limit))
        chosen.crop_head_fixed = int(np.random.randint(0, limit))
        chosen.crop_feet_fixed = int(np.random.randint(0, limit))
        _crop_z(volumes, moving_keys, chosen.crop_head_moving, chosen.crop_feet_moving)
        _crop_z(volumes, fixed_keys, chosen.crop_head_fixed, chosen.crop_feet_fixed)

        # Intensity jitter is applied to both scans independently: scanner
        # calibration and reconstruction differ between sessions.
        for key in ("moving_ct", "fixed_ct"):
            shift = np.random.uniform(*self.ct_shift_range)
            scale = np.random.uniform(*self.ct_scale_range)
            volumes[key] = volumes[key] * scale + shift
        for key in ("moving_pet", "fixed_pet"):
            volumes[key] = volumes[key] * np.random.uniform(*self.pet_scale_range)

        return chosen

    def __getitem__(self, index: int) -> dict:
        case, timepoint = self.pairs[index]
        moving = self._load_side(case, timepoint)
        fixed = self._load_side(case, BASELINE)

        volumes = {f"moving_{k}": v for k, v in moving.items()}
        volumes.update({f"fixed_{k}": v for k, v in fixed.items()})
        # The moving body mask is unused: NDV is measured in the fixed frame.
        volumes.pop("moving_body_mask")

        chosen = self._augment(volumes) if self.augment else Augmentation()

        return {
            "moving": torch.cat([volumes["moving_ct"], volumes["moving_pet"]], dim=0),
            "fixed": torch.cat([volumes["fixed_ct"], volumes["fixed_pet"]], dim=0),
            "moving_label_ct": volumes["moving_label_ct"],
            "moving_label_pet": volumes["moving_label_pet"],
            "fixed_label_ct": volumes["fixed_label_ct"],
            "fixed_label_pet": volumes["fixed_label_pet"],
            "fixed_body_mask": volumes["fixed_body_mask"],
            "case": case,
            "timepoint": timepoint,
            **chosen.as_dict(),
        }
