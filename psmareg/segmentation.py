"""Label sources for instance optimization, when masks are not supplied.

IO needs two things the images alone do not provide: a PET lesion mask in the
moving frame, and CT organ labels in both frames. On the challenge's own
training split these ship with the data; anywhere else they have to be
predicted.

Both dependencies are imported lazily. Registration without IO needs neither,
and between them they pull in most of a second deep-learning stack.
"""

from pathlib import Path
from typing import Optional, Sequence

import nibabel as nib
import numpy as np
import torch

# TotalSegmentator's "total" task in fast mode produces the 117-label set the
# rigidity and Dice terms are written against.
TOTALSEG_TASK = "total"


def _as_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(array.astype(np.float32))[None, None].to(device)


def load_mask(path: Path, device: torch.device) -> torch.Tensor:
    """Read a label map or mask from disk as ``(1, 1, D, H, W)``."""
    return _as_tensor(nib.load(str(path)).get_fdata(), device)


def _patch_trainer_lookup() -> None:
    """Resolve the trainer name our checkpoint was trained with.

    nnU-Net records the trainer in the checkpoint and refuses to load one it
    cannot import. The lesion model was trained with ``nnUNetTrainer_PGPSplus``,
    which grows the patch size over the course of training and lives only in the
    training fork — but it overrides nothing that matters at inference, so the
    architecture it asks for is the stock one.

    Rather than requiring that fork here, an unresolvable trainer name falls
    back to the base trainer. A trainer that *did* change the architecture would
    then fail loudly at ``load_state_dict`` rather than silently building the
    wrong network.

    nnU-Net renamed this lookup between releases, so whichever symbol the
    installed version imported into its predictor module is the one wrapped.
    """
    from nnunetv2.inference import predict_from_raw_data
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

    for symbol in ("recursive_find_trainer_class_by_name", "recursive_find_python_class"):
        original = getattr(predict_from_raw_data, symbol, None)
        if original is None:
            continue

        def fallback(*args, _original=original, **kwargs):
            try:
                found = _original(*args, **kwargs)
            except Exception:
                found = None
            return found if found is not None else nnUNetTrainer

        setattr(predict_from_raw_data, symbol, fallback)
        return

    raise RuntimeError(
        "could not find nnU-Net's trainer lookup to patch; "
        "pass --moving-lesion with a precomputed mask instead"
    )


def segment_lesion(
    ct_path: Path,
    pet_path: Path,
    model_dir: Path,
    device: torch.device,
    folds: Sequence[int] = (0,),
    checkpoint: str = "checkpoint_final.pth",
    use_mirroring: bool = False,
) -> np.ndarray:
    """PET lesion mask for one timepoint, from an nnU-Net model directory.

    ``model_dir`` is an nnU-Net results folder — the one holding ``plans.json``,
    ``dataset.json`` and ``fold_N/``. The model takes CT in HU and PET in SUV as
    its two channels, unnormalised: nnU-Net applies its own preprocessing and
    resampling from the plans, and returns the mask on the input grid.

    One fold by default rather than the full five-fold ensemble, which costs
    five times the runtime. Pass ``folds=(0, 1, 2, 3, 4)`` to reproduce the
    published configuration.

    Mirror test-time augmentation is off by default: eight times the tile
    forward passes for a small gain.
    """
    _patch_trainer_lookup()
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=use_mirroring,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        str(model_dir), use_folds=tuple(folds), checkpoint_name=checkpoint
    )

    ct_image = nib.load(str(ct_path))
    stacked = np.stack(
        [
            ct_image.get_fdata().astype(np.float32),
            nib.load(str(pet_path)).get_fdata().astype(np.float32),
        ],
        axis=0,
    )
    properties = {"spacing": ct_image.header.get_zooms()[:3]}
    return predictor.predict_single_npy_array(stacked, properties, None, None, False)


def segment_organs(ct_path: Path, output_path: Path, fast: bool = True) -> np.ndarray:
    """CT organ labels from TotalSegmentator.

    ``fast=True`` runs the 3 mm model. That is the resolution the submitted
    container uses — at full resolution a single scan takes longer than the
    challenge's entire per-pair budget — and the labels feed a Dice term and a
    per-structure rigid fit, neither of which resolves detail the 3 mm model
    misses.
    """
    from totalsegmentator.python_api import totalsegmentator

    output_path.parent.mkdir(parents=True, exist_ok=True)
    totalsegmentator(
        str(ct_path),
        str(output_path),
        task=TOTALSEG_TASK,
        fast=fast,
        body_seg=True,
        ml=True,  # one multi-label file rather than a file per structure
        quiet=True,
    )
    return nib.load(str(output_path)).get_fdata().astype(np.float32)


def prepare_io_labels(
    moving_ct: Path,
    moving_pet: Path,
    fixed_ct: Path,
    device: torch.device,
    work_dir: Path,
    lesion_path: Optional[Path] = None,
    moving_labels_path: Optional[Path] = None,
    fixed_labels_path: Optional[Path] = None,
    lesion_model: Optional[Path] = None,
    lesion_folds: Sequence[int] = (0,),
    segment_ct: bool = True,
) -> tuple:
    """Collect the three label maps IO wants, predicting whatever is missing.

    Returns ``(moving_lesion, moving_ct_labels, fixed_ct_labels)``, any of which
    may be ``None`` — IO drops the corresponding terms rather than failing.

    Prediction failures are reported and swallowed for the same reason: a field
    refined without one term still scores, whereas a crashed run produces no
    field at all.
    """
    lesion = ct_labels_moving = ct_labels_fixed = None

    if lesion_path is not None:
        lesion = load_mask(lesion_path, device)
    elif lesion_model is not None:
        try:
            print("segmenting PET lesions (nnU-Net)...", flush=True)
            mask = segment_lesion(
                moving_ct, moving_pet, lesion_model, device, folds=lesion_folds
            )
            lesion = _as_tensor(mask, device)
            print(f"  {int(mask.sum())} lesion voxels", flush=True)
        except Exception as error:
            print(f"WARNING: lesion segmentation failed ({error}); PET terms are off", flush=True)

    if moving_labels_path is not None and fixed_labels_path is not None:
        ct_labels_moving = load_mask(moving_labels_path, device)
        ct_labels_fixed = load_mask(fixed_labels_path, device)
    elif segment_ct:
        try:
            print("segmenting CT organs (TotalSegmentator)...", flush=True)
            work_dir.mkdir(parents=True, exist_ok=True)
            ct_labels_moving = _as_tensor(
                segment_organs(moving_ct, work_dir / "moving_labels.nii.gz"), device
            )
            ct_labels_fixed = _as_tensor(
                segment_organs(fixed_ct, work_dir / "fixed_labels.nii.gz"), device
            )
        except Exception as error:
            print(f"WARNING: CT segmentation failed ({error}); Dice and rigidity are off", flush=True)
            ct_labels_moving = ct_labels_fixed = None

    return lesion, ct_labels_moving, ct_labels_fixed
