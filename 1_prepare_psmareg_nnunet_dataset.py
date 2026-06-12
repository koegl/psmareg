"""
Phase 1a: Prepare nnU-Net dataset for PSMA lesion segmentation (Model B).

Creates a nnU-Net-compatible Dataset from the Learn2Reg PSMAReg data using
symlinks only — no files are copied or duplicated.

Each individual scan (baseline + all follow-ups) from train_paired and
train_unpaired subjects becomes one nnU-Net training case with:
  - Channel 0 (_0000): CT image
  - Channel 1 (_0001): PET image
  - Label: PET lesion mask (binary, 0=background, 1=lesion)

Output structure:
    nnUNet_raw/
    └── Dataset001_PSMALesions/
        ├── imagesTr/
        │   ├── PSMA_0000_0000.nii.gz -> <symlink to CT>
        │   ├── PSMA_0000_0001.nii.gz -> <symlink to PET>
        │   ...
        ├── labelsTr/
        │   ├── PSMA_0000.nii.gz -> <symlink to PET lesion mask>
        │   ...
        └── dataset.json

Usage:
    python 1a_prepare_nnunet_dataset.py

Expects psmareg_split.json in the same directory (output of 0_phase_data_split.py).
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

# --- Paths -------------------------------------------------------------------

DATASET_ROOT = Path("/home/iml/fryderyk.koegl/data/PSMAReg_dataset")
NNUNET_RAW_ROOT = Path("/home/iml/fryderyk.koegl/data/PSMAReg_dataset_nnunet")
SPLIT_JSON_PATH = Path(
    "/home/iml/fryderyk.koegl/data/PSMAReg_dataset/psmareg_split.json"
)

DATASET_ID = 1
DATASET_NAME = f"Dataset{DATASET_ID:03d}_PSMALesions"

# -----------------------------------------------------------------------------


def get_all_scans(subject: Dict) -> List[Tuple[Path, Path, Path]]:
    """
    Extract all (ct_path, pet_path, lesion_label_path) tuples for a subject.

    Covers baseline + all follow-up timepoints.

    Args:
        subject: Subject entry from psmareg_split.json.

    Returns:
        List of (ct_path, pet_path, lesion_label_path) tuples, one per timepoint.
    """
    scans: List[Tuple[Path, Path, Path]] = []

    # Baseline
    ct = DATASET_ROOT / subject["Baseline CT"].lstrip("./")
    pet = DATASET_ROOT / subject["Baseline PET"].lstrip("./")
    label = DATASET_ROOT / subject["Baseline PET Label"].lstrip("./")
    scans.append((ct, pet, label))

    # Follow-ups
    fu_idx = 1
    while f"Follow-up {fu_idx:02d} CT" in subject:
        ct = DATASET_ROOT / subject[f"Follow-up {fu_idx:02d} CT"].lstrip("./")
        pet = DATASET_ROOT / subject[f"Follow-up {fu_idx:02d} PET"].lstrip("./")
        label = DATASET_ROOT / subject[f"Follow-up {fu_idx:02d} PET Label"].lstrip("./")
        scans.append((ct, pet, label))
        fu_idx += 1

    return scans


def make_symlink(src: Path, dst: Path) -> None:
    """
    Create a symlink at dst pointing to src.

    Uses absolute paths for the symlink target so the link works regardless
    of the working directory. Skips if the symlink already exists and is valid.

    Args:
        src: Absolute path to the source file.
        dst: Absolute path where the symlink should be created.
    """
    src_abs = src.resolve()
    if not src_abs.exists():
        raise FileNotFoundError(f"Source file not found: {src_abs}")
    if dst.is_symlink():
        if dst.resolve() == src_abs:
            return  # Already correct
        dst.unlink()  # Stale symlink — replace
    os.symlink(src_abs, dst)


def write_dataset_json(dataset_dir: Path, num_training: int) -> None:
    """
    Write the nnU-Net dataset.json to the dataset directory.

    Args:
        dataset_dir: Path to the Dataset directory.
        num_training: Total number of training cases.
    """
    dataset_json = {
        "channel_names": {
            "0": "CT",
            "1": "PET_SUV",
        },
        "labels": {
            "background": 0,
            "lesion": 1,
        },
        "numTraining": num_training,
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "NibabelIOWithReorient",
    }
    out_path = dataset_dir / "dataset.json"
    with open(out_path, "w") as f:
        json.dump(dataset_json, f, indent=2)
    print(f"  Written: {out_path}")


def main() -> None:
    # Load split
    with open(SPLIT_JSON_PATH, "r") as f:
        split = json.load(f)

    train_paired: List[Dict] = split["train_paired"]
    train_unpaired: List[Dict] = split["train_unpaired"]

    print(f"Train paired subjects  : {len(train_paired)}")
    print(f"Train unpaired subjects: {len(train_unpaired)}")

    # Set up output directories
    dataset_dir = NNUNET_RAW_ROOT / DATASET_NAME
    images_dir = dataset_dir / "imagesTr"
    labels_dir = dataset_dir / "labelsTr"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nDataset directory: {dataset_dir}")

    # Collect all scans
    all_scans: List[Tuple[Path, Path, Path]] = []

    for subject in train_paired:
        all_scans.extend(get_all_scans(subject))

    for subject in train_unpaired:
        all_scans.extend(get_all_scans(subject))

    print(f"\nTotal training cases (individual scans): {len(all_scans)}")

    # Create symlinks
    n_created = 0
    missing_files: List[str] = []

    for case_idx, (ct_path, pet_path, label_path) in enumerate(all_scans):
        case_id = f"PSMA_{case_idx:04d}"

        ct_dst = images_dir / f"{case_id}_0000.nii.gz"
        pet_dst = images_dir / f"{case_id}_0001.nii.gz"
        label_dst = labels_dir / f"{case_id}.nii.gz"

        # Check sources exist before creating symlinks
        missing = [str(p) for p in [ct_path, pet_path, label_path] if not p.exists()]
        if missing:
            missing_files.extend(missing)
            continue

        prev_count = n_created
        make_symlink(ct_path, ct_dst)
        make_symlink(pet_path, pet_dst)
        make_symlink(label_path, label_dst)
        n_created += 3

    print(f"Symlinks created : {n_created}")

    if missing_files:
        print(f"\nWARNING: {len(missing_files)} source files not found:")
        for p in missing_files[:10]:
            print(f"  {p}")
        if len(missing_files) > 10:
            print(f"  ... and {len(missing_files) - 10} more")

    # Write dataset.json
    print()
    write_dataset_json(dataset_dir, len(all_scans) - len(missing_files) // 3)

    # Print nnU-Net next steps
    print("\n" + "=" * 60)
    print("Next steps:")
    print("=" * 60)
    print("\n1. Set environment variables (add to ~/.bashrc or run before training):")
    print(f"   export nnUNet_raw={NNUNET_RAW_ROOT}")
    print(f"   export nnUNet_preprocessed={NNUNET_RAW_ROOT.parent}/nnUNet_preprocessed")
    print(f"   export nnUNet_results={NNUNET_RAW_ROOT.parent}/nnUNet_results")
    print("\n2. Verify the dataset:")
    print(f"   nnUNetv2_print_dataset_fingerprint {DATASET_ID}")
    print("\n3. Plan and preprocess:")
    print(f"   nnUNetv2_plan_and_preprocess -d {DATASET_ID} --verify_dataset_integrity")
    print("\n4. Train (fold 0, 3d_fullres config, single GPU):")
    print(f"   nnUNetv2_train {DATASET_ID} 3d_fullres 0 --npz")
    print("\n   Or on SLURM (adjust as needed):")
    print(f"   CUDA_VISIBLE_DEVICES=0 nnUNetv2_train {DATASET_ID} 3d_fullres 0 --npz")


if __name__ == "__main__":
    main()
