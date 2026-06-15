"""
Prepare nnU-Net Dataset002_DEEPPSMALesions from the DEEP-PSMA Zenodo dataset.
https://zenodo.org/records/15281784

For each case (100 PSMA + 100 FDG = 200 total):
  - Resample CT to PET space (CT is 512x512, PET is 192x192)
  - Write CT (_0000) and PET (_0001) as NIfTI files
  - Copy TTB as the lesion label

Output structure (nnU-Net raw):
    Dataset002_DEEPPSMALesions/
        imagesTr/
            DEEPPSMA_0000_0000.nii.gz   <- CT
            DEEPPSMA_0000_0001.nii.gz   <- PET
            ...
        labelsTr/
            DEEPPSMA_0000.nii.gz        <- TTB lesion mask
            ...
        dataset.json

Resampling: CT is resampled to PET space using SimpleITK.
PET and TTB are used as-is (already in the same space).

Usage:
    python 1b_prepare_nnunet_dataset002.py
"""

import json
import shutil
from pathlib import Path
from typing import List, Tuple

import SimpleITK as sitk
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DEEPPSMA_ROOT = Path("/home/iml/fryderyk.koegl/data/psma_lesions")
NNUNET_RAW_ROOT = Path("/home/iml/fryderyk.koegl/data/PSMAReg_dataset_nnunet")

DATASET_ID = 2
DATASET_NAME = f"Dataset{DATASET_ID:03d}_DEEPPSMALesions"

# ---------------------------------------------------------------------------


def resample_ct_to_pet(ct_path: Path, pet_path: Path) -> sitk.Image:
    """
    Resample CT image to PET image space using linear interpolation.

    Args:
        ct_path: Path to CT NIfTI file.
        pet_path: Path to PET NIfTI file (reference space).

    Returns:
        Resampled CT as SimpleITK image.
    """
    ct = sitk.ReadImage(str(ct_path), sitk.sitkFloat32)
    pet = sitk.ReadImage(str(pet_path), sitk.sitkFloat32)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(pet)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(-1024.0)
    resampler.SetOutputPixelType(sitk.sitkFloat32)

    return resampler.Execute(ct)


def collect_cases(root: Path) -> List[Tuple[Path, Path, Path, str]]:
    """
    Collect all (ct_path, pet_path, ttb_path, case_id) tuples from DEEP-PSMA.

    Iterates over train_0001 to train_0100, collecting both PSMA and FDG
    sub-folders for each case.

    Args:
        root: Root directory containing train_XXXX folders.

    Returns:
        List of (ct_path, pet_path, ttb_path, case_id) tuples.
    """
    cases: List[Tuple[Path, Path, Path, str]] = []

    case_dirs = sorted(root.glob("train_????"))
    for case_dir in case_dirs:
        for tracer in ["PSMA", "FDG"]:
            tracer_dir = case_dir / tracer
            ct_path = tracer_dir / "CT.nii.gz"
            pet_path = tracer_dir / "PET.nii.gz"
            ttb_path = tracer_dir / "TTB.nii.gz"

            if not ct_path.exists() or not pet_path.exists() or not ttb_path.exists():
                print(f"  Skipping {case_dir.name}/{tracer} — missing files")
                continue

            case_id = f"{case_dir.name}_{tracer}"
            cases.append((ct_path, pet_path, ttb_path, case_id))

    return cases


def main() -> None:
    # Set up output directories
    dataset_dir = NNUNET_RAW_ROOT / DATASET_NAME
    images_dir = dataset_dir / "imagesTr"
    labels_dir = dataset_dir / "labelsTr"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {dataset_dir}")

    # Collect cases
    cases = collect_cases(DEEPPSMA_ROOT)
    print(f"Found {len(cases)} cases (PSMA + FDG)")

    n_ok = 0
    n_failed = 0

    for idx, (ct_path, pet_path, ttb_path, case_id) in enumerate(
        tqdm(cases, desc="Preprocessing")
    ):
        case_name = f"DEEPPSMA_{idx:04d}"

        ct_dst = images_dir / f"{case_name}_0000.nii.gz"
        pet_dst = images_dir / f"{case_name}_0001.nii.gz"
        ttb_dst = labels_dir / f"{case_name}.nii.gz"

        try:
            # Resample CT to PET space
            ct_resampled = resample_ct_to_pet(ct_path, pet_path)
            sitk.WriteImage(ct_resampled, str(ct_dst))

            # PET — copy as-is (already in correct space)
            shutil.copy2(pet_path, pet_dst)

            # TTB — copy as-is (same space as PET)
            shutil.copy2(ttb_path, ttb_dst)

            n_ok += 1

        except Exception as e:
            tqdm.write(f"  ERROR processing {case_id}: {e}")
            n_failed += 1

    print(f"\nDone: {n_ok} succeeded, {n_failed} failed")

    # Write dataset.json
    dataset_json = {
        "channel_names": {
            "0": "CT",
            "1": "PET_SUV",
        },
        "labels": {
            "background": 0,
            "lesion": 1,
        },
        "numTraining": n_ok,
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "NibabelIOWithReorient",
    }
    with open(dataset_dir / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)
    print(f"Written: {dataset_dir / 'dataset.json'}")

    # Print next steps
    print("\n" + "=" * 60)
    print("Next steps — run these in order:")
    print("=" * 60)
    print("""
1. Copy Dataset001 plans to Dataset002 so both use the same architecture:

    cp $nnUNet_preprocessed/Dataset001_PSMALesions/nnUNetPlans.json \
    $nnUNet_preprocessed/Dataset002_DEEPPSMALesions/nnUNetPlans.json
    cp $nnUNet_raw/Dataset002_DEEPPSMALesions/dataset.json \
   $nnUNet_preprocessed/Dataset002_DEEPPSMALesions/dataset.json

   (run this AFTER step 2 creates the Dataset002 preprocessed folder)

2. Extract fingerprint for Dataset002:

   nnUNetv2_extract_fingerprint -d 2

3. Preprocess Dataset002 using Dataset001's plan:

   nnUNetv2_preprocess -d 2 -plans_name nnUNetPlans

4. Pretrain on Dataset002:

   nnUNetv2_train 2 3d_fullres 0 -p nnUNetPlans --npz

5. Fine-tune on Dataset001 using Dataset002 weights:

   nnUNetv2_train 1 3d_fullres 0 -p nnUNetPlans --npz \
    --pretrained_weights \
    $nnUNet_results/Dataset002_DEEPPSMALesions/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth
""")


if __name__ == "__main__":
    main()
