"""Score displacement fields against the challenge's metrics.

    python evaluate.py --fields runs/predictions --data-dir /path/to/PSMAReg_dataset

Reads every ``disp_<case>_<fixed>_<case>_<moving>.nii.gz`` in a directory — the
naming the container writes — and reports, per pair and averaged:

  DSC, HD95        registration accuracy, on the CT organ labels
  MTV, TLG error   PET biomarker preservation, on the lesion mask
  NDV              non-diffeomorphic volume, inside the body mask

The same functions the training loop validates with, so a number here and a
number in a training log mean the same thing. Two caveats apply to both: HD95 is
a distance-transform implementation rather than the challenge's surfel-based
one, and these are *your* labels — on the official validation and test sets the
organizers score against labels that are not released, so results will differ
from the leaderboard.
"""

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional

import nibabel as nib
import numpy as np
import torch

from psmareg import losses, metrics
from psmareg.config import ModelConfig
from psmareg.data import body_mask, load_displacement, norm_pet
from psmareg.jacobian import non_diff_volume_loss
from psmareg.transforms import identity_grid_tensor, unit_flow_to_voxel, warp

# disp_0001_00_0001_01.nii.gz -> fixed 0001/00, moving 0001/01
FIELD_PATTERN = re.compile(
    r"disp_(?P<fixed_case>\w+?)_(?P<fixed_tp>\d+)_(?P<moving_case>\w+?)_(?P<moving_tp>\d+)\.nii\.gz$"
)


def load_volume(path: Path, device: torch.device) -> torch.Tensor:
    array = nib.load(str(path)).get_fdata().astype(np.float32)
    return torch.from_numpy(array)[None, None].to(device)


def score_pair(
    field_path: Path,
    data_dir: Path,
    cfg: ModelConfig,
    grid: torch.Tensor,
    device: torch.device,
    lesion_dir: Optional[Path] = None,
) -> Optional[Dict[str, float]]:
    """Every scored metric for one displacement field."""
    match = FIELD_PATTERN.search(field_path.name)
    if match is None:
        return None
    parts = match.groupdict()

    images, labels = data_dir / "imagesTr", data_dir / "labelsTr"
    if not (images / f"PSMARegPSMA_{parts['moving_case']}_0000_{parts['moving_tp']}.nii.gz").exists():
        images, labels = data_dir / "imagesTs", data_dir / "labelsTs"

    name = lambda case, channel, tp: f"PSMARegPSMA_{case}_{channel}_{tp}.nii.gz"
    moving_case, moving_tp = parts["moving_case"], parts["moving_tp"]
    fixed_case, fixed_tp = parts["fixed_case"], parts["fixed_tp"]

    flow = load_displacement(field_path, cfg.img_shape, device)

    moving_label_ct = load_volume(labels / name(moving_case, "0000", moving_tp), device)
    fixed_label_ct = load_volume(labels / name(fixed_case, "0000", fixed_tp), device)
    warped_label_ct = warp(moving_label_ct, flow, grid, nearest=True)

    # The lesion mask is warped with NEAREST and the PET image with bilinear —
    # the combination the scorer uses. A bilinear mask would sit between whole
    # voxels, which is not a volume anyone reports.
    lesion_path = (lesion_dir or labels) / name(moving_case, "0001", moving_tp)
    lesion = (load_volume(lesion_path, device) == 1).float()
    moving_pet = load_volume(images / name(moving_case, "0001", moving_tp), device)
    moving_pet = torch.from_numpy(
        norm_pet(moving_pet.cpu().numpy(), cfg.pet_suv_max)
    ).to(device)
    warped_lesion = warp(lesion, flow, grid, nearest=True)
    warped_pet = warp(moving_pet, flow, grid)

    fixed_ct = nib.load(str(images / name(fixed_case, "0000", fixed_tp))).get_fdata()
    mask = torch.from_numpy(body_mask(fixed_ct.astype(np.float32)).astype(np.float32))
    mask = mask[None, None].to(device)

    return {
        "dsc": metrics.multilabel_dice(
            warped_label_ct[0, 0].round().long(), fixed_label_ct[0, 0].round().long()
        ),
        "hd95": metrics.hd95(
            warped_label_ct, fixed_label_ct, moving_label_ct, cfg.spacing
        ),
        "mtv": losses.mtv_bias(warped_lesion, lesion).item(),
        "tlg": losses.tlg_bias(warped_pet, warped_lesion, moving_pet, lesion).item(),
        "ndv": non_diff_volume_loss(unit_flow_to_voxel(flow), mask=mask).item(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score displacement fields.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--fields", type=Path, required=True, help="directory of disp_*.nii.gz"
    )
    parser.add_argument(
        "--data-dir", type=Path, required=True, help="dataset root, for images and labels"
    )
    parser.add_argument(
        "--lesion-masks",
        type=Path,
        help="directory of PET lesion masks for the MTV and TLG terms, named like "
        "the labels. Defaults to the label directory. MTV and TLG depend strongly "
        "on which mask is used, so score against the same masks you are comparing "
        "to — the reported numbers use the annotations, not a predicted mask.",
    )
    parser.add_argument("--csv", type=Path, help="write per-pair results here")
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    cfg = ModelConfig()
    grid = identity_grid_tensor(cfg.img_shape, device)

    fields = sorted(args.fields.glob("disp_*.nii.gz"))
    if not fields:
        raise SystemExit(f"no disp_*.nii.gz in {args.fields}")

    rows: List[Dict[str, float]] = []
    for path in fields:
        scores = score_pair(
            path, args.data_dir, cfg, grid, device, args.lesion_masks
        )
        if scores is None:
            print(f"skipping {path.name}: unrecognised name")
            continue
        rows.append({"pair": path.stem.removesuffix(".nii"), **scores})
        print(
            f"{rows[-1]['pair']}  DSC {scores['dsc'] * 100:5.1f}%  "
            f"HD95 {scores['hd95']:5.2f}mm  MTV {scores['mtv'] * 100:5.2f}%  "
            f"TLG {scores['tlg'] * 100:5.2f}%  NDV {scores['ndv'] * 1e4:6.1f}ppm",
            flush=True,
        )

    if not rows:
        raise SystemExit("nothing scored")

    # nanmean: a pair whose follow-up holds no lesion has no defined MTV or TLG,
    # and should not drag the average towards zero.
    mean = {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0] if k != "pair"}
    print(
        f"\nmean over {len(rows)} pairs:  DSC {mean['dsc'] * 100:.1f}%  "
        f"HD95 {mean['hd95']:.2f}mm  MTV {mean['mtv'] * 100:.2f}%  "
        f"TLG {mean['tlg'] * 100:.2f}%  NDV {mean['ndv'] * 1e4:.1f}ppm"
    )

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            writer.writerow({"pair": "mean", **mean})
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
