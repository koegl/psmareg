"""Register one PSMA PET/CT pair and write the displacement field.

    python inference.py --fixed-ct ... --fixed-pet ... \
                        --moving-ct ... --moving-pet ... \
                        --weights model.pth --out disp.nii.gz

Fixed is the baseline scan, moving the follow-up. The stages are:

  1. ANTs affine pre-registration, from the CT channels alone
  2. the LapIRN pyramid, on the affinely aligned pair
  3. composition of the two into one transform
  4. optionally, instance optimization of that transform for this pair

Instance optimization needs a PET lesion mask and CT organ labels. Pass them
with ``--moving-lesion`` / ``--moving-labels`` / ``--fixed-labels``, or let them
be predicted: CT organs come from TotalSegmentator, and PET lesions from an
nnU-Net model directory given with ``--lesion-model``. Whatever is missing
simply switches off the terms that read it.

The output is a dense ``(3, H, W, D)`` field of voxel displacements mapping the
follow-up onto the baseline grid.
"""

import argparse
import tempfile
import time
from pathlib import Path

import torch

from psmareg.affine import affine_flow
from psmareg.config import ModelConfig
from psmareg.data import load_pair, save_displacement
from psmareg.instance_opt import IOConfig, IOInputs, run_io
from psmareg.model import build_model
from psmareg.transforms import compose, identity_grid_tensor, warp


def register(
    moving: torch.Tensor,
    fixed: torch.Tensor,
    fixed_ct: Path,
    moving_ct: Path,
    weights: Path,
    grid: torch.Tensor,
    cfg: ModelConfig,
    device: torch.device,
) -> torch.Tensor:
    """Affine, network, and the composition of the two.

    Returns the total unit flow ``(1, 3, H, W, D)`` — the [-1, 1] convention of
    ``grid_sample``, which :func:`psmareg.data.save_displacement` converts to
    the voxel displacements the challenge scores.
    """
    flow_affine = affine_flow(fixed_ct, moving_ct, cfg, device)
    moving_affine = warp(moving, flow_affine, grid)

    model = build_model(cfg, device, weights).eval()
    with torch.no_grad():
        prediction = model(moving_affine, fixed)

    # The affine goes outside the network field: the network saw an already
    # aligned image, so its prediction is defined in that frame. Composing this
    # way gives one transform from the ORIGINAL moving image to the fixed grid,
    # which is what the scorer expects and what makes the Jacobian reflect the
    # total volume change.
    return compose(flow_affine, prediction.flow.permute(0, 2, 3, 4, 1), grid)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Register one PSMA PET/CT pair.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--fixed-ct", type=Path, required=True, help="baseline CT")
    parser.add_argument("--fixed-pet", type=Path, required=True, help="baseline PET")
    parser.add_argument("--moving-ct", type=Path, required=True, help="follow-up CT")
    parser.add_argument("--moving-pet", type=Path, required=True, help="follow-up PET")
    parser.add_argument("--weights", type=Path, required=True, help="model.pth")
    parser.add_argument("--out", type=Path, required=True, help="output field")
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )

    io_group = parser.add_argument_group("instance optimization")
    io_group.add_argument(
        "--io", action="store_true", help="refine the field for this pair"
    )
    io_group.add_argument("--io-steps", type=int, default=IOConfig.steps)
    io_group.add_argument("--io-lr", type=float, default=IOConfig.lr)
    io_group.add_argument(
        "--moving-lesion", type=Path, help="PET lesion mask of the follow-up scan"
    )
    io_group.add_argument("--moving-labels", type=Path, help="CT labels, follow-up")
    io_group.add_argument("--fixed-labels", type=Path, help="CT labels, baseline")
    io_group.add_argument(
        "--lesion-model",
        type=Path,
        help="nnU-Net model directory, used when --moving-lesion is not given",
    )
    io_group.add_argument(
        "--lesion-folds",
        type=int,
        nargs="+",
        default=[0],
        help="folds of the lesion model to ensemble",
    )
    io_group.add_argument(
        "--no-segment-ct",
        action="store_true",
        help="skip TotalSegmentator; the Dice and rigidity terms are then off",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # The affine stage has its own unseeded RNG inside ITK, but everything on
    # this side is reproducible.
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device(args.device)
    cfg = ModelConfig()
    grid = identity_grid_tensor(cfg.img_shape, device)

    start = time.time()
    moving, fixed = load_pair(
        args.fixed_ct, args.fixed_pet, args.moving_ct, args.moving_pet, cfg
    )
    moving, fixed = moving.to(device), fixed.to(device)

    flow = register(
        moving, fixed, args.fixed_ct, args.moving_ct, args.weights, grid, cfg, device
    )
    print(f"registration done in {time.time() - start:.1f}s", flush=True)

    if args.io:
        from psmareg.segmentation import prepare_io_labels

        with tempfile.TemporaryDirectory() as work_dir:
            lesion, moving_labels, fixed_labels = prepare_io_labels(
                moving_ct=args.moving_ct,
                moving_pet=args.moving_pet,
                fixed_ct=args.fixed_ct,
                device=device,
                work_dir=Path(work_dir),
                lesion_path=args.moving_lesion,
                moving_labels_path=args.moving_labels,
                fixed_labels_path=args.fixed_labels,
                lesion_model=args.lesion_model,
                lesion_folds=tuple(args.lesion_folds),
                segment_ct=not args.no_segment_ct,
            )

        inputs = IOInputs(lesion, moving_labels, fixed_labels)
        io_start = time.time()
        flow = run_io(
            flow,
            moving,
            fixed,
            inputs,
            grid,
            IOConfig(steps=args.io_steps, lr=args.io_lr),
            device,
        )
        print(f"instance optimization done in {time.time() - io_start:.1f}s", flush=True)

    save_displacement(flow, args.out)
    print(f"wrote {args.out} in {time.time() - start:.1f}s total", flush=True)


if __name__ == "__main__":
    main()
