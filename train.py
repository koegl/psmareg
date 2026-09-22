"""Train one level of the pyramid.

    python train.py --data-dir /path/to/PSMAReg_dataset --out-dir runs/psmareg --level 1
    python train.py ... --level 2 --init runs/psmareg/level1_best.pth
    python train.py ... --level 3 --init runs/psmareg/level2_best.pth

Levels are trained in order and each finer one starts from the checkpoint of the
level below. Only the full-resolution level carries the complete objective; the
two coarser ones use similarity, labels and regularization alone, because a
lesion or a rib spans too few voxels there for the volume and rigidity terms to
measure anatomy rather than interpolation.

Checkpoints are selected on the challenge's composite score rather than on
alignment: registration accuracy keeps improving after the MTV and TLG errors
have reached their minimum, so the best-aligned checkpoint is not the one to
submit.
"""

import argparse
from pathlib import Path

import torch

from psmareg.config import TrainConfig
from psmareg.trainer import train_level


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one LapIRN pyramid level.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", type=Path, required=True, help="dataset root (imagesTr, labelsTr)"
    )
    parser.add_argument(
        "--out-dir", type=Path, required=True, help="checkpoints, split and affine cache"
    )
    parser.add_argument("--level", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument(
        "--init", type=Path, help="checkpoint of the previous level (required for 2 and 3)"
    )
    parser.add_argument("--steps", type=int, help="override the schedule for this level")
    parser.add_argument("--lr", type=float, help="override the learning rate")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="affine pre-registrations; defaults to <out-dir>/affine_cache. "
        "Share one across levels — the affine does not depend on the level.",
    )
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--num-workers", type=int, default=TrainConfig.num_workers)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.level > 1 and args.init is None:
        raise SystemExit(f"--init is required for level {args.level}")

    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    cfg = TrainConfig()
    cfg.augment = not args.no_augment
    cfg.num_workers = args.num_workers
    if args.lr is not None:
        cfg.lr[args.level] = args.lr

    best = train_level(
        level=args.level,
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        cfg=cfg,
        device=torch.device(args.device),
        init=args.init,
        steps=args.steps,
        cache_dir=args.cache_dir,
    )
    print(f"best checkpoint: {best}")


if __name__ == "__main__":
    main()
