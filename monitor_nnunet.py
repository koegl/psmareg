"""
Monitor nnU-Net training progress by parsing the log file and saving a PNG plot.
Run in a tmux session — updates every 60 seconds.

Usage:
    python monitor_training.py
"""

import re
import time
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt

LOG_PATH = Path(
    "/home/iml/fryderyk.koegl/data/nnUNet_results/Dataset001_PSMALesions/nnUNetTrainer__nnUNetResEncUNetXLPlans__3d_fullres/fold_0/training_log_2026_6_12_13_58_48.txt"
)
OUTPUT_PNG = Path("/home/iml/fryderyk.koegl/code/psmareg/training_progress.png")
UPDATE_INTERVAL_S = 60


def parse_log(
    log_path: Path,
) -> Tuple[List[int], List[float], List[float], List[float]]:
    """
    Parse nnU-Net training log file.

    Args:
        log_path: Path to the training log text file.

    Returns:
        Tuple of (epochs, train_losses, val_losses, pseudo_dices).
    """
    epochs: List[int] = []
    train_losses: List[float] = []
    val_losses: List[float] = []
    pseudo_dices: List[float] = []

    current_epoch = None
    current_train_loss = None
    current_val_loss = None

    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()

            epoch_match = re.search(r":\s+Epoch (\d+)\s*$", line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                current_train_loss = None
                current_val_loss = None
                continue

            train_match = re.search(r":\s+train_loss\s+([-\d.]+)", line)
            if train_match:
                current_train_loss = float(train_match.group(1))
                continue

            val_match = re.search(r":\s+val_loss\s+([-\d.]+)", line)
            if val_match:
                current_val_loss = float(val_match.group(1))
                continue

            dice_match = re.search(r":\s+Pseudo dice \[.*?([\d.]+)\s*\]", line)
            if dice_match and current_epoch is not None:
                pseudo_dices.append(float(dice_match.group(1)))
                epochs.append(current_epoch)
                train_losses.append(
                    current_train_loss
                    if current_train_loss is not None
                    else float("nan")
                )
                val_losses.append(
                    current_val_loss if current_val_loss is not None else float("nan")
                )

    return epochs, train_losses, val_losses, pseudo_dices


def plot_and_save(
    epochs: List[int],
    train_losses: List[float],
    val_losses: List[float],
    pseudo_dices: List[float],
    output_path: Path,
) -> None:
    """
    Plot training metrics and save as PNG.

    Args:
        epochs: List of epoch numbers.
        train_losses: List of training losses.
        val_losses: List of validation losses.
        pseudo_dices: List of pseudo dice scores.
        output_path: Path to save the PNG.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(
        f"nnU-Net Training Progress — Epoch {epochs[-1]}/1000"
        if epochs
        else "nnU-Net Training Progress",
        fontsize=13,
    )

    # Losses
    ax1.plot(epochs, train_losses, label="Train loss", color="steelblue", linewidth=1.2)
    ax1.plot(epochs, val_losses, label="Val loss", color="coral", linewidth=1.2)
    ax1.set_ylabel("Loss")
    ax1.legend(framealpha=0.3)
    ax1.grid(True, alpha=0.3)

    # Pseudo dice
    ax2.plot(
        epochs, pseudo_dices, label="Pseudo dice", color="mediumseagreen", linewidth=1.2
    )
    if len(pseudo_dices) >= 10:
        # Simple moving average
        window = 20
        smoothed = [
            sum(pseudo_dices[max(0, i - window) : i + 1])
            / len(pseudo_dices[max(0, i - window) : i + 1])
            for i in range(len(pseudo_dices))
        ]
        ax2.plot(
            epochs,
            smoothed,
            label=f"Smoothed (w={window})",
            color="darkgreen",
            linewidth=2,
            linestyle="--",
        )
    ax2.set_ylabel("Pseudo Dice")
    ax2.set_xlabel("Epoch")
    ax2.set_ylim(0, 1)
    ax2.legend(framealpha=0.3)
    ax2.grid(True, alpha=0.3)

    if pseudo_dices:
        ax2.annotate(
            f"Latest: {pseudo_dices[-1]:.4f}",
            xy=(epochs[-1], pseudo_dices[-1]),
            xytext=(-60, 15),
            textcoords="offset points",
            fontsize=9,
            arrowprops={"arrowstyle": "->", "color": "gray"},
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()


def main() -> None:
    print(f"Monitoring: {LOG_PATH}")
    print(f"Output PNG: {OUTPUT_PNG}")
    print(f"Updating every {UPDATE_INTERVAL_S}s — Ctrl+C to stop\n")

    while True:
        if not LOG_PATH.exists():
            # Try globbing for the log file since the suffix may vary
            candidates = list(LOG_PATH.parent.glob("training_log*.txt"))
            if not candidates:
                print("Log file not found yet, waiting...")
                time.sleep(UPDATE_INTERVAL_S)
                continue
            actual_log = candidates[0]
        else:
            actual_log = LOG_PATH

        epochs, train_losses, val_losses, pseudo_dices = parse_log(actual_log)

        if not epochs:
            print("No epochs logged yet, waiting...")
            time.sleep(UPDATE_INTERVAL_S)
            continue

        plot_and_save(epochs, train_losses, val_losses, pseudo_dices, OUTPUT_PNG)
        print(
            f"Epoch {epochs[-1]:4d}/1000 | "
            f"train_loss: {train_losses[-1]:6.4f} | "
            f"val_loss: {val_losses[-1]:6.4f} | "
            f"pseudo_dice: {pseudo_dices[-1]:.4f} | "
            f"PNG saved"
        )

        time.sleep(UPDATE_INTERVAL_S)


if __name__ == "__main__":
    main()
