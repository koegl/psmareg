"""
Phase 0: Create internal train/test split for the PSMAReg Learn2Reg dataset.

Operates entirely on the dataset JSON — no files are copied or moved.
Outputs a split JSON with the following structure:

{
    "train_paired":   [ { <subject entry> }, ... ],  # paired subjects for training
    "test_paired":    [ { <subject entry> }, ... ],  # paired subjects held out for evaluation
    "train_unpaired": [ { <subject entry> }, ... ],  # unpaired subjects (train only)
    "val_paired":     [ { <subject entry> }, ... ],  # original challenge val set (unchanged)
    "split_stats": { ... }                           # summary statistics
}

Split strategy:
- Patient-level split (never split a patient across train/test)
- ~10% of paired subjects held out as internal test (~15 of 146)
- Stratified by number of timepoints per subject as a proxy for disease complexity
- Unpaired subjects always go to train (no paired follow-up to evaluate registration)
- Challenge validation set is kept separate and untouched
- Random seed is fixed for reproducibility
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

DATASET_JSON_PATH = Path(
    "/home/iml/fryderyk.koegl/data/PSMAReg_dataset/PSMAReg_dataset.json"
)
OUTPUT_SPLIT_PATH = Path(
    "/home/iml/fryderyk.koegl/data/PSMAReg_dataset/psmareg_split.json"
)
TEST_FRACTION = 0.10
RANDOM_SEED = 42


def count_timepoints(subject: Dict) -> int:
    """Count total number of timepoints (baseline + follow-ups) for a subject."""
    n = 1  # baseline always present
    fu_idx = 1
    while f"Follow-up {fu_idx:02d} CT" in subject:
        n += 1
        fu_idx += 1
    return n


def stratified_split(
    subjects: List[Dict],
    test_fraction: float,
    seed: int,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split subjects into train/test stratified by number of timepoints.

    Subjects are grouped by timepoint count, then test cases are drawn
    proportionally from each group to preserve the timepoint distribution.

    Args:
        subjects: List of paired subject entries from the dataset JSON.
        test_fraction: Fraction of subjects to hold out for testing.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_subjects, test_subjects).
    """
    rng = random.Random(seed)

    # Group subjects by timepoint count
    groups: Dict[int, List[Dict]] = {}
    for subject in subjects:
        n_tp = count_timepoints(subject)
        groups.setdefault(n_tp, [])
        groups[n_tp].append(subject)

    train_subjects: List[Dict] = []
    test_subjects: List[Dict] = []

    for n_tp, group in sorted(groups.items()):
        rng.shuffle(group)
        n_test = max(1, round(len(group) * test_fraction))
        # Ensure we never take more than half a group as test
        n_test = min(n_test, len(group) // 2)
        test_subjects.extend(group[:n_test])
        train_subjects.extend(group[n_test:])

    return train_subjects, test_subjects


def compute_stats(
    train_paired: List[Dict],
    test_paired: List[Dict],
    train_unpaired: List[Dict],
    val_paired: List[Dict],
) -> Dict:
    """Compute summary statistics for the split."""

    def timepoint_distribution(subjects: List[Dict]) -> Dict[int, int]:
        dist: Dict[int, int] = {}
        for s in subjects:
            n = count_timepoints(s)
            dist[n] = dist.get(n, 0) + 1
        return dict(sorted(dist.items()))

    def count_pairs(subjects: List[Dict]) -> int:
        """Count total registration pairs (all pairwise combinations per subject)."""
        total = 0
        for s in subjects:
            n = count_timepoints(s)
            # Pairs: (baseline→fu1), (baseline→fu2), (fu1→fu2), etc.
            # = n*(n-1)/2 but only forward in time, so n-1 + n-2 + ... = n*(n-1)/2
            total += n * (n - 1) // 2
        return total

    return {
        "train_paired_subjects": len(train_paired),
        "test_paired_subjects": len(test_paired),
        "train_unpaired_subjects": len(train_unpaired),
        "val_paired_subjects": len(val_paired),
        "train_paired_pairs": count_pairs(train_paired),
        "test_paired_pairs": count_pairs(test_paired),
        "train_timepoint_distribution": timepoint_distribution(train_paired),
        "test_timepoint_distribution": timepoint_distribution(test_paired),
        "test_fraction_actual": (
            len(test_paired) / (len(train_paired) + len(test_paired))
        ),
        "random_seed": RANDOM_SEED,
    }


def main() -> None:
    # Load dataset JSON
    with open(DATASET_JSON_PATH, "r") as f:
        dataset = json.load(f)

    paired_subjects: List[Dict] = dataset["training_paired"]
    unpaired_subjects: List[Dict] = dataset.get("training_unpaired", [])
    val_subjects: List[Dict] = dataset.get("validation_paired", [])

    print(f"Loaded {len(paired_subjects)} paired training subjects")
    print(f"Loaded {len(unpaired_subjects)} unpaired training subjects")
    print(f"Loaded {len(val_subjects)} validation subjects (kept separate)")

    # Timepoint distribution before split
    tp_dist: Dict[int, int] = {}
    for s in paired_subjects:
        n = count_timepoints(s)
        tp_dist[n] = tp_dist.get(n, 0) + 1
    print("\nTimepoint distribution in paired training set:")
    for n_tp, count in sorted(tp_dist.items()):
        print(f"  {n_tp} timepoints: {count} subjects")

    # Stratified split
    train_paired, test_paired = stratified_split(
        paired_subjects, TEST_FRACTION, RANDOM_SEED
    )

    # Compute and display stats
    stats = compute_stats(train_paired, test_paired, unpaired_subjects, val_subjects)

    print("\nSplit result:")
    print(f"  Train paired subjects : {stats['train_paired_subjects']}")
    print(f"  Test  paired subjects : {stats['test_paired_subjects']}")
    print(f"  Train unpaired        : {stats['train_unpaired_subjects']}")
    print(
        f"  Val   paired          : {stats['val_paired_subjects']} (challenge, untouched)"
    )
    print(f"  Train registration pairs : {stats['train_paired_pairs']}")
    print(f"  Test  registration pairs : {stats['test_paired_pairs']}")
    print(f"  Actual test fraction  : {stats['test_fraction_actual']:.1%}")
    print(f"\nTrain timepoint distribution: {stats['train_timepoint_distribution']}")
    print(f"Test  timepoint distribution: {stats['test_timepoint_distribution']}")

    # Write split JSON
    split = {
        "train_paired": train_paired,
        "test_paired": test_paired,
        "train_unpaired": unpaired_subjects,
        "val_paired": val_subjects,
        "split_stats": stats,
    }

    with open(OUTPUT_SPLIT_PATH, "w") as f:
        json.dump(split, f, indent=2)

    print(f"\nSplit saved to: {OUTPUT_SPLIT_PATH}")
    print("No files were copied or moved — split is reference-only.")


if __name__ == "__main__":
    main()
