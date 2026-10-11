"""Evaluate a trained run on the clean test split and under attack.

    python -m guardnids.evaluate --run runs/CTU/menti/seed2/guardnids_rr \
        --test data/CTU/seed2/menti_test.csv --attacks clean c2xb padding

Writes <run>/scores/<attack>.csv. With --retrieval-only (GuardNIDS-RR runs) the
graph-free control is scored instead, into the sibling run 'retrieval_only'.
"""
import argparse
import os
import random
import time

import numpy as np
import pandas as pd

from guardnids.attacks import ATTACKS
from guardnids.detectors import load_detector


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run directory written by guardnids.train")
    ap.add_argument("--test", required=True, help="<scenario>_test.csv")
    ap.add_argument("--attacks", nargs="+", default=["all"], help=f"'all' or any of: {' '.join(ATTACKS)}")
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--seed", type=int, default=0, help="seed of the attack generators")
    ap.add_argument("--force", action="store_true", help="recompute attacks that already have scores")
    args = ap.parse_args()

    attacks = list(ATTACKS) if args.attacks == ["all"] else args.attacks
    unknown = set(attacks) - set(ATTACKS)
    if unknown:
        ap.error(f"unknown attacks: {sorted(unknown)}")

    out_run = os.path.join(os.path.dirname(args.run.rstrip("/")), "retrieval_only") if args.retrieval_only else args.run
    score_dir = os.path.join(out_run, "scores")
    os.makedirs(score_dir, exist_ok=True)

    detector = load_detector(args.run, retrieval_only=args.retrieval_only)
    flows = pd.read_csv(args.test)
    for name in attacks:
        path = os.path.join(score_dir, f"{name}.csv")
        if os.path.exists(path) and not args.force:
            continue
        random.seed(args.seed)
        np.random.seed(args.seed)
        start = time.time()
        rows = pd.DataFrame(ATTACKS[name](detector, flows))
        rows.to_csv(path, index=False)
        if rows.empty:
            print(f"{name:24s} not applicable to this dataset")
            continue
        print(f"{name:24s} F1 {rows.f1.mean():.3f}  recall {rows.recall.mean():.3f}  ({time.time() - start:.0f}s)")


if __name__ == "__main__":
    main()
