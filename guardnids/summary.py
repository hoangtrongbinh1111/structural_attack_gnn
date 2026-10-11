"""Summarize all scored runs into one table per metric.

    python -m guardnids.summary --runs runs --metric f1

Each attack is reduced to one number per run:
  * header_edits, port_class_padding, port_choice: the attacker keeps its most
    damaging variant (lowest recall; for header edits, the lowest-F1 edit type);
  * every other attack: mean over its levels.
Runs are averaged over seeds per scenario, then over scenarios per dataset.
"""
import argparse
import glob
import os

import pandas as pd

METRICS = ["recall", "precision", "f1", "fpr"]
BEST_VARIANT = {"port_class_padding", "port_choice"}


def reduce_attack(attack, rows):
    if attack == "header_edits":
        rows = rows.assign(group=rows.variant.str.split(":").str[0])
        worst_per_type = rows.loc[rows.groupby("group").recall.idxmin()]
        return worst_per_type.loc[worst_per_type.f1.idxmin(), METRICS]
    if attack in BEST_VARIANT:
        return rows.loc[rows.recall.idxmin(), METRICS]
    return rows[METRICS].mean()


def load_scores(runs):
    records = []
    for path in glob.glob(os.path.join(runs, "*", "*", "seed*", "*", "scores", "*.csv")):
        parts = path.split(os.sep)
        ds, scenario, seed, model, attack = parts[-6], parts[-5], parts[-4], parts[-3], parts[-1][:-4]
        rows = pd.read_csv(path)
        if rows.empty:
            continue
        values = reduce_attack(attack, rows)
        records.append({"dataset": ds, "scenario": scenario, "seed": int(seed[4:]), "model": model,
                        "attack": attack, **values.to_dict()})
    return pd.DataFrame(records)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--metric", default="f1", choices=METRICS)
    ap.add_argument("--seeds", type=int, nargs="+", help="only these seeds (default: all)")
    ap.add_argument("--out", help="also write the table to this CSV")
    args = ap.parse_args()

    scores = load_scores(args.runs)
    if scores.empty:
        raise SystemExit(f"no scores under {args.runs}/")
    if args.seeds:
        scores = scores[scores.seed.isin(args.seeds)]

    per_scenario = scores.groupby(["dataset", "attack", "model", "scenario"])[args.metric].mean()
    table = per_scenario.groupby(["dataset", "attack", "model"]).mean().unstack("model")
    pd.set_option("display.width", 200)
    print(table.round(3).to_string())
    if args.out:
        table.to_csv(args.out)


if __name__ == "__main__":
    main()
