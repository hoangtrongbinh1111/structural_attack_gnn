"""Build the train/test split of one CTU-13 botnet scenario.

    python preprocessing/ctu13.py --mal datasets/CTU/menti.csv --ben datasets/CTU/benign.csv \
        --out data/CTU/seed2 --seed 2

TCP flows only, 20:1 benign:malicious, 75/25 split. Nodes are IP:port. The
cleaned benign flows are cached next to --ben (cleaning ~20M rows is slow).
The test split is capped at 300 malicious flows (+20x benign) to keep the
attack sweeps tractable; the training split is never capped.
"""
import argparse
import os

import numpy as np
import pandas as pd

STATE_FLAGS = ["P", "A", "S", "C", "F", "U", "R", "E"]
LOCAL_NETWORK = "147.32."
BENIGN_RATIO = 20


def clean(flows):
    """Drop non-TCP and incomplete records, derive ratios, remove outliers."""
    flows = flows.drop(columns=["StartTime"])
    flows = flows[flows.Proto == "tcp"].copy()
    flows["Proto"] = 0
    flows = flows.replace([np.inf, -np.inf], np.nan)
    flows["sTos"] = flows["sTos"].fillna(0)
    flows["dTos"] = flows["dTos"].fillna(0)
    flows = flows.dropna(how="any", axis=0)

    flows["BytesPerPkt"] = flows["TotBytes"] / flows["TotPkts"]
    flows["PktsPerSec"] = flows["TotPkts"] / flows["Dur"]
    flows["RatioOutIn"] = (flows["TotBytes"] - flows["SrcBytes"]) / flows["SrcBytes"]
    for col in ("PktsPerSec", "RatioOutIn"):
        finite = flows.loc[flows[col] != np.inf, col]
        if len(finite):
            flows[col] = flows[col].replace(np.inf, finite.max())
    flows["DstBytes"] = flows["TotBytes"] - flows["SrcBytes"]

    return flows[(flows.Dur <= 300) & (flows.SrcBytes < 60000) & (flows.TotPkts < 100) & (flows.PktsPerSec < 10000)]


def port_class(ports, side):
    return np.where(ports <= 1023, f"{side}PortWellKnown",
                    np.where(ports <= 49151, f"{side}PortRegistered", f"{side}PortPrivate"))


def encode(flows):
    """One-hot port classes, direction and Argus TCP states; IP:port node names."""
    flows = flows.copy()
    flows["Sport"] = flows.Sport.astype(np.int64)
    flows["Dport"] = flows.Dport.astype(np.int64)
    flows["SrcPort"] = port_class(flows.Sport, "Src")
    flows["DstPort"] = port_class(flows.Dport, "Dst")
    flows = pd.get_dummies(flows, columns=["SrcPort", "DstPort"], prefix="", prefix_sep="")

    direction = pd.get_dummies(flows.Dir, prefix="Dir")
    flows[direction.columns] = direction
    flows = flows.drop(columns="Dir")

    states = flows.State.str.split("_")
    src_state, dst_state = states.str[0], states.str[1]
    for flag in STATE_FLAGS:
        flows[f"SrcState_{flag}"] = src_state.apply(lambda s: int(flag in s))
        flows[f"DstState_{flag}"] = dst_state.apply(lambda s: int(flag in s))
    flows = flows.drop(columns="State")

    flows["IPSrcType"] = flows.SrcAddr.str.startswith(LOCAL_NETWORK).astype(int)
    flows["IPDstType"] = flows.DstAddr.str.startswith(LOCAL_NETWORK).astype(int)
    flows["Src IP"] = flows.SrcAddr.astype(str) + ":" + flows.Sport.astype(str)
    flows["Dst IP"] = flows.DstAddr.astype(str) + ":" + flows.Dport.astype(str)
    return flows.drop(columns=["Sport", "Dport", "SrcAddr", "DstAddr"])


def balance(mal, ben, seed):
    if len(ben) >= BENIGN_RATIO * len(mal):
        return mal, ben.sample(BENIGN_RATIO * len(mal), random_state=seed)
    return mal.sample(len(ben) // BENIGN_RATIO, random_state=seed), ben


def cleaned_benign(path):
    cache = os.path.splitext(path)[0] + "_clean.pkl"
    if not os.path.exists(cache):
        print("cleaning benign flows (one-time)...")
        clean(pd.read_csv(path)).to_pickle(cache)
    return pd.read_pickle(cache)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mal", required=True, help="malicious flows of one botnet, e.g. datasets/CTU/menti.csv")
    ap.add_argument("--ben", required=True, help="benign flows, e.g. datasets/CTU/benign.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--max-test-mal", type=int, default=300)
    args = ap.parse_args()
    seed = args.seed

    mal, ben = balance(clean(pd.read_csv(args.mal)), cleaned_benign(args.ben), seed)
    flows = encode(pd.concat([mal, ben], ignore_index=True))
    mal, ben = balance(flows[flows.Label == 1], flows[flows.Label == 0], seed)

    train_mal = mal.sample(frac=0.75, random_state=seed)
    train_ben = ben.sample(frac=0.75, random_state=seed)
    test_mal, test_ben = mal.drop(train_mal.index), ben.drop(train_ben.index)
    if len(test_mal) > args.max_test_mal:
        test_mal = test_mal.sample(args.max_test_mal, random_state=seed)
        test_ben = test_ben.sample(min(len(test_ben), BENIGN_RATIO * len(test_mal)), random_state=seed)

    train = pd.concat([train_mal, train_ben], ignore_index=True).sample(frac=1, random_state=seed)
    test = pd.concat([test_mal, test_ben], ignore_index=True).sample(frac=1, random_state=seed)
    os.makedirs(args.out, exist_ok=True)
    scenario = os.path.splitext(os.path.basename(args.mal))[0]
    train.to_csv(os.path.join(args.out, f"{scenario}_train.csv"), index=False)
    test.to_csv(os.path.join(args.out, f"{scenario}_test.csv"), index=False)
    print(f"{scenario}: train {len(train)} ({len(train_mal)} malicious), test {len(test)} ({len(test_mal)} malicious)")


if __name__ == "__main__":
    main()
