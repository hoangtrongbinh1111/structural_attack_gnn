"""Build the train/test split of one CIC-IDS2017 attack type.

    python preprocessing/cicids2017.py --mal datasets/IDS2017/bot.csv \
        --ben datasets/IDS2017/benign.csv --out data/IDS2017

10:1 benign:malicious, 75/25 split, nodes are IP:port. Durations stay in
microseconds (CICFlowMeter's unit). For the three largest scenarios (ddos,
dos_hulk, port_scan) pass --max-train 300000 --max-test 30000 --suffix _sub.
"""
import argparse
import os

import numpy as np
import pandas as pd

from ton_iot import PRIVATE_IP, port_class, split

BENIGN_RATIO = 10
UNUSED = [
    "Flow ID", "Timestamp", "Protocol", "Source IP", "Source Port", "Destination IP", "Destination Port",
    "Fwd Packet Length Min", "Fwd Packet Length Std", "Bwd Packet Length Max", "Flow IAT Mean", "Flow IAT Std",
    "Flow IAT Max", "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Max", "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std",
    "Bwd IAT Max", "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length",
    "Bwd Header Length", "Fwd Packets/s", "Min Packet Length", "Packet Length Std", "Packet Length Variance",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
    "CWE Flag Count", "ECE Flag Count", "Down/Up Ratio", "Avg Fwd Segment Size", "Avg Bwd Segment Size",
    "Fwd Header Length.1", "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate", "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets",
    "Subflow Bwd Bytes", "Active Mean", "Active Std", "Active Max", "Active Min", "Idle Mean", "Idle Std",
    "Idle Max", "Idle Min",
]
RENAME = {"Flow Duration": "Dur", "Total Length of Fwd Packets": "SrcBytes", "Total Length of Bwd Packets": "DstBytes",
          "Total Fwd Packets": "SrcPkts", "Total Backward Packets": "DstPkts"}


def encode(flows):
    flows = flows.copy()
    flows["Src IP"] = flows["Source IP"].astype(str) + ":" + flows["Source Port"].astype(str)
    flows["Dst IP"] = flows["Destination IP"].astype(str) + ":" + flows["Destination Port"].astype(str)
    flows["IPSrcType"] = flows["Source IP"].astype(str).str.contains(PRIVATE_IP).astype(int)
    flows["IPDstType"] = flows["Destination IP"].astype(str).str.contains(PRIVATE_IP).astype(int)
    flows["SrcPort"] = port_class(flows["Source Port"], "Src")
    flows["DstPort"] = port_class(flows["Destination Port"], "Dst")
    flows = pd.get_dummies(flows, columns=["SrcPort", "DstPort"], prefix="", prefix_sep="")
    flows = flows.drop(columns=UNUSED)

    flows = flows.replace([np.inf, -np.inf], np.nan).dropna()
    flows = flows.rename(columns=RENAME)
    flows["TotBytes"] = flows.SrcBytes + flows.DstBytes
    flows["TotPkts"] = flows.SrcPkts + flows.DstPkts
    flows["Label"] = flows.Label.astype(int)
    return flows


def subsample(out_dir, scenario, suffix, sizes, seed):
    """Stratified random subsample of an existing split (benign:malicious ratio kept)."""
    for part, n in sizes.items():
        flows = pd.read_csv(os.path.join(out_dir, f"{scenario}_{part}.csv"))
        frac = min(1.0, n / len(flows))
        flows = flows.groupby("Label", group_keys=False).sample(frac=frac, random_state=seed).sample(frac=1, random_state=seed)
        flows.to_csv(os.path.join(out_dir, f"{scenario}{suffix}_{part}.csv"), index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mal", required=True)
    ap.add_argument("--ben", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-train", type=int)
    ap.add_argument("--max-test", type=int)
    ap.add_argument("--suffix", default="_sub")
    args = ap.parse_args()

    scenario = os.path.splitext(os.path.basename(args.mal))[0]
    flows = pd.concat([pd.read_csv(args.mal, on_bad_lines="skip"), pd.read_csv(args.ben, on_bad_lines="skip")],
                      ignore_index=True)
    split(encode(flows), args.out, scenario, args.seed, benign_ratio=BENIGN_RATIO)
    if args.max_train and args.max_test:
        subsample(args.out, scenario, args.suffix, {"train": args.max_train, "test": args.max_test}, seed=0)


if __name__ == "__main__":
    main()
