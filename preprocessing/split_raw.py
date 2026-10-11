"""Split the raw dataset files into one CSV per attack plus one benign CSV.

    python preprocessing/split_raw.py ctu13      --raw datasets/CTU_raw      --out datasets/CTU
    python preprocessing/split_raw.py ton_iot    --raw datasets/ToN-IoT_raw  --out datasets/ToN-IoT
    python preprocessing/split_raw.py cicids2017 --raw datasets/IDS2017_raw  --out datasets/IDS2017

CTU-13: every `<botnet>.binetflow` file under --raw (rename the capture files
after their botnet, e.g. 1/neris.binetflow, 10/rbot.binetflow).
ToN-IoT: train_test_network.csv. CIC-IDS2017: the MachineLearningCVE CSV files.
"""
import argparse
import glob
import os

import pandas as pd

TON_IOT_ATTACKS = ["backdoor", "ddos", "dos", "injection", "mitm", "password", "ransomware", "scanning", "xss"]
CIC_LABELS = {
    "bot": "bot", "ddos": "ddos", "dos goldeneye": "dos_goldeneye", "dos hulk": "dos_hulk",
    "dos slowhttptest": "dos_slowhttptest", "dos slowloris": "dos_slowloris", "ftp-patator": "ftp_patator",
    "heartbleed": "heartbleed", "infiltration": "infiltration", "portscan": "port_scan", "ssh-patator": "ssh_patator",
}


def append_csv(df, path):
    if len(df):
        df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def split_ctu13(raw_dir, out_dir):
    for path in sorted(glob.glob(os.path.join(raw_dir, "**", "*.binetflow"), recursive=True)):
        print(path)
        botnet = os.path.basename(path).replace(".binetflow", "")
        flows = pd.read_csv(path, on_bad_lines="skip")
        flows["Label"] = flows["Label"].astype(str).str.contains("Botnet").astype(int)
        append_csv(flows[flows.Label == 1], os.path.join(out_dir, f"{botnet}.csv"))
        append_csv(flows[flows.Label == 0], os.path.join(out_dir, "benign.csv"))


def split_ton_iot(raw_dir, out_dir):
    flows = pd.read_csv(os.path.join(raw_dir, "train_test_network.csv"), on_bad_lines="skip")
    for attack in TON_IOT_ATTACKS:
        flows[flows["type"] == attack].to_csv(os.path.join(out_dir, f"{attack}.csv"), index=False)
    flows[flows["type"] == "normal"].to_csv(os.path.join(out_dir, "benign.csv"), index=False)


def cic_scenario(label):
    label = str(label).strip().lower()
    if label == "benign":
        return "benign"
    if label.startswith("web attack"):
        for key, name in (("brute force", "web_attack_bruteforce"), ("xss", "web_attack_xss"),
                          ("sql injection", "web_attack_sql_injection")):
            if key in label:
                return name
    return CIC_LABELS.get(label)


def split_cicids2017(raw_dir, out_dir):
    for path in sorted(glob.glob(os.path.join(raw_dir, "*.csv"))):
        print(path)
        flows = pd.read_csv(path, on_bad_lines="skip", encoding="latin-1")
        flows.columns = flows.columns.str.strip()
        scenario = flows["Label"].map(cic_scenario)
        for name, group in flows.groupby(scenario):
            group = group.assign(Label=int(name != "benign"))
            append_csv(group, os.path.join(out_dir, f"{name}.csv"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=["ctu13", "ton_iot", "cicids2017"])
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if glob.glob(os.path.join(args.out, "*.csv")):
        raise SystemExit(f"{args.out} already has CSV files; files are appended to, so start from an empty folder")
    {"ctu13": split_ctu13, "ton_iot": split_ton_iot, "cicids2017": split_cicids2017}[args.dataset](args.raw, args.out)


if __name__ == "__main__":
    main()
