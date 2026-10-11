"""Build the train/test split of one ToN-IoT attack type.

    python preprocessing/ton_iot.py --mal datasets/ToN-IoT/backdoor.csv \
        --ben datasets/ToN-IoT/benign.csv --out data/ToN_IoT

TCP flows only, 20:1 benign:malicious, 75/25 split, nodes are IP:port.
"""
import argparse
import os

import numpy as np
import pandas as pd

PRIVATE_IP = r"^(?:10\.|172\.[1-3][678901]\.|192\.168\.)"
BENIGN_RATIO = 20
UNUSED = ["src_port", "dst_port", "src_ip", "dst_ip", "http_uri", "weird_name", "weird_addl", "weird_notice",
          "dns_query", "ssl_version", "ssl_cipher", "ssl_subject", "ssl_issuer", "http_user_agent", "http_method",
          "http_version", "http_request_body_len", "http_response_body_len", "http_status_code",
          "http_orig_mime_types", "http_resp_mime_types", "http_trans_depth", "type"]
BOOLEAN = ["dns_AA", "dns_RA", "dns_RD", "dns_rejected", "ssl_resumed", "ssl_established"]
RENAME = {"label": "Label", "duration": "Dur", "src_bytes": "SrcBytes", "dst_bytes": "DstBytes",
          "src_pkts": "SrcPkts", "dst_pkts": "DstPkts"}


def port_class(ports, side):
    return np.where(ports <= 1023, f"{side}PortWellKnown",
                    np.where(ports <= 49151, f"{side}PortRegistered", f"{side}PortPrivate"))


def encode(flows):
    flows = flows[flows.proto == "tcp"].copy()
    flows["proto"] = 0
    flows["Src IP"] = flows.src_ip.astype(str) + ":" + flows.src_port.astype(str)
    flows["Dst IP"] = flows.dst_ip.astype(str) + ":" + flows.dst_port.astype(str)
    flows["IPSrcType"] = flows.src_ip.astype(str).str.contains(PRIVATE_IP).astype(int)
    flows["IPDstType"] = flows.dst_ip.astype(str).str.contains(PRIVATE_IP).astype(int)
    flows["SrcPort"] = port_class(flows.src_port, "Src")
    flows["DstPort"] = port_class(flows.dst_port, "Dst")
    flows = pd.get_dummies(flows, columns=["SrcPort", "DstPort"], prefix="", prefix_sep="")
    flows = flows.drop(columns=UNUSED)

    for col in BOOLEAN:
        flows[col] = (flows[col] == "T").astype(int)
    flows = pd.get_dummies(flows, columns=["conn_state", "service"])
    flows = flows.rename(columns=RENAME)
    flows["TotBytes"] = flows.SrcBytes + flows.DstBytes
    flows["TotPkts"] = flows.SrcPkts + flows.DstPkts
    flows["Label"] = flows.Label.astype(int)
    return flows


def split(flows, out_dir, scenario, seed, benign_ratio=BENIGN_RATIO):
    """Balance to benign_ratio:1 and write a 75/25 train/test split."""
    mal, ben = flows[flows.Label == 1], flows[flows.Label == 0]
    if len(ben) >= benign_ratio * len(mal):
        ben = ben.sample(benign_ratio * len(mal), random_state=seed)
    else:
        mal = mal.sample(len(ben) // benign_ratio, random_state=seed)
    train_mal = mal.sample(frac=0.75, random_state=seed)
    train_ben = ben.sample(frac=0.75, random_state=seed)
    train = pd.concat([train_mal, train_ben], ignore_index=True).sample(frac=1, random_state=seed)
    test = pd.concat([mal.drop(train_mal.index), ben.drop(train_ben.index)], ignore_index=True)
    test = test.sample(frac=1, random_state=seed)
    os.makedirs(out_dir, exist_ok=True)
    train.to_csv(os.path.join(out_dir, f"{scenario}_train.csv"), index=False)
    test.to_csv(os.path.join(out_dir, f"{scenario}_test.csv"), index=False)
    print(f"{scenario}: train {len(train)}, test {len(test)}, {int(train.Label.sum())} malicious in train")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mal", required=True)
    ap.add_argument("--ben", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    flows = pd.concat([pd.read_csv(args.mal, on_bad_lines="skip"), pd.read_csv(args.ben, on_bad_lines="skip")],
                      ignore_index=True)
    split(encode(flows), args.out, os.path.splitext(os.path.basename(args.mal))[0], args.seed)


if __name__ == "__main__":
    main()
