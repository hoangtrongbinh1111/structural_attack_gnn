"""Build one train/test split per attack class of NF-UNSW-NB15-v2.

    python preprocessing/nf_unsw.py --csv datasets/NF-UNSW-NB15-v2/NF-UNSW-NB15-v2.csv --out data/NF_UNSW

NetFlow fields are renamed to the shared names the attacks edit (Dur in
seconds, SrcBytes/DstBytes, SrcPkts/DstPkts, TotBytes, TotPkts); discrete
fields become 0/1 columns (protocol, top L7 protocols, every TCP flag bit,
port class, private/public IP). 20:1 benign:malicious, 75/25 split, at most
--max-mal malicious flows per class.
"""
import argparse
import ipaddress
import os

import numpy as np
import pandas as pd

TCP_FLAG_BITS = ["FIN", "SYN", "RST", "PSH", "ACK", "URG", "ECE", "CWR"]
TOP_L7 = 10
DROP = ["DNS_QUERY_ID", "ICMP_TYPE", "ICMP_IPV4_TYPE", "FTP_COMMAND_RET_CODE", "L4_SRC_PORT", "L4_DST_PORT",
        "IPV4_SRC_ADDR", "IPV4_DST_ADDR", "PROTOCOL", "L7_PROTO", "TCP_FLAGS", "CLIENT_TCP_FLAGS",
        "SERVER_TCP_FLAGS", "Attack", "IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS",
        "FLOW_DURATION_MILLISECONDS", "Label"]


def is_private(ip):
    try:
        return int(ipaddress.ip_address(ip).is_private)
    except ValueError:
        return 0


def encode(raw, l7_top):
    out = pd.DataFrame(index=raw.index)
    out["Src IP"] = raw.IPV4_SRC_ADDR.astype(str) + ":" + raw.L4_SRC_PORT.astype(str)
    out["Dst IP"] = raw.IPV4_DST_ADDR.astype(str) + ":" + raw.L4_DST_PORT.astype(str)
    out["Dur"] = raw.FLOW_DURATION_MILLISECONDS / 1000.0
    out["SrcBytes"] = raw.IN_BYTES
    out["DstBytes"] = raw.OUT_BYTES
    out["SrcPkts"] = raw.IN_PKTS
    out["DstPkts"] = raw.OUT_PKTS
    continuous = raw.drop(columns=DROP).replace([np.inf, -np.inf], np.nan).fillna(0)
    out = pd.concat([out, continuous], axis=1)

    private = {ip: is_private(ip) for ip in pd.unique(pd.concat([raw.IPV4_SRC_ADDR, raw.IPV4_DST_ADDR]))}
    out["IPSrcType"] = raw.IPV4_SRC_ADDR.map(private)
    out["IPDstType"] = raw.IPV4_DST_ADDR.map(private)
    for side, col in (("Src", "L4_SRC_PORT"), ("Dst", "L4_DST_PORT")):
        ports = raw[col].to_numpy()
        out[f"{side}PortWellKnown"] = (ports <= 1023).astype(int)
        out[f"{side}PortRegistered"] = ((ports > 1023) & (ports <= 49151)).astype(int)
        out[f"{side}PortPrivate"] = (ports > 49151).astype(int)
    out["proto_tcp"] = (raw.PROTOCOL == 6).astype(int)
    out["proto_udp"] = (raw.PROTOCOL == 17).astype(int)
    out["proto_other"] = (~raw.PROTOCOL.isin([6, 17])).astype(int)
    l7 = raw.L7_PROTO.round().astype(int)
    for v in l7_top:
        out[f"l7_{v}"] = (l7 == v).astype(int)
    out["l7_other"] = (~l7.isin(l7_top)).astype(int)
    for col, prefix in (("TCP_FLAGS", "flag"), ("CLIENT_TCP_FLAGS", "cflag"), ("SERVER_TCP_FLAGS", "sflag")):
        for bit, name in enumerate(TCP_FLAG_BITS):
            out[f"{prefix}_{name}"] = ((raw[col].to_numpy() >> bit) & 1).astype(int)

    out["TotBytes"] = out.SrcBytes + out.DstBytes
    out["TotPkts"] = out.SrcPkts + out.DstPkts
    out["Label"] = raw.Label.astype(int)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-mal", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    raw = pd.read_csv(args.csv)
    benign = raw[raw.Label == 0]
    # one L7 vocabulary for the whole file, so every scenario has the same columns
    l7_top = raw.L7_PROTO.round().astype(int).value_counts().index[:TOP_L7].tolist()

    for attack, mal in raw[raw.Label == 1].groupby("Attack"):
        name = attack.lower().replace(" ", "_")
        if len(mal) > args.max_mal:
            mal = mal.sample(args.max_mal, random_state=rng)
        ben = benign.sample(min(len(benign), 20 * len(mal)), random_state=rng)
        flows = encode(pd.concat([mal, ben]), l7_top)
        mal_f, ben_f = flows[flows.Label == 1], flows[flows.Label == 0]
        train_mal = mal_f.sample(frac=0.75, random_state=rng)
        train_ben = ben_f.sample(frac=0.75, random_state=rng)
        train = pd.concat([train_mal, train_ben]).sample(frac=1, random_state=rng)
        test = pd.concat([mal_f.drop(train_mal.index), ben_f.drop(train_ben.index)]).sample(frac=1, random_state=rng)
        train.to_csv(os.path.join(args.out, f"{name}_train.csv"), index=False)
        test.to_csv(os.path.join(args.out, f"{name}_test.csv"), index=False)
        print(f"{name}: train {len(train)}, test {len(test)}")


if __name__ == "__main__":
    main()
