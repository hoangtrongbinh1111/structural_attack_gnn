"""Loading preprocessed flow splits and the per-dataset conventions."""
import os

import pandas as pd

NODE_COLUMNS = ("Src IP", "Dst IP")
NON_FEATURES = NODE_COLUMNS + ("Label",)

# CTU-13 (Argus) behavior statistics. These are the only scaled columns on CTU-13;
# every other column there is a one-hot or a small categorical value.
CTU_BEHAVIOR = ["Dur", "TotPkts", "TotBytes", "SrcBytes", "BytesPerPkt",
                "PktsPerSec", "RatioOutIn", "DstBytes"]

# Large CTU-13 scenario that does not fit full-batch training on one GPU.
LARGE_SCENARIOS = {"neris": 30_000}


def dataset_of(df):
    """Identify the dataset from its column set."""
    if "src_ip_bytes" in df.columns:
        return "ToN_IoT"
    if "NUM_PKTS_UP_TO_128_BYTES" in df.columns:
        return "NF_UNSW"
    if "Flow Bytes/s" in df.columns:
        return "IDS2017"
    return "CTU"


def is_ctu(df):
    return dataset_of(df) == "CTU"


def scenario_of(path):
    """'.../menti_train.csv' -> 'menti'."""
    return os.path.basename(path).rsplit("_", 1)[0]


def feature_columns(df):
    return [c for c in df.columns if c not in NON_FEATURES]


def columns_to_scale(df):
    return list(CTU_BEHAVIOR) if is_ctu(df) else feature_columns(df)


def subsample_large_scenario(df, scenario, seed):
    """Keep `LARGE_SCENARIOS[scenario]` malicious flows and 10x as many benign ones."""
    if scenario not in LARGE_SCENARIOS:
        return df
    mal = df[df.Label == 1].sample(LARGE_SCENARIOS[scenario], random_state=seed)
    ben = df[df.Label == 0].sample(len(mal) * 10, random_state=seed)
    return pd.concat([mal, ben]).sample(frac=1, random_state=seed)
