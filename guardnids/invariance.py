"""Which flow fields a client-side attacker controls.

GuardNIDS-RR keys its retrieval only on fields the attacker cannot set. The
rule removes:
  * client-controllable discrete fields: source-port class, client ToS, client
    TCP flags / Argus source state, Zeek conn_state (kept only as the
    server-determined outcome, see CONN_OUTCOMES), DNS recursion-desired bit;
  * free continuous header fields a client sets per flow (initial TCP window,
    TCP header size, IP TTL, DNS query type/class). These are not read at all.
"""
import numpy as np
import pandas as pd

from guardnids.data import CTU_BEHAVIOR, NON_FEATURES, is_ctu

FREE_HEADER = ("Init_Win_bytes_forward", "min_seg_size_forward", "TCP_WIN_MAX_IN", "MIN_TTL", "MAX_TTL",
               "DNS_QUERY_TYPE", "dns_qtype", "dns_qclass")

CLIENT_PREFIXES = ("SrcPort", "conn_state_", "SrcState_", "flag_", "cflag_")
CLIENT_COLUMNS = ("sTos", "dns_RD")

# Zeek connection states grouped by what the responder decided. The originator
# can move between states of one group (e.g. close with RST instead of FIN) but
# cannot turn a rejected or unanswered connection into an established one.
CONN_OUTCOMES = {
    "noreply": ["S0", "SH", "RSTOS0"],
    "rejected": ["REJ"],
    "established": ["SF", "S1", "S2", "S3", "RSTO", "RSTR"],
    "responder_only": ["SHR", "RSTRH"],
    "other": ["OTH"],
}


def is_free_header(col):
    return col in FREE_HEADER


def is_client_controllable(col):
    return col.startswith(CLIENT_PREFIXES) or col in CLIENT_COLUMNS


def is_binary(values):
    return bool(np.all(np.isin(np.unique(values), [0.0, 1.0])))


def split_key_columns(feature_cols, sample, ctu):
    """Split the model's features into behavior and invariant-key columns.

    Returns (behavior_idx, key_idx). `key_idx` holds column indices and, for
    Zeek conn_state, lists of indices that are OR-ed into one outcome bit.
    `sample` is a (rows, features) array used to detect binary columns on
    datasets without a fixed schema.
    """
    if ctu:
        behavior = [feature_cols.index(c) for c in CTU_BEHAVIOR]
        discrete = [i for i, c in enumerate(feature_cols) if c not in CTU_BEHAVIOR and c != "Proto"]
    else:
        discrete = [i for i in range(len(feature_cols)) if is_binary(sample[:, i])]
        behavior = [i for i in range(len(feature_cols)) if i not in set(discrete)]

    key = [i for i in discrete if not is_client_controllable(feature_cols[i])]
    for states in CONN_OUTCOMES.values():
        group = [feature_cols.index(f"conn_state_{s}") for s in states if f"conn_state_{s}" in feature_cols]
        if group:
            key.append(group)
    return behavior, key


def discrete_columns(flows):
    """All discrete (signature) columns of a raw flow frame, controllable or not."""
    feats = [c for c in flows.columns if c not in NON_FEATURES]
    if is_ctu(flows):
        return [c for c in feats if c not in CTU_BEHAVIOR and c != "Proto"]
    return [c for c in feats if np.all(np.isin(pd.unique(flows[c]), [0, 1]))]

