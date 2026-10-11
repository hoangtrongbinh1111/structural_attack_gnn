"""Feasible and adaptive attacks.

Padding attacks go through guardnids.physics.pad_flows with HELD_OUT_PHYSICS
(52-byte headers, one ACK per segment), which GuardNIDS-RR's anchors never use;
`padding_anchor_physics` repeats the padding with the anchors' own physics.

  padding             physical padding at steps 1, 3, 5, 7, 9
  over_budget         step-9 padding scaled x1 .. x20
  cde                 each malicious flow grows toward the benign centroid (fraction alpha)
  cea                 CDE + C2xB injection (alpha x beta grid)
  port_class_padding  source-port class rewritten (best class kept) + step-9 padding
  header_edits        client-set fields only: source-port class, controllable flag/state
                      bits, free header fields (best edit kept, no padding)
  port_choice         new source port per malicious flow, i.e. a new client node
  routing_poison      benign flows from C carrying the discrete fields of C's malicious flows
  benign_padding      control: benign flows padded, malicious untouched
"""
import numpy as np
import pandas as pd

from guardnids.attacks.common import FLOAT_COLUMNS, as_float, malicious_mask, score
from guardnids.attacks.feature import STEPS
from guardnids.attacks.structure import AMOUNTS, benign_from_controlled
from guardnids.data import NON_FEATURES
from guardnids.invariance import CONN_OUTCOMES, FREE_HEADER, discrete_columns, is_client_controllable
from guardnids.physics import ANCHOR_PHYSICS, HELD_OUT_PHYSICS, MSS, duration_unit, exporter_caps, pad_flows

PADDING_STEPS = [1, 3, 5, 7, 9]
OVER_BUDGET_FACTORS = [1, 2, 5, 10, 20]
CDE_ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
CEA_ALPHAS = [0.3, 0.5, 0.7, 0.9]
CEA_BETAS = [1, 5, 20]
HEADER_CANDIDATES = 10
EDITED = ["Dur", "SrcBytes", "DstBytes", "TotPkts"]
SRC_PORT_CLASSES = {"SrcPortWellKnown": (1, 1023), "SrcPortRegistered": (1024, 49151), "SrcPortPrivate": (49152, 65535)}


def _pad_malicious(flows, physics, dur=0.0, src=0.0, dst=0.0, pkts=0.0):
    flows = flows.copy()
    numeric = [c for c in flows.columns if c not in NON_FEATURES]
    flows[numeric] = flows[numeric].astype("float64")
    mal = flows.Label == 1
    flows.loc[mal, :] = pad_flows(flows[mal], physics, add_dur=dur, add_src=src, add_dst=dst, add_pkts=pkts,
                                  caps=exporter_caps(flows))
    return flows


def _pad_step(flows, physics, step, factor=1):
    o = STEPS[step]
    return _pad_malicious(flows, physics, factor * o["Dur"], factor * o["SrcBytes"],
                          factor * o["DstBytes"], factor * o["TotPkts"])


def _set_src_port_class(flows, cls, cols):
    flows[cols] = flows[cols].astype("float64")
    mal = flows.index[flows.Label == 1]
    for c in cols:
        flows.loc[mal, c] = 1.0 if c == cls else 0.0
    return flows


# --------------------------------------------------------------------------- padding
def padding_attack(detector, flows, physics=HELD_OUT_PHYSICS):
    return [score(detector, _pad_step(flows, physics, s), "padding", s) for s in PADDING_STEPS]


def padding_anchor_physics_attack(detector, flows):
    return padding_attack(detector, flows, ANCHOR_PHYSICS)


def over_budget_attack(detector, flows):
    return [score(detector, _pad_step(flows, HELD_OUT_PHYSICS, 9, m), "over_budget", m) for m in OVER_BUDGET_FACTORS]


def port_class_padding_attack(detector, flows):
    cols = [c for c in SRC_PORT_CLASSES if c in flows.columns]
    rows = []
    for cls in cols:
        attacked = _set_src_port_class(as_float(flows), cls, cols)
        rows.append(score(detector, _pad_step(attacked, HELD_OUT_PHYSICS, 9), f"srcport:{cls}", 9))
    return rows


def benign_padding_attack(detector, flows):
    rows = []
    for s in PADDING_STEPS:
        o = STEPS[s]
        attacked = flows.copy()
        numeric = [c for c in attacked.columns if c not in NON_FEATURES]
        attacked[numeric] = attacked[numeric].astype("float64")
        ben = attacked.Label == 0
        attacked.loc[ben, :] = pad_flows(attacked[ben], HELD_OUT_PHYSICS, o["Dur"], o["SrcBytes"], o["DstBytes"],
                                         o["TotPkts"], caps=exporter_caps(attacked))
        rows.append(score(detector, attacked, "benign_padding", s))
    return rows


# --------------------------------------------------------------------------- centroid steering
def _cde_increments(flows, alpha):
    """Move a fraction alpha toward the benign centroid, only on statistics the client can grow."""
    centroid = flows[flows.Label == 0][EDITED].astype(float).mean()
    mal = flows[flows.Label == 1]
    inc = {c: (alpha * (centroid[c] - mal[c].astype(float))).clip(lower=0).to_numpy() for c in EDITED}
    inc["Dur"] = inc["Dur"] / duration_unit(flows)
    inc["TotPkts"] = np.maximum(inc["TotPkts"], np.ceil(inc["SrcBytes"] / MSS))  # packets to carry the bytes
    return inc


def _cde(flows, alpha):
    inc = _cde_increments(flows, alpha)
    return _pad_malicious(flows, HELD_OUT_PHYSICS, inc["Dur"], inc["SrcBytes"], inc["DstBytes"], inc["TotPkts"])


def cde_attack(detector, flows):
    return [score(detector, _cde(flows, a), "cde", a) for a in CDE_ALPHAS]


def cea_attack(detector, flows):
    rows = []
    for a in CEA_ALPHAS:
        steered = _cde(flows, a)
        for b in CEA_BETAS:
            attacked = pd.concat([steered, benign_from_controlled(steered, b)], ignore_index=True)
            rows.append(score(detector, attacked, f"alpha={a}", b))
    return rows


# --------------------------------------------------------------------------- client-set fields
def header_edits_attack(detector, flows):
    """Variants are grouped by prefix; aggregation keeps the best one per group."""
    rows = []
    port_cols = [c for c in SRC_PORT_CLASSES if c in flows.columns]
    for cls in port_cols:
        rows.append(score(detector, _set_src_port_class(as_float(flows), cls, port_cols), f"srcport:{cls}", 0))
    rows.append(score(detector, _benign_controllable_bits(flows), "ctrl_bits", 0))
    rows.extend(_free_header_edits(detector, flows))
    return rows


def _benign_controllable_bits(flows):
    """Controllable discrete fields of every malicious flow set to the most common
    benign values. Zeek conn_state only moves inside its responder outcome group."""
    controllable = [c for c in discrete_columns(flows) if is_client_controllable(c)]
    conn = [c for c in controllable if c.startswith("conn_state_")]
    bits = [c for c in controllable if not c.startswith("conn_state_")]
    benign = flows[flows.Label == 0]
    attacked = as_float(flows)
    mal = attacked.index[attacked.Label == 1]

    if bits:
        mode = benign[bits].astype(float).astype(str).agg("|".join, axis=1).mode().iloc[0].split("|")
        attacked[bits] = attacked[bits].astype("float64")
        for c, v in zip(bits, mode):
            attacked.loc[mal, c] = float(v)
    if conn:
        attacked[conn] = attacked[conn].astype("float64")
        current = attacked.loc[mal, conn].idxmax(axis=1).str.replace("conn_state_", "")
        benign_counts = benign[conn].sum()
        for states in CONN_OUTCOMES.values():
            group = [f"conn_state_{s}" for s in states if f"conn_state_{s}" in conn]
            if not group:
                continue
            target = benign_counts[group].idxmax()
            in_group = current.index[current.isin(states)]
            attacked.loc[in_group, conn] = 0.0
            attacked.loc[in_group, target] = 1.0
    return attacked


def _free_header_edits(detector, flows):
    cols = [c for c in FREE_HEADER if c in flows.columns]
    if not cols:
        return []
    candidates = flows[flows.Label == 0][cols].astype(float).value_counts().index[:HEADER_CANDIDATES]
    rows = []
    for k, values in enumerate(candidates):
        attacked = flows.copy()
        attacked[cols] = attacked[cols].astype("float64")
        mal = attacked.index[attacked.Label == 1]
        for c, v in zip(cols, values if isinstance(values, tuple) else (values,)):
            attacked.loc[mal, c] = v
        rows.append(score(detector, attacked, f"hdr:{k}", 0))
    return rows


def _split_endpoint(s):
    parts = s.astype(str).str.rsplit(":", n=1)
    return parts.str[0], pd.to_numeric(parts.str[1], errors="coerce")


def port_choice_attack(detector, flows):
    """Nodes are IP:port, so a client that picks a new source port creates a new node.
    Client-side malicious flows (source port above destination port) get a new port in
    each class, and the class bits follow:
      fresh: one new port per flow;
      reuse: the k-th flow of a host to one destination uses port base + k."""
    cols = [c for c in SRC_PORT_CLASSES if c in flows.columns]
    src_ip, src_port = _split_endpoint(flows["Src IP"])
    _, dst_port = _split_endpoint(flows["Dst IP"])
    client_side = malicious_mask(flows) & (src_port > dst_port).fillna(False).to_numpy()
    idx = np.where(client_side)[0]
    reuse_rank = flows.iloc[idx].groupby([src_ip.iloc[idx].values,
                                          flows["Dst IP"].iloc[idx].values]).cumcount().to_numpy()
    rows = []
    for cls in (cols or ["SrcPortPrivate"]):
        low, high = SRC_PORT_CLASSES[cls]
        for variant, ranks in (("fresh", np.arange(len(idx))), ("reuse", reuse_rank)):
            attacked = as_float(flows, FLOAT_COLUMNS + cols)
            for c in cols:
                attacked.iloc[idx, attacked.columns.get_loc(c)] = 1.0 if c == cls else 0.0
            ports = low + ranks % (high - low + 1)
            attacked["Src IP"] = attacked["Src IP"].astype(object)
            attacked.iloc[idx, attacked.columns.get_loc("Src IP")] = [
                f"{src_ip.iloc[i]}:{int(p)}" for i, p in zip(idx, ports)]
            rows.append(score(detector, attacked, f"{cls}:{variant}", 0))
    return rows


def routing_poison_attack(detector, flows, seed=0):
    """Benign flows sent from each C, with the discrete fields copied from C's own
    malicious flows, so retrieval routes them into C's malicious channel."""
    discrete = discrete_columns(flows)
    rng = np.random.RandomState(seed)
    rows = []
    for amount in AMOUNTS:
        injected = []
        benign = flows[flows.Label == 0]
        for host, own in flows[flows.Label == 1].groupby("Src IP"):
            extra = benign.sample(amount, replace=True, random_state=rng.randint(1 << 30)).copy()
            donor = own.sample(amount, replace=True, random_state=rng.randint(1 << 30))
            extra[discrete] = donor[discrete].values
            extra["Src IP"] = host
            extra["Label"] = 0
            injected.append(extra)
        attacked = pd.concat([flows, pd.concat(injected, ignore_index=True)], ignore_index=True)
        rows.append(score(detector, attacked, "poison", amount))
    return rows
