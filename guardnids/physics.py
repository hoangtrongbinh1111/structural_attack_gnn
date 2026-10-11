"""Physically consistent padding of flow records.

A client that pads its own flow cannot choose the flow statistics freely:
  * payload can only be added on established connections;
  * every added packet carries a header, counted wherever the exporter counts
    L3 (or, on CTU-13/Argus, Ethernet) bytes;
  * the server acknowledges added segments (one ACK per `ack_every` segments);
  * the response size only grows where the server already sends payload;
  * duration and packet count stay below the exporter's limits;
  * derived statistics (totals, rates, per-packet means, ...) follow.

`pad_flows` applies requested increments under these rules. GuardNIDS-RR uses
it with ANCHOR_PHYSICS to build its padded anchors; the evaluation attacks use
HELD_OUT_PHYSICS (TCP timestamps, one ACK per segment), which the anchors never
contain, and ANCHOR_PHYSICS as a sanity check.
"""
from dataclasses import dataclass

import numpy as np

from guardnids.data import NON_FEATURES, dataset_of

MSS = 1460
ESTABLISHED_STATES = ("SF", "S1", "S2", "S3", "RSTO", "RSTR")


@dataclass(frozen=True)
class Physics:
    header_bytes: int
    ack_every: int


ANCHOR_PHYSICS = Physics(header_bytes=40, ack_every=2)
HELD_OUT_PHYSICS = Physics(header_bytes=52, ack_every=1)


def duration_unit(flows):
    """CIC-IDS2017 durations are in microseconds, the other datasets use seconds."""
    return 1e6 if "Flow Bytes/s" in flows.columns else 1.0


def established(flows):
    """Connections on which the client can send payload and the server acknowledges."""
    zeek = [f"conn_state_{s}" for s in ESTABLISHED_STATES if f"conn_state_{s}" in flows.columns]
    if zeek:
        return (flows[zeek].astype(float).sum(1) > 0).to_numpy()
    if "sflag_SYN" in flows.columns and "sflag_ACK" in flows.columns:
        return ((flows["sflag_SYN"] > 0) & (flows["sflag_ACK"] > 0)).to_numpy()
    if "Init_Win_bytes_backward" in flows.columns:
        return (flows["Init_Win_bytes_backward"] > 0).to_numpy()
    # CTU-13 keeps no handshake fields: a reply from the server, except a bare RST
    # answering a SYN (Argus counts the RST frame as DstBytes)
    est = (flows["DstBytes"] > 0).to_numpy()
    if {"SrcState_S", "DstState_S", "DstState_R"} <= set(flows.columns):
        rejected = ((flows["SrcState_S"] > 0) & (flows["DstState_S"] == 0) & (flows["DstState_R"] > 0)).to_numpy()
        est = est & ~rejected
    return est


def exporter_caps(flows):
    """Longest duration (exporter timeout) and, on CTU-13, the packet-count limit."""
    caps = {"Dur": float(flows["Dur"].max())}
    if dataset_of(flows) == "CTU":
        caps["TotPkts"] = float(flows["TotPkts"].max())
    return caps


def _as_array(x, n):
    a = np.asarray(x, dtype=np.float64)
    return np.full(n, float(a)) if a.ndim == 0 else a


def pad_flows(flows, physics, add_dur=0.0, add_src=0.0, add_dst=0.0, add_pkts=0.0, caps=None):
    """Return a padded copy of the raw (unscaled) flow frame.

    Increments are scalars or per-row arrays (>= 0); `add_dur` is in seconds,
    `add_pkts` counts client packets.
    """
    before = flows.copy()
    numeric = [c for c in before.columns if c not in NON_FEATURES]
    before[numeric] = before[numeric].astype("float64")
    n = len(before)
    unit = duration_unit(before)
    add_dur, add_src, add_dst = _as_array(add_dur, n), _as_array(add_src, n), _as_array(add_dst, n)
    add_pkts = np.round(_as_array(add_pkts, n))

    if caps:
        room = np.clip(caps["Dur"] - before["Dur"].to_numpy(), 0, None) / unit
        add_dur = np.minimum(add_dur, room)
        if "TotPkts" in caps:
            # every client packet draws 1/ack_every server ACKs: keep the total under the cap
            room_pkts = np.clip(caps["TotPkts"] - before["TotPkts"].to_numpy(), 0, None)
            add_pkts = np.minimum(add_pkts, np.floor(room_pkts / (1 + 1 / physics.ack_every)))

    est = established(before)
    answered = before["DstBytes"].to_numpy() > 0
    add_src = np.where(est, add_src, 0.0)
    add_dst = np.where(est & answered, add_dst, 0.0)
    acks = np.where(est, add_pkts // physics.ack_every, 0.0)

    after = _pad_counters(before, physics, add_dur * unit, add_src, add_dst, add_pkts, acks)
    ds = dataset_of(before)
    if ds == "IDS2017":
        _update_cic_statistics(before, after, add_dur * unit, add_src, add_pkts, acks)
    elif ds == "NF_UNSW":
        _update_netflow_statistics(before, after, physics, add_dur, add_src, add_pkts, answered)

    after = after.replace([np.inf, -np.inf], np.nan)
    after[numeric] = after[numeric].fillna(0.0)
    return after


def _pad_counters(before, physics, add_dur, add_src, add_dst, add_pkts, acks):
    """Byte/packet counters, header bytes and the simple derived statistics."""
    v = before.copy()
    ds = dataset_of(v)
    v["Dur"] = v["Dur"] + add_dur
    v["SrcBytes"] = v["SrcBytes"] + add_src
    v["DstBytes"] = v["DstBytes"] + add_dst
    if ds in ("CTU", "NF_UNSW"):
        header = physics.header_bytes
        if ds == "CTU":
            header = max(header + 14, 60)  # Argus counts Ethernet frames (60-byte minimum)
        v["SrcBytes"] = v["SrcBytes"] + header * add_pkts
        v["DstBytes"] = v["DstBytes"] + header * acks
    if "SrcPkts" in v.columns and "DstPkts" in v.columns:
        v["SrcPkts"] = v["SrcPkts"] + add_pkts
        v["DstPkts"] = v["DstPkts"] + acks
        v["TotPkts"] = v["SrcPkts"] + v["DstPkts"]
    else:
        v["TotPkts"] = v["TotPkts"] + add_pkts + acks
    v["TotBytes"] = v["SrcBytes"] + v["DstBytes"]

    if ds == "ToN_IoT":
        v["src_ip_bytes"] = v["src_ip_bytes"] + add_src + physics.header_bytes * add_pkts
        v["dst_ip_bytes"] = v["dst_ip_bytes"] + add_dst + physics.header_bytes * acks
    elif ds == "NF_UNSW":
        added = add_pkts + acks
        v["NUM_PKTS_UP_TO_128_BYTES"] = v["NUM_PKTS_UP_TO_128_BYTES"] + added
        for c in ("SHORTEST_FLOW_PKT", "MIN_IP_PKT_LEN"):
            v[c] = np.where(added > 0, np.minimum(v[c], physics.header_bytes), v[c])
    elif ds == "IDS2017":
        secs = (v["Dur"] / 1e6).clip(lower=1e-6)
        v["Fwd Packet Length Mean"] = v["SrcBytes"] / v["SrcPkts"].clip(lower=1)
        v["Bwd Packet Length Mean"] = v["DstBytes"] / v["DstPkts"].clip(lower=1)
        v["Bwd Packet Length Min"] = np.where(acks > 0, 0.0, v["Bwd Packet Length Min"])
        v["Average Packet Size"] = v["TotBytes"] / v["TotPkts"].clip(lower=1)
        v["Packet Length Mean"] = v["TotBytes"] / (v["TotPkts"] + 1)
        v["act_data_pkt_fwd"] = v["act_data_pkt_fwd"] + np.where(add_src > 0, add_pkts, 0.0)
        v["Flow Bytes/s"] = v["TotBytes"] / secs
        v["Flow Packets/s"] = v["TotPkts"] / secs
        v["Bwd Packets/s"] = v["DstPkts"] / secs
    else:
        v["BytesPerPkt"] = v["TotBytes"] / v["TotPkts"].clip(lower=1)
        # undefined ratios (zero duration / zero client bytes) keep their preprocessed value
        v["PktsPerSec"] = np.where(v["Dur"] > 0, v["TotPkts"] / v["Dur"].where(v["Dur"] > 0, 1.0), v["PktsPerSec"])
        v["RatioOutIn"] = np.where(v["SrcBytes"] > 0, v["DstBytes"] / v["SrcBytes"].where(v["SrcBytes"] > 0, 1.0),
                                   v["RatioOutIn"])
    numeric = [c for c in v.columns if c not in NON_FEATURES]
    v = v.replace([np.inf, -np.inf], np.nan)
    v[numeric] = v[numeric].fillna(0.0)
    return v


def _update_cic_statistics(before, after, add_dur, add_src, add_pkts, acks):
    """CICFlowMeter inter-arrival and packet-length statistics."""
    def gap(dur, k):
        return dur / np.maximum(k - 1, 1)

    has_pkts = add_pkts > 0
    # added packets are spread evenly over the flow: minima can only shrink
    after["Fwd IAT Min"] = np.where(has_pkts, np.minimum(before["Fwd IAT Min"], gap(after["Dur"], after["SrcPkts"])),
                                    before["Fwd IAT Min"])
    after["Flow IAT Min"] = np.where(has_pkts,
                                     np.minimum(before["Flow IAT Min"].clip(lower=0), gap(after["Dur"], after["TotPkts"])),
                                     before["Flow IAT Min"])
    after["Bwd IAT Min"] = np.where(acks > 0, np.minimum(before["Bwd IAT Min"], gap(after["Dur"], after["DstPkts"])),
                                    before["Bwd IAT Min"])

    # forward IAT std of the merged sequence: n0 original gaps (mean m0, std s0) plus add_pkts gaps of size g
    n0 = np.maximum(before["SrcPkts"] - 1, 1)
    m0 = before["Dur"] / n0
    g = np.where(has_pkts, add_dur / np.maximum(add_pkts, 1), 0.0)
    n_total = n0 + add_pkts
    mean = (n0 * m0 + add_pkts * g) / n_total
    var = (n0 * (before["Fwd IAT Std"] ** 2 + (m0 - mean) ** 2) + add_pkts * (g - mean) ** 2) / n_total
    after["Fwd IAT Std"] = np.where(has_pkts, np.sqrt(np.clip(var, 0, None)), before["Fwd IAT Std"])

    payload = np.where(add_pkts > 0, add_src / np.maximum(add_pkts, 1), 0.0)
    after["Fwd Packet Length Max"] = np.maximum(before["Fwd Packet Length Max"], payload)
    after["Max Packet Length"] = np.maximum(before["Max Packet Length"], payload)

    # backward length std after adding bare ACKs
    nb = before["DstPkts"].clip(lower=0)
    mb = before["DstBytes"] / nb.clip(lower=1)
    second_moment = before["Bwd Packet Length Std"] ** 2 + mb ** 2
    nb2 = nb + acks
    mb2 = after["DstBytes"] / nb2.clip(lower=1)
    var_b = (nb * second_moment) / nb2.clip(lower=1) - mb2 ** 2
    after["Bwd Packet Length Std"] = np.where(acks > 0, np.sqrt(np.clip(var_b, 0, None)), before["Bwd Packet Length Std"])


def _update_netflow_statistics(before, after, physics, add_dur, add_src, add_pkts, answered):
    """NetFlow v9 per-direction durations, rates, longest packet and size histogram."""
    d_old, d_new = before["Dur"].to_numpy(), after["Dur"].to_numpy()
    after["DURATION_IN"] = before["DURATION_IN"] + np.round(add_dur)
    after["DURATION_OUT"] = before["DURATION_OUT"] + np.where(answered, np.round(add_dur), 0.0)

    for col, side, min_dur in (("SRC_TO_DST_SECOND_BYTES", "SrcBytes", 1.0), ("DST_TO_SRC_SECOND_BYTES", "DstBytes", 1.0),
                               ("SRC_TO_DST_AVG_THROUGHPUT", "SrcBytes", 1e-3),
                               ("DST_TO_SRC_AVG_THROUGHPUT", "DstBytes", 1e-3)):
        byte_ratio = (after[side] / before[side].replace(0, np.nan)).fillna(1.0)
        after[col] = before[col] * byte_ratio * np.maximum(d_old, min_dur) / np.maximum(d_new, min_dur)

    size = physics.header_bytes + np.where(add_pkts > 0, add_src / np.maximum(add_pkts, 1), 0.0)
    for c in ("LONGEST_FLOW_PKT", "MAX_IP_PKT_LEN"):
        after[c] = np.where(add_pkts > 0, np.maximum(before[c], np.minimum(size, 1500)), before[c])

    # client packets go to the histogram bin of their size (ACKs stay in the <=128 bin)
    bins = [(128, "NUM_PKTS_UP_TO_128_BYTES"), (256, "NUM_PKTS_128_TO_256_BYTES"), (512, "NUM_PKTS_256_TO_512_BYTES"),
            (1024, "NUM_PKTS_512_TO_1024_BYTES"), (1514, "NUM_PKTS_1024_TO_1514_BYTES")]
    target = np.select([size <= limit for limit, _ in bins], list(range(len(bins))), default=len(bins) - 1)
    after["NUM_PKTS_UP_TO_128_BYTES"] = after["NUM_PKTS_UP_TO_128_BYTES"] - np.where(target > 0, add_pkts, 0.0)
    for k, (_, col) in enumerate(bins[1:], start=1):
        after[col] = after[col] + np.where(target == k, add_pkts, 0.0)
