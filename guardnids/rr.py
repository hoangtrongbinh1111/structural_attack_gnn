"""GuardNIDS-RR: invariant-key retrieval + evidence-routed message passing.

1. Reference set. Up to 6,000 clean training flows (half malicious) plus padded
   copies of malicious training flows ("anchors"), padded the way a client can
   pad its own flows (guardnids.physics).
2. Invariant-key retrieval. Every flow retrieves its K nearest references,
   ranked by Jaccard similarity over the invariant key (fields the attacker
   cannot set, guardnids.invariance). Cosine over behavior features only breaks
   ties. Two posteriors are read off the retrieved labels:
     p_top: mean label of the K retrieved references,
     p_inv: mean label of every reference that ties or beats the K-th best key
            score, so the cosine tie-break (which padding can steer) never
            decides which labels are counted.
   The edge posterior is p = max(p_inv, p_top): an evasion attacker only gains
   by lowering p, and cannot lower p_inv.
3. Evidence routing. Each layer keeps two weighted means per node: a malicious
   channel weighted by p and a benign channel weighted by 1 - p. Benign-looking
   flows injected at an attacker's node only move its benign channel.

Message edge features are the mean feature vector of the retrieved references
(plus p), so a flow's own editable statistics reach the model only through
which references it retrieves.
"""
from dataclasses import asdict, dataclass

import dgl.function as fn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from guardnids.invariance import split_key_columns
from guardnids.physics import ANCHOR_PHYSICS, exporter_caps, pad_flows

REFERENCE_CAP = 6000
QUERY_CHUNK = 1024
TIE_EPS = 1e-3
HIDDEN = 128
DROPOUT = 0.2

# per-feature padding budget of the anchors (largest step of the padding attack)
ANCHOR_BUDGET = {"Dur": 120, "SrcBytes": 1024, "DstBytes": 1024, "TotPkts": 100}


@dataclass
class RRConfig:
    """Defaults are GuardNIDS-RR; the other values are the ablations."""
    k: int = 7
    key: str = "invariant"       # 'invariant' | 'cosine' (rank by behavior cosine only)
    posterior: str = "max"       # 'max' | 'inv' (p_inv only) | 'top' (p_top only)
    n_anchors: int = 3000        # 0 disables the padded anchors
    routing: bool = True         # False: single mean channel, p only as an edge feature
    routing_control: str = ""    # 'const' (p = 0.5) | 'shuffle' (p permuted across edges)
    backbone: str = "sage"       # 'sage' (mean) | 'gat' (attention inside each channel)

    def run_name(self, base="guardnids_rr"):
        parts = [base]
        if self.key != "invariant":
            parts.append(f"key-{self.key}")
        if self.posterior != "max":
            parts.append(f"posterior-{self.posterior}")
        if self.n_anchors == 0:
            parts.append("no-anchors")
        if not self.routing:
            parts.append("no-routing")
        if self.routing_control:
            parts.append(f"routing-{self.routing_control}")
        if self.backbone != "sage":
            parts.append(self.backbone)
        return "_".join(parts)


# --------------------------------------------------------------------------- retrieval
@dataclass
class ReferenceSet:
    features: torch.Tensor   # (n_ref, d) scaled features
    key: torch.Tensor        # (n_ref, k) 0/1 invariant key
    behavior: torch.Tensor   # (n_ref, b) L2-normalized behavior features
    labels: torch.Tensor     # (n_ref,)
    behavior_idx: list
    key_idx: list
    feature_cols: list

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d, device):
        tensors = {k: v.to(device) for k, v in d.items() if isinstance(v, torch.Tensor)}
        return cls(**{**d, **tensors})

    @property
    def device(self):
        return self.features.device


def encode_keys(features, behavior_idx, key_idx):
    if any(isinstance(i, list) for i in key_idx):
        bits = [(features[:, i] > 0).any(1) if isinstance(i, list) else features[:, i] > 0 for i in key_idx]
        key = torch.stack(bits, 1).float()
    else:
        key = (features[:, key_idx] > 0).float()
    behavior = F.normalize(features[:, behavior_idx], dim=1, eps=1e-8)
    return key, behavior


def padded_anchors(raw_flows, rng, caps):
    """Pad each flow on a random non-empty subset of (Dur, SrcBytes, DstBytes, TotPkts),
    with log-uniform magnitudes up to ANCHOR_BUDGET."""
    cols = list(ANCHOR_BUDGET)
    pick = rng.rand(len(raw_flows), len(cols)) < 0.5
    pick[pick.sum(1) == 0, rng.randint(len(cols))] = True
    inc = {}
    for j, c in enumerate(cols):
        magnitude = np.exp(rng.uniform(0, np.log(ANCHOR_BUDGET[c]), len(raw_flows)))
        inc[c] = pick[:, j] * magnitude
    return pad_flows(raw_flows, ANCHOR_PHYSICS, add_dur=inc["Dur"], add_src=inc["SrcBytes"],
                     add_dst=inc["DstBytes"], add_pkts=inc["TotPkts"], caps=caps)


def build_reference_set(train, train_raw, scaler, cols_to_scale, feature_cols, cfg, seed, ctu, device):
    """`train` is the scaled training frame, `train_raw` the same rows unscaled.

    Returns (references, owner): owner[j] is the training row reference j comes
    from (anchors carry their source row), so a training flow never retrieves
    itself or its own padded copies.
    """
    rng = np.random.RandomState(seed)
    rows = np.arange(len(train))
    labels = train["Label"].to_numpy()
    mal_rows, ben_rows = rows[labels == 1], rows[labels == 0]
    n_mal = min(len(mal_rows), REFERENCE_CAP // 2)
    n_ben = min(len(ben_rows), REFERENCE_CAP - n_mal)
    owner = np.concatenate([rng.choice(mal_rows, n_mal, replace=False),
                            rng.choice(ben_rows, n_ben, replace=False)])

    all_features = train[feature_cols].to_numpy(dtype=np.float32)
    behavior_idx, key_idx = split_key_columns(feature_cols, all_features[owner], ctu)
    ref_features, ref_labels = all_features[owner], labels[owner]

    if cfg.n_anchors > 0 and len(mal_rows) > 0:
        sources = rng.choice(mal_rows, cfg.n_anchors, replace=True)
        anchors = padded_anchors(train_raw.iloc[sources], rng, exporter_caps(train_raw))
        anchors[cols_to_scale] = scaler.transform(anchors[cols_to_scale])
        ref_features = np.concatenate([ref_features, anchors[feature_cols].to_numpy(dtype=np.float32)])
        ref_labels = np.concatenate([ref_labels, np.ones(cfg.n_anchors, dtype=ref_labels.dtype)])
        owner = np.concatenate([owner, sources])

    features = torch.tensor(np.ascontiguousarray(ref_features), device=device)
    key, behavior = encode_keys(features, behavior_idx, key_idx)
    refs = ReferenceSet(features, key, behavior, torch.tensor(ref_labels, dtype=torch.float32, device=device),
                        behavior_idx, key_idx, list(feature_cols))
    return refs, owner


def retrieve(refs, queries, cfg, query_rows=None, owner=None):
    """Return (f, p) per query: mean features of the retrieved references and the
    retrieved malicious posterior. With query_rows/owner, a query never retrieves
    a reference owned by its own training row."""
    q = torch.tensor(np.ascontiguousarray(queries, dtype=np.float32), device=refs.device)
    q_key, q_behavior = encode_keys(q, refs.behavior_idx, refs.key_idx)
    ref_key_size = refs.key.sum(1)
    owner_t = torch.tensor(owner, device=refs.device) if query_rows is not None else None

    f_out, p_out = [], []
    for start in range(0, len(q), QUERY_CHUNK):
        end = min(start + QUERY_CHUNK, len(q))
        cosine = q_behavior[start:end] @ refs.behavior.T
        if cfg.key == "invariant":
            inter = q_key[start:end] @ refs.key.T
            union = q_key[start:end].sum(1, keepdim=True) + ref_key_size - inter
            jaccard = inter / union.clamp(min=1)
            score = jaccard + TIE_EPS * cosine
        else:
            score = cosine
        if owner_t is not None:
            rows = torch.tensor(query_rows[start:end], device=refs.device)
            own = rows[:, None] == owner_t[None, :]
            score = score.masked_fill(own, -1e9)

        top = score.topk(cfg.k, dim=1).indices
        f_out.append(refs.features[top].mean(1))
        p_top = refs.labels[top].mean(1)
        if cfg.key == "invariant" and cfg.posterior != "top":
            if owner_t is not None:
                jaccard = jaccard.masked_fill(own, -1.0)
            kth = jaccard.topk(cfg.k, dim=1).values[:, -1:]
            ties = (jaccard >= kth - 1e-6).float()
            p_inv = (ties @ refs.labels) / ties.sum(1)
            p_out.append(torch.maximum(p_inv, p_top) if cfg.posterior == "max" else p_inv)
        else:
            p_out.append(p_top)

    p = torch.cat(p_out)
    if cfg.routing_control == "const":
        p = torch.full_like(p, 0.5)
    elif cfg.routing_control == "shuffle":
        perm = torch.randperm(len(p), generator=torch.Generator(device="cpu").manual_seed(0))
        p = p[perm.to(p.device)]
    return torch.cat(f_out), p


# --------------------------------------------------------------------------- model
class RoutedLayer(nn.Module):
    def __init__(self, in_dim, edge_dim, out_dim, routing, attention):
        super().__init__()
        self.routing = routing
        self.attention = attention
        self.W_msg = nn.Linear(in_dim + edge_dim, out_dim)
        n_channels = 2 if routing else 1
        self.W_apply = nn.Linear(in_dim + n_channels * out_dim, out_dim)
        if attention:
            self.attn_fc = nn.Linear(out_dim, 1)

    @staticmethod
    def _weighted_mean(g, weights, eps):
        g.edata["w"] = weights.unsqueeze(1)
        g.edata["wm"] = g.edata["m"] * g.edata["w"]
        g.update_all(fn.copy_e("wm", "x"), fn.sum("x", "num"))
        g.update_all(fn.copy_e("w", "x"), fn.sum("x", "den"))
        return g.ndata["num"] / g.ndata["den"].clamp(min=eps)

    def forward(self, g, h, e, p_malicious):
        with g.local_scope():
            g.ndata["h"] = h
            g.edata["e"] = e
            g.apply_edges(lambda edges: {"m": self.W_msg(torch.cat([edges.src["h"], edges.data["e"]], 1))})
            att = None
            if self.attention:
                score = F.leaky_relu(self.attn_fc(g.edata["m"]).squeeze(1), 0.2)
                att = torch.exp(score - score.max())  # normalized per node by the weighted mean

            if self.routing:
                channels = []
                for w in (p_malicious, 1.0 - p_malicious):
                    channels.append(self._weighted_mean(g, w if att is None else w * att, eps=1e-6))
                agg = torch.cat(channels, 1)
            elif att is not None:
                agg = self._weighted_mean(g, att, eps=1e-12)
            else:
                g.update_all(fn.copy_e("m", "x"), fn.mean("x", "agg"))
                agg = g.ndata["agg"]
            return F.relu(self.W_apply(torch.cat([h, agg], 1)))


class GuardNIDSRR(nn.Module):
    def __init__(self, n_features, routing=True, backbone="sage"):
        super().__init__()
        attention = backbone == "gat"
        edge_dim = n_features + 1  # retrieved features + p
        self.l1 = RoutedLayer(n_features, edge_dim, HIDDEN, routing, attention)
        self.l2 = RoutedLayer(HIDDEN, edge_dim, HIDDEN, routing, attention)
        self.dropout = nn.Dropout(DROPOUT)
        self.classifier = nn.Linear(2 * HIDDEN, 2)

    def forward(self, g, node_feats, f, p):
        """f: (E, d) retrieved edge features, p: (E,) retrieved malicious posterior,
        both already in the edge order of the graph."""
        e = torch.cat([f, p.unsqueeze(1)], 1)
        h = self.l1(g, node_feats, e, p)
        h = self.l2(g, self.dropout(h), e, p)
        src, dst = g.edges()
        return self.classifier(torch.cat([h[src], h[dst]], 1))
