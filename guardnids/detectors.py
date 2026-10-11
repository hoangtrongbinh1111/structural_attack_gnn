"""Trained detectors behind one interface: `predict(raw_flows) -> (y_true, y_pred)`.

A run directory holds `meta.json`, `scaler.joblib`, `model.pt` and, for
GuardNIDS-RR, `references.pt`.
"""
import json
import os

import joblib
import numpy as np
import torch

from guardnids.baselines import build_baseline
from guardnids.graph import build_flow_graph
from guardnids.rr import GuardNIDSRR, ReferenceSet, RRConfig, retrieve


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BaselineDetector:
    """E-GraphSAGE, GNNGuard, E-GAT and AdvTrain (an adversarially trained E-GraphSAGE)."""

    def __init__(self, net, scaler, feature_cols):
        self.net = net.eval()
        self.scaler = scaler
        self.feature_cols = feature_cols

    @torch.no_grad()
    def predict(self, flows):
        flows = scale(flows, self.scaler)
        dev = next(self.net.parameters()).device
        g, edge_flow = build_flow_graph(flows, dev, self_loops_once=True)
        x = torch.tensor(flows[self.feature_cols].to_numpy(dtype=np.float32)[edge_flow], device=dev)
        node_feats = torch.ones(g.num_nodes(), len(self.feature_cols), device=dev)
        pred = self.net(g, node_feats, x).argmax(1).cpu().numpy()
        return flows.Label.to_numpy()[edge_flow], pred


class GuardNIDSRRDetector:
    def __init__(self, net, scaler, refs, cfg):
        self.net = net.eval()
        self.scaler = scaler
        self.refs = refs
        self.cfg = cfg

    @torch.no_grad()
    def predict(self, flows):
        flows = scale(flows, self.scaler)
        f, p = retrieve(self.refs, flows[self.refs.feature_cols].to_numpy(dtype=np.float32), self.cfg)
        g, edge_flow = build_flow_graph(flows, self.refs.device)
        idx = torch.tensor(edge_flow, device=self.refs.device)
        node_feats = torch.ones(g.num_nodes(), len(self.refs.feature_cols), device=self.refs.device)
        pred = self.net(g, node_feats, f[idx], p[idx]).argmax(1).cpu().numpy()
        return flows.Label.to_numpy()[edge_flow], pred


class RetrievalOnlyDetector:
    """Control without the graph: a flow is malicious iff its retrieved posterior p >= 0.5."""

    def __init__(self, scaler, refs, cfg):
        self.scaler = scaler
        self.refs = refs
        self.cfg = cfg

    @torch.no_grad()
    def predict(self, flows):
        flows = scale(flows, self.scaler)
        _, p = retrieve(self.refs, flows[self.refs.feature_cols].to_numpy(dtype=np.float32), self.cfg)
        return flows.Label.astype(int).to_numpy(), (p.cpu().numpy() >= 0.5).astype(int)


def scale(flows, scaler):
    flows = flows.reset_index(drop=True).copy()
    cols = list(scaler.feature_names_in_)
    flows[cols] = scaler.transform(flows[cols])
    return flows


# --------------------------------------------------------------------------- persistence
def save_run(run_dir, kind, net, scaler, feature_cols, cfg=None, refs=None):
    os.makedirs(run_dir, exist_ok=True)
    meta = {"kind": kind, "feature_cols": list(feature_cols)}
    if cfg is not None:
        meta["rr_config"] = vars(cfg)
    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    joblib.dump(scaler, os.path.join(run_dir, "scaler.joblib"))
    torch.save(net.state_dict(), os.path.join(run_dir, "model.pt"))
    if refs is not None:
        torch.save(refs.to_dict(), os.path.join(run_dir, "references.pt"))


def load_detector(run_dir, retrieval_only=False):
    with open(os.path.join(run_dir, "meta.json")) as f:
        meta = json.load(f)
    scaler = joblib.load(os.path.join(run_dir, "scaler.joblib"))
    state = torch.load(os.path.join(run_dir, "model.pt"), map_location=device())
    n_features = len(meta["feature_cols"])

    if meta["kind"] == "guardnids_rr":
        cfg = RRConfig(**meta["rr_config"])
        refs = ReferenceSet.from_dict(torch.load(os.path.join(run_dir, "references.pt"), map_location=device()),
                                      device())
        if retrieval_only:
            return RetrievalOnlyDetector(scaler, refs, cfg)
        net = GuardNIDSRR(n_features, routing=cfg.routing, backbone=cfg.backbone).to(device())
        net.load_state_dict(state)
        return GuardNIDSRRDetector(net, scaler, refs, cfg)

    if retrieval_only:
        raise ValueError("the retrieval-only control needs a GuardNIDS-RR run")
    net = build_baseline(meta["kind"], n_features).to(device())
    net.load_state_dict(state)
    return BaselineDetector(net, scaler, meta["feature_cols"])
