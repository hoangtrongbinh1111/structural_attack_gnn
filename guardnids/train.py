"""Train one detector on one scenario.

    python -m guardnids.train --model guardnids_rr --train data/CTU/seed2/menti_train.csv --seed 2

Models: egraphsage, gnnguard, egat, advtrain, guardnids_rr, guardnids_rr_at.
GuardNIDS-RR ablations are selected with the --key/--posterior/--no-* flags.
The run is saved to <runs>/<dataset>/<scenario>/seed<seed>/<run name>/.
"""
import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from tqdm.auto import tqdm

from guardnids.attacks.common import as_float
from guardnids.attacks.feature import pad_on_grid
from guardnids.attacks.structure import benign_from_controlled, benign_to_new_nodes, malicious_from_controlled
from guardnids.baselines import build_baseline
from guardnids.data import (columns_to_scale, dataset_of, feature_columns, is_ctu, scenario_of,
                            subsample_large_scenario)
from guardnids.detectors import device, save_run
from guardnids.graph import build_flow_graph
from guardnids.invariance import is_free_header
from guardnids.rr import GuardNIDSRR, RRConfig, build_reference_set, retrieve

EPOCHS = 200
BASELINES = ("egraphsage", "gnnguard", "egat")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fit(model, forward, labels, weights, epochs):
    """Full-batch training with a per-edge weighted cross-entropy."""
    opt = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    for epoch in tqdm(range(1, epochs + 1), desc="training"):
        model.train()
        out = forward()
        loss = (criterion(out, labels) * weights).sum() / weights.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch % 50 == 0:
            acc = (out.argmax(1) == labels).float().mean().item()
            tqdm.write(f"epoch {epoch}: loss {loss.item():.4f}, acc {acc:.4f}")
    return model


def fit_scaler(flows, cols):
    scaler = MinMaxScaler()
    flows[cols] = scaler.fit_transform(flows[cols])
    return scaler


def separate_copies(copies, first=0):
    """Give every copy its own node namespace, so the copies are disjoint graphs."""
    out = []
    for k, c in enumerate(copies, start=first):
        c = c.copy()
        c["Src IP"] = f"{k}|" + c["Src IP"].astype(str)
        c["Dst IP"] = f"{k}|" + c["Dst IP"].astype(str)
        out.append(c)
    return out


def train_edge_classifier(arch, flows, feature_cols, weights, epochs, self_loops_once):
    dev = device()
    g, edge_flow = build_flow_graph(flows, dev, self_loops_once)
    edge_feats = torch.tensor(flows[feature_cols].to_numpy(dtype=np.float32)[edge_flow], device=dev)
    node_feats = torch.ones(g.num_nodes(), len(feature_cols), device=dev)
    labels = torch.tensor(flows.Label.to_numpy().astype(np.int64)[edge_flow], device=dev)
    weights = torch.tensor(weights.astype(np.float32)[edge_flow], device=dev)
    model = build_baseline(arch, len(feature_cols)).to(dev)
    return fit(model, lambda: model(g, node_feats, edge_feats), labels, weights, epochs)


# --------------------------------------------------------------------------- baselines
def train_baseline(arch, train_path, seed, epochs):
    flows = pd.read_csv(train_path)
    scaler = fit_scaler(flows, columns_to_scale(flows))
    feature_cols = feature_columns(flows)
    flows = subsample_large_scenario(flows, scenario_of(train_path), seed)
    model = train_edge_classifier(arch, flows, feature_cols, np.ones(len(flows)), epochs, self_loops_once=True)
    return dict(kind=arch, net=model, scaler=scaler, feature_cols=feature_cols)


def adversarial_training_set(raw, ctu, seed):
    """Clean training graph + one disjoint copy per adversarial example set:
    grid padding (steps 1, 5, 9), C2xB and C2xM (2 and 5 flows per host), add-node."""
    mal = raw[raw.Label == 1]
    copies = [raw.copy()]
    for step in (1, 5, 9):
        padded = as_float(raw)
        padded.loc[mal.index] = pad_on_grid(mal, step, ctu)
        copies.append(padded)
    injections = []
    for amount in (2, 5):
        injections.append(benign_from_controlled(raw, amount))
        injections.append(malicious_from_controlled(raw, amount))
    injections.append(benign_to_new_nodes(raw, 10, 2))
    copies += [pd.concat([raw, extra], ignore_index=True) for extra in injections]
    return pd.concat(separate_copies(copies), ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


def train_advtrain(train_path, seed, epochs):
    raw = pd.read_csv(train_path)
    raw = subsample_large_scenario(raw, scenario_of(train_path), seed).reset_index(drop=True)
    flows = adversarial_training_set(raw, is_ctu(raw), seed)
    scaler = fit_scaler(flows, columns_to_scale(flows))
    feature_cols = feature_columns(flows)
    # the clean copy weighs as much as all adversarial copies together
    copy_id = flows["Src IP"].str.split("|", n=1).str[0].astype(int).to_numpy()
    weights = np.where(copy_id == 0, float(copy_id.max()), 1.0)
    # unlike the plain baselines, AdvTrain's graph keeps both directions of a self-loop flow
    model = train_edge_classifier("egraphsage", flows, feature_cols, weights, epochs, self_loops_once=False)
    return dict(kind="egraphsage", net=model, scaler=scaler, feature_cols=feature_cols)


# --------------------------------------------------------------------------- GuardNIDS-RR
def train_guardnids_rr(train_path, seed, cfg, epochs, adversarial=False):
    flows = pd.read_csv(train_path)
    raw = flows.copy()
    ctu = is_ctu(flows)
    scale_cols = columns_to_scale(flows)
    scaler = fit_scaler(flows, scale_cols)
    feature_cols = [c for c in feature_columns(flows) if not is_free_header(c)]

    flows = subsample_large_scenario(flows, scenario_of(train_path), seed)
    raw = raw.loc[flows.index].reset_index(drop=True)
    flows = flows.reset_index(drop=True)

    dev = device()
    refs, owner = build_reference_set(flows, raw, scaler, scale_cols, feature_cols, cfg, seed, ctu, dev)
    print(f"reference set: {len(owner)} flows, {len(refs.behavior_idx)} behavior / {len(refs.key_idx)} key columns")
    f, p = retrieve(refs, flows[feature_cols].to_numpy(dtype=np.float32), cfg,
                    query_rows=np.arange(len(flows)), owner=owner)
    weights = np.ones(len(flows), dtype=np.float32)

    if adversarial:
        flows, f, p, weights = _with_injected_copies(flows, raw, f, p, refs, owner, scaler, scale_cols, cfg)

    g, edge_flow = build_flow_graph(flows, dev)
    idx = torch.tensor(edge_flow, device=dev)
    f, p = f[idx], p[idx]
    labels = torch.tensor(flows.Label.to_numpy().astype(np.int64)[edge_flow], device=dev)
    weights = torch.tensor(weights[edge_flow], device=dev)
    node_feats = torch.ones(g.num_nodes(), len(feature_cols), device=dev)

    model = GuardNIDSRR(len(feature_cols), routing=cfg.routing, backbone=cfg.backbone).to(dev)
    fit(model, lambda: model(g, node_feats, f, p), labels, weights, epochs)
    return dict(kind="guardnids_rr", net=model, scaler=scaler, feature_cols=feature_cols, cfg=cfg, refs=refs)


def _with_injected_copies(flows, raw, f, p, refs, owner, scaler, scale_cols, cfg):
    """GuardNIDS-RR+AT: one disjoint graph copy per injection set (C2xB, C2xM, add-node).
    The reference set is unchanged; an injected flow copies a training row and is
    retrieved leave-one-out like that row. The clean graph weighs as much as all copies."""
    generators = [lambda d: benign_from_controlled(d, 2), lambda d: benign_from_controlled(d, 5),
                  lambda d: malicious_from_controlled(d, 2), lambda d: malicious_from_controlled(d, 5),
                  lambda d: benign_to_new_nodes(d, 10, 2)]
    copies, fs, ps = [], [f], [p]
    weights = [np.full(len(flows), float(len(generators)), dtype=np.float32)]
    for make in generators:
        injected = make(raw.copy())[raw.columns]
        injected[scale_cols] = scaler.transform(injected[scale_cols])
        fi, pi = retrieve(refs, injected[refs.feature_cols].to_numpy(dtype=np.float32), cfg,
                          query_rows=injected.index.to_numpy(), owner=owner)
        copies.append(pd.concat([flows, injected], ignore_index=True))
        fs += [f, fi]
        ps += [p, pi]
        weights.append(np.ones(len(flows) + len(injected), dtype=np.float32))
    graph = pd.concat([flows] + separate_copies(copies, first=1), ignore_index=True)
    return graph, torch.cat(fs), torch.cat(ps), np.concatenate(weights)


# --------------------------------------------------------------------------- CLI
def run_dir(runs, train_path, seed, name):
    ds = dataset_of(pd.read_csv(train_path, nrows=5))
    return os.path.join(runs, ds, scenario_of(train_path), f"seed{seed}", name)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=BASELINES + ("advtrain", "guardnids_rr", "guardnids_rr_at"))
    ap.add_argument("--train", required=True, help="<scenario>_train.csv")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--name", help="run name (default: model name, with ablation suffixes)")
    rr = ap.add_argument_group("GuardNIDS-RR ablations")
    rr.add_argument("--key", choices=["invariant", "cosine"], default="invariant")
    rr.add_argument("--posterior", choices=["max", "inv", "top"], default="max")
    rr.add_argument("--no-anchors", action="store_true")
    rr.add_argument("--no-routing", action="store_true")
    rr.add_argument("--routing-control", choices=["const", "shuffle"], default="")
    rr.add_argument("--backbone", choices=["sage", "gat"], default="sage")
    args = ap.parse_args()

    set_seed(args.seed)
    if args.model in BASELINES:
        name, run = args.model, train_baseline(args.model, args.train, args.seed, args.epochs)
    elif args.model == "advtrain":
        name, run = "advtrain", train_advtrain(args.train, args.seed, args.epochs)
    else:
        cfg = RRConfig(key=args.key, posterior=args.posterior, n_anchors=0 if args.no_anchors else RRConfig.n_anchors,
                       routing=not args.no_routing, routing_control=args.routing_control, backbone=args.backbone)
        name = cfg.run_name(args.model)
        run = train_guardnids_rr(args.train, args.seed, cfg, args.epochs, adversarial=args.model == "guardnids_rr_at")

    out = run_dir(args.runs, args.train, args.seed, args.name or name)
    save_run(out, **run)
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
