"""Baseline edge classifiers: E-GraphSAGE, GNNGuard and E-GAT.

All three share the same shape: two message-passing layers over the flow graph
(edge features enter the messages), then an edge is classified from the
embeddings of its two endpoints. Node inputs are constant, as in E-GraphSAGE.
"""
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.functional import edge_softmax

HIDDEN = 128
DROPOUT = 0.2


class SAGELayer(nn.Module):
    """E-GraphSAGE layer: mean of W_msg[h_src, e] over incoming edges."""

    def __init__(self, in_dim, edge_dim, out_dim):
        super().__init__()
        self.W_msg = nn.Linear(in_dim + edge_dim, out_dim)
        self.W_apply = nn.Linear(in_dim + out_dim, out_dim)

    def forward(self, g, h, e, prev_weights=None):
        with g.local_scope():
            g.ndata["h"] = h
            g.edata["e"] = e
            g.apply_edges(lambda edges: {"m": self.W_msg(torch.cat([edges.src["h"], edges.data["e"]], 1))})
            g.update_all(fn.copy_e("m", "m"), fn.mean("m", "agg"))
            return F.relu(self.W_apply(torch.cat([h, g.ndata["agg"]], 1))), None


class GNNGuardLayer(nn.Module):
    """GNNGuard (Zhang & Zitnik, 2020) on top of the E-GraphSAGE layer.

    Each edge is weighted by the cosine similarity of its endpoint embeddings,
    edges below `prune_threshold` are dropped, weights are normalized per node and
    blended with the previous layer's weights through a learned gate.
    """

    def __init__(self, in_dim, edge_dim, out_dim, prune_threshold=0.1):
        super().__init__()
        self.W_msg = nn.Linear(in_dim + edge_dim, out_dim)
        self.W_apply = nn.Linear(in_dim + out_dim, out_dim)
        self.prune_threshold = prune_threshold
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, g, h, e, prev_weights=None):
        src, dst = g.edges()
        sim = F.cosine_similarity(h[src], h[dst], dim=-1).clamp(min=0.0)
        sim = torch.where(sim > self.prune_threshold, sim, torch.zeros_like(sim))
        if prev_weights is None:
            weights = sim
        else:
            gate = torch.sigmoid(self.gate)
            weights = gate * sim + (1.0 - gate) * prev_weights

        with g.local_scope():
            g.ndata["h"] = h
            g.edata["e"] = e
            g.edata["w"] = weights
            g.update_all(fn.copy_e("w", "w"), fn.sum("w", "w_sum"))
            g.apply_edges(lambda edges: {"w_norm": edges.data["w"] / (edges.dst["w_sum"] + 1e-8)})
            g.apply_edges(lambda edges: {
                "m": self.W_msg(torch.cat([edges.src["h"], edges.data["e"]], 1)) * edges.data["w_norm"].unsqueeze(1)})
            g.update_all(fn.copy_e("m", "m"), fn.sum("m", "agg"))
            out = F.relu(self.W_apply(torch.cat([h, g.ndata["agg"]], 1)))
        return out, weights.detach()


class EGATLayer(nn.Module):
    """Edge-feature GAT layer: softmax attention over incoming messages."""

    def __init__(self, in_dim, edge_dim, out_dim):
        super().__init__()
        self.W_msg = nn.Linear(in_dim + edge_dim, out_dim)
        self.W_apply = nn.Linear(in_dim + out_dim, out_dim)
        self.attn_fc = nn.Linear(out_dim, 1)

    def forward(self, g, h, e, prev_weights=None):
        with g.local_scope():
            g.ndata["h"] = h
            g.edata["e"] = e
            g.apply_edges(lambda edges: {"m": self.W_msg(torch.cat([edges.src["h"], edges.data["e"]], 1))})
            score = F.leaky_relu(self.attn_fc(g.edata["m"]), 0.2)
            g.edata["m"] = g.edata["m"] * edge_softmax(g, score)
            g.update_all(fn.copy_e("m", "m"), fn.sum("m", "agg"))
            return F.relu(self.W_apply(torch.cat([h, g.ndata["agg"]], 1))), None


class EdgeClassifier(nn.Module):
    """Two message-passing layers + a linear classifier on [h_src, h_dst]."""

    def __init__(self, layer_cls, n_features):
        super().__init__()
        self.layers = nn.ModuleList([layer_cls(n_features, n_features, HIDDEN),
                                     layer_cls(HIDDEN, n_features, HIDDEN)])
        self.dropout = nn.Dropout(DROPOUT)
        self.classifier = nn.Linear(2 * HIDDEN, 2)

    def forward(self, g, node_feats, edge_feats):
        h, weights = node_feats, None
        for i, layer in enumerate(self.layers):
            if i > 0:
                h = self.dropout(h)
            h, weights = layer(g, h, edge_feats, weights)
        src, dst = g.edges()
        return self.classifier(torch.cat([h[src], h[dst]], 1))


LAYERS = {"egraphsage": SAGELayer, "gnnguard": GNNGuardLayer, "egat": EGATLayer}


def build_baseline(arch, n_features):
    return EdgeClassifier(LAYERS[arch], n_features)
