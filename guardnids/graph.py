"""Flow graph: nodes are IP:port endpoints, every flow is an edge in both directions."""
import dgl
import numpy as np
import pandas as pd
import torch


def build_flow_graph(flows, device, self_loops_once=False):
    """Return (graph, edge_flow): edge_flow[i] is the row of `flows` behind edge i.

    Edges are all flows src->dst followed by all flows dst->src. With
    `self_loops_once`, a flow whose two endpoints coincide gets a single edge
    (as a networkx MultiGraph.to_directed() does; the E-GraphSAGE family was
    built that way). Per-flow arrays are mapped to edges with `x[edge_flow]`.
    """
    endpoints = pd.Index(pd.unique(pd.concat([flows["Src IP"], flows["Dst IP"]], ignore_index=True)))
    node_id = {name: i for i, name in enumerate(endpoints)}
    src = flows["Src IP"].map(node_id).to_numpy()
    dst = flows["Dst IP"].map(node_id).to_numpy()
    rows = np.arange(len(flows))
    back = src != dst if self_loops_once else np.ones(len(flows), dtype=bool)

    edge_flow = np.concatenate([rows, rows[back]])
    g = dgl.graph((torch.tensor(np.concatenate([src, dst[back]]), dtype=torch.int64),
                   torch.tensor(np.concatenate([dst, src[back]]), dtype=torch.int64)),
                  num_nodes=len(endpoints))
    return g.to(device), edge_flow
