"""Structural attacks: the attacker adds flows from the hosts it controls.

Controlled hosts (C) are the sources of malicious flows. Injected flows are
copies of real flows of the split, so they are valid records.
  C2xB     : each C sends `amount` extra benign flows (dilutes C's neighborhood)
  C2xM     : each C sends `amount` extra malicious flows to benign destinations
  add-node : C sends `amount` benign flows to each of `n_nodes` new endpoints
"""
import random

import pandas as pd

from guardnids.attacks.common import score

AMOUNTS = [1, 2, 5, 10, 20]
NEW_NODE_COUNTS = [10, 100]


def benign_from_controlled(flows, amount):
    controlled = list(flows[flows.Label == 1]["Src IP"].unique())
    injected = flows[flows.Label == 0].sample(len(controlled) * amount, replace=True)
    injected["Src IP"] = controlled * amount
    return injected.copy()


def malicious_from_controlled(flows, amount):
    malicious = flows[flows.Label == 1]
    injected = malicious.sample(malicious["Src IP"].nunique() * amount, replace=True)
    injected["Dst IP"] = flows[flows.Label == 0].sample(len(injected), replace=True)["Dst IP"].to_list()
    return injected.copy()


def random_endpoint():
    ip = ".".join(str(random.randint(0, 255)) for _ in range(4))
    return f"{ip}:{random.randint(49152, 65535)}"


def benign_to_new_nodes(flows, n_nodes, amount, max_tries=100):
    existing = set(flows["Src IP"]) | set(flows["Dst IP"])
    new_nodes, tries = [], 0
    while len(new_nodes) < n_nodes:
        node = random_endpoint()
        tries += 1
        if node not in existing:
            new_nodes.append(node)
            tries = 0
        if tries > max_tries:
            raise RuntimeError("could not generate unused endpoints")

    senders = flows[flows.Label == 1].sample(n_nodes * amount, replace=True)
    injected = flows[flows.Label == 0].sample(n_nodes * amount, replace=True)
    injected["Src IP"] = senders["Src IP"].values
    injected["Dst IP"] = new_nodes * amount
    return injected.copy()


def _inject(flows, extra):
    return pd.concat([flows, extra], ignore_index=True)


def c2xb_attack(detector, flows):
    return [score(detector, _inject(flows, benign_from_controlled(flows, a)), "C2xB", a) for a in AMOUNTS]


def c2xm_attack(detector, flows):
    return [score(detector, _inject(flows, malicious_from_controlled(flows, a)), "C2xM", a) for a in AMOUNTS]


def add_node_attack(detector, flows):
    return [score(detector, _inject(flows, benign_to_new_nodes(flows, n, a)), f"{n}_nodes", a)
            for n in NEW_NODE_COUNTS for a in AMOUNTS]
