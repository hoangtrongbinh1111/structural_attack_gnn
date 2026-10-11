"""Feature attacks: the attacker pads its own malicious flows.

`padding_grid` is the unconstrained grid of Venturi et al.: the same offset is
added to Dur, SrcBytes, DstBytes and TotPkts of every malicious flow (step 1-9)
and the derived statistics are recomputed. The physically consistent versions
are in guardnids.attacks.adaptive.
"""
from guardnids.attacks.common import as_float, malicious_mask, replace_inf_with_max, score

STEPS = {
    1: {"Dur": 1, "SrcBytes": 1, "DstBytes": 1, "TotPkts": 1},
    2: {"Dur": 2, "SrcBytes": 2, "DstBytes": 2, "TotPkts": 2},
    3: {"Dur": 5, "SrcBytes": 8, "DstBytes": 8, "TotPkts": 5},
    4: {"Dur": 10, "SrcBytes": 16, "DstBytes": 16, "TotPkts": 10},
    5: {"Dur": 15, "SrcBytes": 64, "DstBytes": 64, "TotPkts": 15},
    6: {"Dur": 30, "SrcBytes": 128, "DstBytes": 128, "TotPkts": 20},
    7: {"Dur": 45, "SrcBytes": 256, "DstBytes": 256, "TotPkts": 30},
    8: {"Dur": 60, "SrcBytes": 512, "DstBytes": 512, "TotPkts": 50},
    9: {"Dur": 120, "SrcBytes": 1024, "DstBytes": 1024, "TotPkts": 100},
}


def update_derived(flows, ctu):
    """Recompute totals and, on CTU-13, the engineered ratios."""
    flows.loc[:, "TotBytes"] = flows["SrcBytes"] + flows["DstBytes"]
    if ctu:
        flows.loc[:, "BytesPerPkt"] = flows["TotBytes"] / flows["TotPkts"]
        flows.loc[:, "PktsPerSec"] = flows["TotPkts"] / flows["Dur"]
        flows.loc[:, "RatioOutIn"] = flows["DstBytes"] / flows["SrcBytes"]
        for col in ("PktsPerSec", "RatioOutIn", "BytesPerPkt"):
            flows = replace_inf_with_max(flows, col)
    return flows


def pad_on_grid(flows, step, ctu):
    """Add the step's offsets to the given flows (all rows) and update derived fields."""
    flows = flows.copy()
    for col, offset in STEPS[step].items():
        flows.loc[:, col] = flows[col] + offset
    return update_derived(flows, ctu)


def padding_grid_attack(detector, flows, ctu):
    rows = []
    for step in STEPS:
        attacked = as_float(flows)
        mal = malicious_mask(attacked)
        attacked.loc[mal] = pad_on_grid(attacked.loc[mal], step, ctu)
        rows.append(score(detector, attacked, "grid", step))
    return rows
