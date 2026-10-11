"""Attack registry. Every attack takes (detector, raw test flows) and returns rows
{variant, level, recall, precision, f1, fpr}."""
from guardnids.attacks import adaptive, feature, structure
from guardnids.attacks.common import score
from guardnids.data import is_ctu


def clean(detector, flows):
    return [score(detector, flows.copy(), "clean", 0)]


ATTACKS = {
    "clean": clean,
    "padding_grid": lambda d, f: feature.padding_grid_attack(d, f, is_ctu(f)),
    "c2xb": structure.c2xb_attack,
    "c2xm": structure.c2xm_attack,
    "add_node": structure.add_node_attack,
    "padding": adaptive.padding_attack,
    "padding_anchor_physics": adaptive.padding_anchor_physics_attack,
    "over_budget": adaptive.over_budget_attack,
    "cde": adaptive.cde_attack,
    "cea": adaptive.cea_attack,
    "port_class_padding": adaptive.port_class_padding_attack,
    "header_edits": adaptive.header_edits_attack,
    "port_choice": adaptive.port_choice_attack,
    "routing_poison": adaptive.routing_poison_attack,
    "benign_padding": adaptive.benign_padding_attack,
}
