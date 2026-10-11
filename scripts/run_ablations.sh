#!/bin/bash
# GuardNIDS-RR ablations and the E-GAT backbone on one scenario.
#
#   scripts/run_ablations.sh data/CTU/seed1 menti 1
set -euo pipefail

SPLIT_DIR=$1
SCENARIO=$2
SEED=$3
RUNS=${RUNS:-runs}
ATTACKS=${ATTACKS:-"clean padding c2xb c2xm add_node over_budget cde cea routing_poison header_edits port_choice"}

TRAIN=$SPLIT_DIR/${SCENARIO}_train.csv
TEST=$SPLIT_DIR/${SCENARIO}_test.csv
DATASET=$(python -c "import pandas as pd, guardnids.data as d; print(d.dataset_of(pd.read_csv('$TRAIN', nrows=5)))")

train_and_attack() {
    local name=$1; shift
    local run=$RUNS/$DATASET/$SCENARIO/seed$SEED/$name
    [ -f "$run/model.pt" ] || python -m guardnids.train --train "$TRAIN" --seed "$SEED" --runs "$RUNS" --name "$name" "$@"
    python -m guardnids.evaluate --run "$run" --test "$TEST" --attacks $ATTACKS
}

train_and_attack rr_cosine_key      --model guardnids_rr --key cosine
train_and_attack rr_no_routing      --model guardnids_rr --no-routing
train_and_attack rr_routing_const   --model guardnids_rr --routing-control const
train_and_attack rr_routing_shuffle --model guardnids_rr --routing-control shuffle
train_and_attack rr_p_inv_only      --model guardnids_rr --posterior inv
train_and_attack rr_p_top_only      --model guardnids_rr --posterior top
train_and_attack rr_no_anchors      --model guardnids_rr --no-anchors
train_and_attack rr_gat             --model guardnids_rr --backbone gat
train_and_attack egat               --model egat
