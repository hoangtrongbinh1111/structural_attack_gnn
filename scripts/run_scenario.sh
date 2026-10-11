#!/bin/bash
# Train every detector on one scenario and run every attack against it.
#
#   scripts/run_scenario.sh data/CTU/seed2 menti 2
#
# Models can be restricted with MODELS="guardnids_rr egraphsage" and attacks
# with ATTACKS="clean c2xb"; runs are written to $RUNS (default: runs).
set -euo pipefail

SPLIT_DIR=$1
SCENARIO=$2
SEED=$3
RUNS=${RUNS:-runs}
MODELS=${MODELS:-"egraphsage gnnguard advtrain guardnids_rr guardnids_rr_at"}
ATTACKS=${ATTACKS:-all}

TRAIN=$SPLIT_DIR/${SCENARIO}_train.csv
TEST=$SPLIT_DIR/${SCENARIO}_test.csv
DATASET=$(python -c "import pandas as pd, guardnids.data as d; print(d.dataset_of(pd.read_csv('$TRAIN', nrows=5)))")

for model in $MODELS; do
    run=$RUNS/$DATASET/$SCENARIO/seed$SEED/$model
    if [ ! -f "$run/model.pt" ]; then
        echo "[$SCENARIO seed $SEED] training $model"
        python -m guardnids.train --model "$model" --train "$TRAIN" --seed "$SEED" --runs "$RUNS"
    fi
    echo "[$SCENARIO seed $SEED] attacking $model"
    python -m guardnids.evaluate --run "$run" --test "$TEST" --attacks $ATTACKS
done

if [[ " $MODELS " == *" guardnids_rr "* ]]; then
    echo "[$SCENARIO seed $SEED] attacking the retrieval-only control"
    python -m guardnids.evaluate --run "$RUNS/$DATASET/$SCENARIO/seed$SEED/guardnids_rr" --test "$TEST" \
        --attacks $ATTACKS --retrieval-only
fi
