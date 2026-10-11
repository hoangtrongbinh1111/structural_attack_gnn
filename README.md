# GuardNIDS

GuardNIDS is a graph NIDS (flows are edges between `IP:port` endpoints) that stays robust when an
attacker pads its own flows or injects extra flows into the graph:

1. **Invariant-key retrieval** – every flow retrieves its nearest neighbours from a reference set of
   training flows, ranked on fields a client-side attacker cannot set (`guardnids/invariance.py`).
   The reference set also holds malicious flows padded under realistic traffic physics
   (`guardnids/physics.py`).
2. **Evidence routing** – each node aggregates two channels, weighted by the retrieved malicious
   posterior *p* and by *1 − p*, so injected benign-looking flows cannot dilute the malicious evidence
   of a node (`guardnids/rr.py`).

Baselines: E-GraphSAGE, GNNGuard, E-GAT and adversarial training (AdvTrain).

## Layout

```
guardnids/
  data.py, graph.py        dataset conventions, flow graph construction
  invariance.py            attacker-controllable fields, invariant key
  physics.py               physically consistent flow padding
  rr.py                    GuardNIDS (retrieval + routed message passing)
  baselines.py             E-GraphSAGE, GNNGuard, E-GAT
  detectors.py             trained-model loading and prediction
  attacks/                 feature, structural and adaptive attacks
  train.py                 training CLI
  evaluate.py              attack/evaluation CLI
  summary.py               result tables
preprocessing/             raw datasets -> train/test splits
scripts/                   end-to-end runs for one scenario
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

A CUDA GPU is recommended; everything also runs on CPU (slowly). Run all commands from the repository root.

## 1. Data

Datasets are not included. Download them into `datasets/`:

| Dataset | Source | Files |
|---|---|---|
| CTU-13 | https://www.stratosphereips.org/datasets-ctu13 | the `.binetflow` file of each capture, renamed after its botnet, e.g. `datasets/CTU_raw/1/neris.binetflow` |
| ToN-IoT | https://research.unsw.edu.au/projects/toniot-datasets | `datasets/ToN-IoT_raw/train_test_network.csv` |
| CIC-IDS2017 | https://www.unb.ca/cic/datasets/ids-2017.html | `MachineLearningCVE/*.csv` in `datasets/IDS2017_raw/` |
| NF-UNSW-NB15-v2 | https://staff.itee.uq.edu.au/marius/NIDS_datasets/ | `NF-UNSW-NB15-v2.csv` |

Split the raw files into one CSV per attack plus `benign.csv`:

```bash
python preprocessing/split_raw.py ctu13      --raw datasets/CTU_raw     --out datasets/CTU
python preprocessing/split_raw.py ton_iot    --raw datasets/ToN-IoT_raw --out datasets/ToN-IoT
python preprocessing/split_raw.py cicids2017 --raw datasets/IDS2017_raw --out datasets/IDS2017
```

Build the train/test splits (`data/<dataset>/<scenario>_{train,test}.csv`):

```bash
# CTU-13: one split per seed
for seed in 1 2 3 4; do
  for s in menti murlo neris nsisay rbot virut; do
    python preprocessing/ctu13.py --mal datasets/CTU/$s.csv --ben datasets/CTU/benign.csv \
        --out data/CTU/seed$seed --seed $seed
  done
done

# ToN-IoT
for s in backdoor ddos dos injection password ransomware scanning xss; do
  python preprocessing/ton_iot.py --mal datasets/ToN-IoT/$s.csv --ben datasets/ToN-IoT/benign.csv --out data/ToN_IoT
done

# CIC-IDS2017 (the three largest scenarios are subsampled to 300k/30k flows: *_sub)
for s in bot dos_goldeneye dos_slowhttptest dos_slowloris ftp_patator ssh_patator \
         web_attack_bruteforce web_attack_xss; do
  python preprocessing/cicids2017.py --mal datasets/IDS2017/$s.csv --ben datasets/IDS2017/benign.csv --out data/IDS2017
done
for s in ddos dos_hulk port_scan; do
  python preprocessing/cicids2017.py --mal datasets/IDS2017/$s.csv --ben datasets/IDS2017/benign.csv --out data/IDS2017 \
      --max-train 300000 --max-test 30000
done

# NF-UNSW-NB15-v2 (all attack classes at once)
python preprocessing/nf_unsw.py --csv datasets/NF-UNSW-NB15-v2/NF-UNSW-NB15-v2.csv --out data/NF_UNSW
```

## 2. Train

```bash
python -m guardnids.train --model guardnids_rr --train data/CTU/seed2/menti_train.csv --seed 2
```

`--model` is one of

| model | description |
|---|---|
| `egraphsage` | E-GraphSAGE |
| `gnnguard` | E-GraphSAGE + GNNGuard neighbour pruning |
| `egat` | edge-feature GAT |
| `advtrain` | E-GraphSAGE trained on the clean graph + adversarial copies |
| `guardnids_rr` | GuardNIDS-RR |
| `guardnids_rr_at` | GuardNIDS-RR also trained on injected graph copies |

The run is saved to `runs/<dataset>/<scenario>/seed<seed>/<model>/`.

## 3. Attack and evaluate

```bash
python -m guardnids.evaluate --run runs/CTU/menti/seed2/guardnids_rr \
    --test data/CTU/seed2/menti_test.csv --attacks all
```

Scores (malicious recall, precision, F1 and benign FPR per attack level) go to `<run>/scores/<attack>.csv`.
Add `--retrieval-only` on a GuardNIDS-RR run to score the graph-free control (malicious iff *p* ≥ 0.5).

| attack | what the attacker does |
|---|---|
| `clean` | nothing |
| `padding_grid` | adds fixed offsets to duration/bytes/packets (unconstrained, steps 1–9) |
| `padding` | pads under realistic physics (headers, ACKs, exporter limits) |
| `padding_anchor_physics` | same, with the physics GuardNIDS-RR's anchors use |
| `over_budget` | step-9 physical padding scaled ×1 … ×20 |
| `cde` | grows each flow toward the benign centroid |
| `cea` | `cde` + benign flows injected from the controlled hosts |
| `c2xb` / `c2xm` | controlled hosts send extra benign / malicious flows |
| `add_node` | controlled hosts send benign flows to new endpoints |
| `header_edits` | rewrites client-set fields (source-port class, flag bits, TTL/window) |
| `port_class_padding` | source-port class + padding |
| `port_choice` | new source port per flow, i.e. a new client node |
| `routing_poison` | benign flows that copy the discrete fields of the host's malicious flows |
| `benign_padding` | control: benign flows padded, malicious untouched |

## 4. Full runs and tables

```bash
# all models x all attacks for one scenario
scripts/run_scenario.sh data/CTU/seed2 menti 2
scripts/run_scenario.sh data/ToN_IoT backdoor 1

# GuardNIDS ablations and the E-GAT backbone
scripts/run_ablations.sh data/CTU/seed1 menti 1

# dataset x attack table (mean over seeds, then over scenarios)
python -m guardnids.summary --metric f1
```

The paper reports seeds 2–4 on CTU-13 (seed 1 was used during development) and seeds 1–3 on the
other datasets: `python -m guardnids.summary --seeds 2 3 4`, etc.
