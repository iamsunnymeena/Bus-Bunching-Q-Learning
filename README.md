# Bus Bunching on a Looped City Route — Q-Learning

**Team:** Sunny Meena (`b23cm1041`) and Faiz Imdad (`b23cm1016`)  
**Repository:** https://github.com/iamsunnymeena/Bus-Bunching-Q-Learning

## Problem
Buses circulating on a loop can bunch together because of stochastic travel times. At each decision, an agent chooses to hold a bus for 0, 2, or 4 minutes. This project compares tabular Q-learning against no holding and a rule-based controller in a synthetic environment.

## Reinforcement-learning formulation
- **Environment:** synthetic circular route of 60 route-minutes, six buses, 12 stop/decision phases, and 48 decisions per episode. No external dataset is needed.
- **State:** five discrete features: forward-gap bin (6 possibilities), backward-gap bin (6), bus-load bin (4), gap-imbalance direction (3), and stop phase (3). The nominal upper bound is `6 × 6 × 4 × 3 × 3 = 1,296` states. Gap thresholds: `[5, 8, 11, 14, 18]`; load thresholds: `[20, 40, 60]`.
- **Action space:** `A = {0, 2, 4}` minutes of holding.
- **Reward:** `r = -(0.6h + 0.7I_after + 0.25 max(0, I_after - I_before))`, where `h` is hold duration and `I = |g_forward - 10| + |g_backward - 10|`. The target headway is 10 route-minutes (60 route-minutes divided by six buses).
- **Dynamics:** buses advance with stochastic increments sampled from `Normal(5, 0.7)`, the chosen bus receives the hold delay, and loads evolve stochastically within `[0, 100]`.
- **Learning:** tabular Q-learning, learning rate `α=0.15`, discount factor `γ=0.92`, and epsilon-greedy exploration from 0.35 with a 0.04 floor.

## Setup and run

Python 3.10+ is recommended.

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py --episodes 12000 --eval-episodes 160
```

This trains a new Q-table, evaluates the three policies, and saves `q_table.json`, `evaluation_results.csv`, and charts in `figures/`. A saved compressed Q-table is supplied as `q_table.json.gz`; evaluate it without retraining with:

```bash
python main.py --load-q-table q_table.json.gz --eval-episodes 160
```

The evaluation command refreshes the CSV and charts. Seeds and hyperparameters are defined in `main.py`.

## Baselines and evaluation
1. **No holding:** always holds for 0 minutes.
2. **Rule-based:** holds 4 minutes when the forward gap is below 7, 2 minutes when below 10, otherwise 0.
3. **Q-learning:** chooses the greedy action from the learned Q-table.

Metrics are mean local spacing imbalance (lower is better), mean reward per decision (higher/less negative is better), and mean hold duration. Values are means over 160 evaluation episodes; `evaluation_results.csv` also contains standard deviations.

| Policy | Mean hold (min) | Mean imbalance | Mean reward |
|---|---:|---:|---:|
| No holding | 0.00 | 13.42 | -9.57 |
| Rule-based | 1.81 | 6.91 | -6.09 |
| Q-learning | 1.58 | 10.22 | -8.27 |

In this reference run, Q-learning improves over no holding, but does not outperform the simple rule-based heuristic. This is an important limitation of the current tabular formulation. These are simulation results, not field evidence; they depend on the dynamics, random seeds, and reward weights.

## Files
- `main.py` — environment, Q-learning, evaluation and plotting code
- `q_table.json.gz` — saved trained state-action values in compressed JSON format
- `q_table.json` — uncompressed Q-table generated when training or evaluating
- `evaluation_results.csv` — policy comparison results
- `figures/` — charts generated from the evaluation
- `requirements.txt` — Python dependencies
- `report.pdf` — supplied separately as the descriptive submission report

## Limitations
This proof of concept does not model passenger-level waiting-time accumulation, calibrated dwell times, real traffic, timetables, or measured route data. Future work should add passenger waiting-time measures, evaluate multiple training seeds and reward settings, and calibrate the simulator against observed arrivals.

## Academic integrity
Both team members should inspect and understand the code and make their own meaningful changes. The repository's commit history should reflect real development; no history is fabricated.
