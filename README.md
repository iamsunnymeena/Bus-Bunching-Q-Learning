# Bus Bunching on a Looped City Route — Q-Learning

**Team:** Sunny Meena (`b23cm1041`) and Faiz Imdad (`b23cm1016`)  
**Repository:** https://github.com/iamsunnymeena/Bus-Bunching-Q-Learning

## Problem
Buses circulating on a loop may bunch together because of stochastic travel times. At each decision, an agent chooses to hold a bus for 0, 2, or 4 minutes. This project compares tabular Q-learning against no holding and a simple rule-based controller in a synthetic environment.

## Reinforcement-learning formulation
- **Environment:** synthetic circular route of 60 route-minutes, six buses, 12 stop/decision phases, and 48 decisions per episode. No external dataset is needed.
- **State:** five discrete features: forward-gap bin (6 possibilities), backward-gap bin (6), bus-load bin (4), gap-imbalance direction (3), and stop phase (3). The nominal upper bound is `6 × 6 × 4 × 3 × 3 = 1,296` states. Gap thresholds: `[5, 8, 11, 14, 18]`; load thresholds: `[20, 40, 60]`.
- **Action space:** `A = {0, 2, 4}` minutes of holding.
- **Reward:** `r = -(0.6h + 0.7I_after + 0.25 max(0, I_after - I_before))`, where `h` is hold duration and `I = |g_forward - 10| + |g_backward - 10|`. A target headway of 10 route-minutes follows from 60 route-minutes divided by six buses.
- **Dynamics:** buses advance with stochastic increments sampled from `Normal(5, 0.7)`, the selected bus receives the hold delay, and loads evolve stochastically within `[0, 100]`.
- **Learning:** tabular Q-learning, learning rate `α=0.15`, discount factor `γ=0.92`, epsilon-greedy exploration from 0.35 with a 0.04 floor.

## Setup and run
Python 3.10+ is recommended.

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py --episodes 12000 --eval-episodes 160
```

The script saves `q_table.json`, `evaluation_results.csv`, and plots in `figures/`. Seeds and hyperparameters are defined in `main.py`. The saved outputs are a reference run; rerunning training regenerates the Q-table, metrics, and figures.

## Baselines and metrics
1. **No holding:** always holds for 0 minutes.
2. **Rule-based:** holds for 4 minutes when forward gap is below 7, for 2 minutes when below 10, otherwise 0.
3. **Q-learning:** chooses the greedy action from the learned Q-table.

Metrics are mean local spacing imbalance (lower is better), mean reward per decision (higher/less negative is better), and mean hold duration. The results below are mean values over 160 evaluation episodes.

| Policy | Mean hold (min) | Mean imbalance | Mean reward |
|---|---:|---:|---:|
| No holding | 0.00 | 24.29 | -17.44 |
| Rule-based | 1.39 | 12.50 | -9.77 |
| Q-learning | 0.87 | 11.13 | -8.49 |

These are simulation results, not field evidence. Values depend on the dynamics, random seeds, and chosen reward weights.

## Files
- `main.py` — environment, Q-learning, evaluation and plotting
- `q_table.json` — saved state-action values
- `evaluation_results.csv` — saved policy comparison
- `figures/` — charts generated from evaluation metrics
- `report.pdf` — two-page descriptive report plus separate references page (PDF will be included in the submission bundle)
- `report.tex` — LaTeX source for the report
- `requirements.txt` — Python dependencies

## Limitations
This proof of concept does not model passenger-level waiting time, calibrated dwell times, real traffic, timetables, or measured route data. Future work should introduce passenger waiting-time accumulation, evaluate multiple training seeds and reward settings, and calibrate the simulator against observed arrivals.

## Academic integrity
Both team members should inspect and understand the code and make and document their own meaningful changes. Commit history should reflect real development; no history is fabricated.
