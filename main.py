"""
Bus bunching on a looped city route: tabular Q-learning.
Run: python main.py --episodes 12000 --eval-episodes 160
Dependencies: numpy, matplotlib.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

N_BUSES = 6
ROUTE_LENGTH = 60.0
N_STOPS = 12
TARGET_GAP = ROUTE_LENGTH / N_BUSES
ACTIONS = (0, 2, 4)
HORIZON = N_STOPS * 4


class BusLoopEnv:
    """Synthetic circular route with stochastic travel increments and bus loads."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.positions = np.sort(self.rng.uniform(0, ROUTE_LENGTH, N_BUSES))
        self.loads = self.rng.integers(10, 61, size=N_BUSES).astype(float)
        self.steps = 0

    def gaps(self, i):
        # Bus identities are fixed and can cross one another in the simulated
        # dynamics, so find the nearest physical neighbour using circular
        # distances instead of assuming array order equals route order.
        here = float(self.positions[i])
        ahead = min(
            (float(self.positions[j]) - here) % ROUTE_LENGTH
            for j in range(N_BUSES)
            if j != i
        )
        behind = min(
            (here - float(self.positions[j])) % ROUTE_LENGTH
            for j in range(N_BUSES)
            if j != i
        )
        return float(ahead), float(behind)

    def state(self, i):
        ahead, behind = self.gaps(i)
        return (
            int(np.digitize(ahead, [5, 8, 11, 14, 18])),
            int(np.digitize(behind, [5, 8, 11, 14, 18])),
            int(np.digitize(self.loads[i], [20, 40, 60])),
            0 if abs(ahead - behind) < 3 else (1 if ahead > behind else 2),
            self.steps % 3,
        )

    def step(self, i, hold):
        ahead, behind = self.gaps(i)
        before = abs(ahead - TARGET_GAP) + abs(behind - TARGET_GAP)
        hold_cost = 0.6 * hold
        self.positions = (
            self.positions + self.rng.normal(5.0, 0.7, N_BUSES)
        ) % ROUTE_LENGTH
        self.positions[i] = (self.positions[i] - hold) % ROUTE_LENGTH
        self.loads = np.clip(
            self.loads + self.rng.normal(2.0, 2.5, N_BUSES) - 1.0, 0, 100
        )
        a2, b2 = self.gaps(i)
        after = abs(a2 - TARGET_GAP) + abs(b2 - TARGET_GAP)
        reward = -(
            hold_cost + 0.7 * after + 0.25 * max(0.0, after - before)
        )
        self.steps += 1
        return self.state(i), float(reward), {"hold": hold, "imbalance": after}


def train(episodes=12000, seed=1041, alpha=0.15, gamma=0.92):
    """Train tabular Q-learning and return a mapping from state tuples to Q-values."""
    rng = np.random.default_rng(seed)
    q = {}
    for ep in range(episodes):
        env = BusLoopEnv(seed + ep)
        epsilon = max(0.04, 0.35 * (0.996 ** ep))
        for t in range(HORIZON):
            i = t % N_BUSES
            state = env.state(i)
            q.setdefault(state, np.zeros(len(ACTIONS)))
            action_index = (
                int(rng.integers(len(ACTIONS)))
                if rng.random() < epsilon
                else int(np.argmax(q[state]))
            )
            next_state, reward, _ = env.step(i, ACTIONS[action_index])
            q.setdefault(next_state, np.zeros(len(ACTIONS)))
            q[state][action_index] += alpha * (
                reward + gamma * np.max(q[next_state]) - q[state][action_index]
            )
    return q


def evaluate(q, episodes=160, seed=90000):
    """Evaluate no-holding, rule-based, and greedy Q-learning policies."""
    results = {}
    for name in ("No holding", "Rule-based", "Q-Learning"):
        episode_rows = []
        for ep in range(episodes):
            env = BusLoopEnv(seed + ep)
            holds, imbalances, rewards = [], [], []
            for t in range(HORIZON):
                i = t % N_BUSES
                state = env.state(i)
                if name == "No holding":
                    hold = 0
                elif name == "Rule-based":
                    ahead, _ = env.gaps(i)
                    hold = 4 if ahead < 7 else (2 if ahead < 10 else 0)
                else:
                    hold = ACTIONS[
                        int(np.argmax(q.get(state, np.zeros(len(ACTIONS)))))
                    ]
                _, reward, info = env.step(i, hold)
                holds.append(hold)
                imbalances.append(info["imbalance"])
                rewards.append(reward)
            episode_rows.append(
                [np.mean(holds), np.mean(imbalances), np.mean(rewards)]
            )
        array = np.asarray(episode_rows)
        results[name] = {
            "mean": array.mean(axis=0).tolist(),
            "std": array.std(axis=0).tolist(),
        }
    return results


def save_outputs(q, results):
    """Write the policy table, metrics, and comparison plots to the working folder."""
    Path("figures").mkdir(exist_ok=True)
    with open("q_table.json", "w", encoding="utf-8") as file:
        json.dump(
            {",".join(map(str, state)): values.tolist() for state, values in q.items()},
            file,
            indent=2,
        )
    with open("evaluation_results.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "policy", "mean_hold_min", "std_hold_min", "mean_imbalance",
            "std_imbalance", "mean_reward", "std_reward"
        ])
        for name, row in results.items():
            means, stds = row["mean"], row["std"]
            writer.writerow([name, means[0], stds[0], means[1], stds[1], means[2], stds[2]])

    policies = list(results)
    means = np.array([results[policy]["mean"] for policy in policies])
    stds = np.array([results[policy]["std"] for policy in policies])
    charts = [
        (1, "Mean local spacing imbalance (lower is better)", "headway_imbalance.png"),
        (2, "Mean reward per decision (higher is better)", "mean_reward.png"),
        (0, "Mean hold duration (minutes)", "mean_hold.png"),
    ]
    for column, ylabel, filename in charts:
        plt.figure(figsize=(6.4, 3.0))
        plt.bar(policies, means[:, column], yerr=stds[:, column], capsize=4)
        plt.ylabel(ylabel)
        plt.title("Policy comparison across evaluation episodes")
        plt.tight_layout()
        plt.savefig(Path("figures") / filename, dpi=170)
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=12000)
    parser.add_argument("--eval-episodes", type=int, default=160)
    arguments = parser.parse_args()
    q_table = train(arguments.episodes)
    evaluation = evaluate(q_table, arguments.eval_episodes)
    save_outputs(q_table, evaluation)
    print("Policy comparison (mean ± standard deviation across episodes)")
    for name, row in evaluation.items():
        means, stds = row["mean"], row["std"]
        print(
            f"{name:12s} | hold={means[0]:.2f}±{stds[0]:.2f} min "
            f"| imbalance={means[1]:.2f}±{stds[1]:.2f} "
            f"| reward={means[2]:.2f}±{stds[2]:.2f}"
        )
