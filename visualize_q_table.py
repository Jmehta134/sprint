"""
Q-Table Visualizer
==================
Produces two separate figures from a .pkl Q-table:
  1. Q-Table Heatmap
  2. Greedy Policy

Supported formats:
  - dict  {state: [q_values]}  or  {state: {action: q_value}}
  - numpy 2-D array  (num_states, num_actions)
  - pandas DataFrame

Usage:
for reference 
(C:\Users\USER\miniconda3\shell\condabin\conda-hook.ps1) ; (conda activate elsi_sprint)
  python visualize_qtable.py qtable.pkl
  python visualize_qtable.py qtable.pkl --cmap plasma
  python visualize_qtable.py qtable.pkl --save-heatmap heatmap.png --save-policy policy.png
"""

import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


# ── helpers ───────────────────────────────────────────────────────────────────

def load_pkl(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def to_matrix(qtable) -> np.ndarray:
    """Convert any supported Q-table format to a 2-D numpy array."""
    if isinstance(qtable, np.ndarray):
        if qtable.ndim == 2:
            return qtable.astype(float)
        raise ValueError(f"numpy array must be 2-D, got shape {qtable.shape}")

    try:
        import pandas as pd
        if isinstance(qtable, pd.DataFrame):
            return qtable.values.astype(float)
    except ImportError:
        pass

    if isinstance(qtable, dict):
        states = sorted(qtable.keys(), key=lambda x: (str(type(x)), str(x)))
        sample = qtable[states[0]]
        if isinstance(sample, dict):
            actions = sorted(sample.keys(), key=lambda x: (str(type(x)), str(x)))
            return np.array(
                [[qtable[s].get(a, 0.0) for a in actions] for s in states],
                dtype=float,
            )
        return np.array([qtable[s] for s in states], dtype=float)

    raise TypeError(f"Unsupported Q-table type: {type(qtable)}")


def label_axes(qtable):
    """Try to extract meaningful state / action labels."""
    state_labels = action_labels = None

    try:
        import pandas as pd
        if isinstance(qtable, pd.DataFrame):
            return [str(i) for i in qtable.index], [str(c) for c in qtable.columns]
    except ImportError:
        pass

    if isinstance(qtable, dict):
        states = sorted(qtable.keys(), key=lambda x: (str(type(x)), str(x)))
        state_labels = [str(s) for s in states]
        sample = qtable[states[0]]
        if isinstance(sample, dict):
            actions = sorted(sample.keys(), key=lambda x: (str(type(x)), str(x)))
            action_labels = [str(a) for a in actions]

    return state_labels, action_labels


def make_ticks(labels, n, max_ticks=40):
    if labels and n <= max_ticks:
        return np.arange(n), labels
    step = max(1, n // max_ticks)
    idx = np.arange(0, n, step)
    lbl = [str(i) for i in idx] if not labels else [labels[i] for i in idx]
    return idx, lbl


# ── figures ───────────────────────────────────────────────────────────────────

def fig_heatmap(matrix, state_labels, action_labels, cmap):
    n_states, n_actions = matrix.shape
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#F7F9FC")
    ax.set_facecolor("#FFFFFF")

    im = ax.imshow(matrix, aspect="auto", cmap=cmap)

    s_idx, s_lbl = make_ticks(state_labels, n_states)
    a_idx, a_lbl = make_ticks(action_labels, n_actions)

    ax.set_yticks(s_idx)
    ax.set_yticklabels(s_lbl, fontsize=max(5, min(9, 200 // n_states)))
    ax.set_xticks(a_idx)
    ax.set_xticklabels(a_lbl, fontsize=9, rotation=45, ha="right")

    ax.set_xlabel("Action", fontsize=12)
    ax.set_ylabel("State", fontsize=12)
    ax.set_title(
        f"Q-Table Heatmap  ({n_states} states × {n_actions} actions)",
        fontsize=14, fontweight="bold",
    )
    plt.colorbar(im, ax=ax, label="Q-value", fraction=0.03, pad=0.02)
    fig.tight_layout()
    return fig


def fig_policy(matrix, state_labels, cmap):
    n_states, n_actions = matrix.shape
    greedy = np.argmax(matrix, axis=1)

    fig, ax = plt.subplots(figsize=(8, max(5, n_states // 8)))
    fig.patch.set_facecolor("#F7F9FC")
    ax.set_facecolor("#FFFFFF")

    colors = plt.get_cmap(cmap)(greedy / max(greedy.max(), 1))
    bars = ax.barh(np.arange(n_states), greedy + 1, color=colors, edgecolor="none")

    label_fs = max(5, min(8, 180 // n_states))
    for bar, val in zip(bars, greedy):
        ax.text(
            val + 1 + 0.05,
            bar.get_y() + bar.get_height() / 2,
            f"A{val}",
            va="center", fontsize=label_fs,
        )

    s_idx, s_lbl = make_ticks(state_labels, n_states)
    ax.set_yticks(s_idx)
    ax.set_yticklabels(s_lbl, fontsize=max(5, min(9, 200 // n_states)))
    ax.invert_yaxis()

    ax.set_xlabel("Greedy Action Index (1-based)", fontsize=12)
    ax.set_ylabel("State", fontsize=12)
    ax.set_title("Greedy Policy", fontsize=14, fontweight="bold")

    sm = ScalarMappable(cmap=cmap, norm=Normalize(0, n_actions - 1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Action index", fraction=0.03, pad=0.02)
    fig.tight_layout()
    return fig


# ── main ──────────────────────────────────────────────────────────────────────

def visualize(path, cmap="RdYlGn", save_heatmap=None, save_policy=None, dpi=150):
    print(f"Loading: {path}")
    raw = load_pkl(path)

    if isinstance(raw, (list, tuple)) and len(raw) > 0 and not isinstance(raw[0], (int, float)):
        print(f"Detected list of {len(raw)} Q-tables — using the last one.")
        raw = raw[-1]

    matrix = to_matrix(raw)
    state_labels, action_labels = label_axes(raw)

    n_states, n_actions = matrix.shape
    print(f"Shape : {n_states} states × {n_actions} actions")
    print(f"Range : [{matrix.min():.4f}, {matrix.max():.4f}]")

    fh = fig_heatmap(matrix, state_labels, action_labels, cmap)
    fp = fig_policy(matrix, state_labels, cmap)

    if save_heatmap:
        fh.savefig(save_heatmap, dpi=dpi, bbox_inches="tight")
        print(f"Heatmap saved → {save_heatmap}")
    if save_policy:
        fp.savefig(save_policy, dpi=dpi, bbox_inches="tight")
        print(f"Policy saved  → {save_policy}")

    if not save_heatmap and not save_policy:
        plt.show()


def parse_args():
    p = argparse.ArgumentParser(description="Visualize a Q-table .pkl as two figures.")
    p.add_argument("pkl_path", help="Path to the .pkl file")
    p.add_argument("--cmap", default="RdYlGn", help="Matplotlib colormap (default: RdYlGn)")
    p.add_argument("--save-heatmap", metavar="FILE", help="Save heatmap figure to FILE")
    p.add_argument("--save-policy",  metavar="FILE", help="Save policy figure to FILE")
    p.add_argument("--dpi", type=int, default=150, help="DPI when saving (default: 150)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    visualize(
        args.pkl_path,
        cmap=args.cmap,
        save_heatmap=args.save_heatmap,
        save_policy=args.save_policy,
        dpi=args.dpi,
    )