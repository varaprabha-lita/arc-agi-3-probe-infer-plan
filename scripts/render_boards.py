"""Render the starting frame of every public game to boards/<game>.png.

Usage (repo root):  .venv/bin/python scripts/render_boards.py [--game ls20,re86]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import arc_agi
from arc_agi import OperationMode

PALETTE = ["#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00", "#AAAAAA",
           "#F012BE", "#FF851B", "#7FDBFF", "#870C25", "#FFFFFF", "#8B4513",
           "#00FF7F", "#4B0082", "#FF69B4", "#556B2F"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--game", default=None)
    args = p.parse_args()
    out = Path("boards")
    out.mkdir(exist_ok=True)
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    ids = [e.game_id.split("-")[0] for e in arc.get_environments()]
    if args.game:
        wanted = {g.strip() for g in args.game.split(",")}
        ids = [g for g in ids if g in wanted]
    cmap = ListedColormap(PALETTE)
    for gid in ids:
        env = arc.make(gid)
        f = env.reset()
        g = np.asarray(f.frame[-1])
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(g, cmap=cmap, vmin=0, vmax=15, interpolation="nearest")
        ax.set_xticks(range(0, 64, 4))
        ax.set_yticks(range(0, 64, 4))
        ax.tick_params(labelsize=6)
        ax.grid(True, color="w", alpha=0.15, linewidth=0.5)
        ax.set_title(f"{gid}  levels={f.win_levels}  actions={list(f.available_actions)}")
        fig.tight_layout()
        fig.savefig(out / f"{gid}.png", dpi=110)
        plt.close(fig)
        print("wrote", out / f"{gid}.png")


if __name__ == "__main__":
    main()
