"""Probe study: what does each action DO in each ARC-AGI-3 public game?

For every game, from a fresh RESET, press each available simple action
REPEATS times in a row and measure the board change after each press:
  - cells changed
  - bounding box of the change
  - whether the change looks like an object translating by (dy, dx)
Also clicks the top-K small rare-coloured objects and reports the effect.

Usage (from the starter repo root, inside the venv):
    .venv/bin/python scripts/probe.py            # all games
    .venv/bin/python scripts/probe.py --game vc33,ls20
Writes probe_results.json next to the repo root.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import deque
from pathlib import Path

import numpy as np

import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction, GameState

REPEATS = 3
TOP_CLICKS = 6


def grid_of(frame) -> np.ndarray | None:
    if frame is None or frame.is_empty():
        return None
    return np.asarray(frame.frame[-1], dtype=np.int16)


def describe_change(before: np.ndarray, after: np.ndarray) -> dict:
    diff = before != after
    n = int(diff.sum())
    if n == 0:
        return {"cells": 0}
    ys, xs = np.nonzero(diff)
    out = {"cells": n, "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]}
    # translation test: cells that vanished vs cells that appeared, per colour
    bg = int(np.bincount(before.ravel(), minlength=16).argmax())
    vanished = {}
    appeared = {}
    for y, x in zip(ys, xs):
        b, a = int(before[y, x]), int(after[y, x])
        if b != bg:
            vanished.setdefault(b, []).append((y, x))
        if a != bg:
            appeared.setdefault(a, []).append((y, x))
    shifts = []
    for c in vanished:
        if c in appeared and len(vanished[c]) == len(appeared[c]):
            v = np.array(sorted(vanished[c]))
            a = np.array(sorted(appeared[c]))
            d = a - v
            if np.all(d == d[0]):
                shifts.append({"colour": c, "dy": int(d[0][0]), "dx": int(d[0][1]), "size": int(len(v))})
    if shifts:
        out["translation"] = shifts
    return out


def click_targets(grid: np.ndarray, k: int) -> list[tuple[int, int, int, int]]:
    h, w = grid.shape
    seen = np.zeros_like(grid, dtype=bool)
    counts = np.bincount(grid.ravel(), minlength=16)
    bg = int(counts.argmax())
    comps = []
    for sy in range(h):
        for sx in range(w):
            if seen[sy, sx] or grid[sy, sx] == bg:
                continue
            c = int(grid[sy, sx])
            q = deque([(sy, sx)])
            seen[sy, sx] = True
            cells = []
            while q:
                y, x = q.popleft()
                cells.append((y, x))
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and grid[ny, nx] == c:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            size = len(cells)
            if size > 400:
                continue
            cy, cx = cells[size // 2]
            score = (1.0 / (1.0 + counts[c])) * 1000.0 / (1.0 + math.log1p(size))
            comps.append((score, cx, cy, c, size))
    comps.sort(reverse=True)
    return [(x, y, c, s) for _, x, y, c, s in comps[:k]]


def probe_game(arc, gid: str) -> dict:
    env = arc.make(gid)
    f = env.reset()
    g0 = grid_of(f)
    res = {
        "game": gid,
        "win_levels": f.win_levels,
        "available_actions": list(f.available_actions),
        "start_colours": {int(c): int(n) for c, n in enumerate(np.bincount(g0.ravel(), minlength=16)) if n},
        "actions": {},
        "clicks": [],
    }
    simple = [a for a in f.available_actions if a in (1, 2, 3, 4, 5, 7)]
    for a in simple:
        env = arc.make(gid)
        f = env.reset()
        prev = grid_of(f)
        seq = []
        for _ in range(REPEATS):
            f = env.step(GameAction.from_id(a))
            cur = grid_of(f)
            if cur is None:
                seq.append({"state": str(f.state)})
                break
            d = describe_change(prev, cur)
            d["state"] = f.state.name
            d["levels"] = f.levels_completed
            seq.append(d)
            prev = cur
            if f.state in (GameState.GAME_OVER, GameState.WIN):
                break
        res["actions"][a] = seq
    if 6 in f.available_actions:
        for x, y, c, size in click_targets(g0, TOP_CLICKS):
            env = arc.make(gid)
            f = env.reset()
            prev = grid_of(f)
            f = env.step(GameAction.ACTION6, data={"x": int(x), "y": int(y)})
            cur = grid_of(f)
            d = describe_change(prev, cur) if cur is not None else {}
            d.update({"x": x, "y": y, "colour": c, "size": size, "state": f.state.name, "levels": f.levels_completed})
            res["clicks"].append(d)
    return res


def summarise(r: dict) -> str:
    lines = [f"=== {r['game']}  levels={r['win_levels']}  actions={r['available_actions']}"]
    for a, seq in r["actions"].items():
        parts = []
        for s in seq:
            if s.get("cells", 0) == 0:
                parts.append("no-op")
            elif "translation" in s:
                t = s["translation"][0]
                parts.append(f"move colour{t['colour']} ({t['dy']:+d},{t['dx']:+d}) size{t['size']}")
            else:
                parts.append(f"{s['cells']} cells")
            if s.get("state") == "GAME_OVER":
                parts[-1] += " DIED"
            if s.get("levels", 0) > 0:
                parts[-1] += " LEVEL!"
        lines.append(f"  A{a}: " + " | ".join(parts))
    for c in r["clicks"]:
        eff = "no-op" if c.get("cells", 0) == 0 else f"{c['cells']} cells"
        if c.get("levels", 0) > 0:
            eff += " LEVEL!"
        lines.append(f"  click ({c['x']},{c['y']}) colour{c['colour']} size{c['size']}: {eff}")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--game", default=None)
    args = p.parse_args()
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    envs = arc.get_environments()
    ids = [e.game_id.split("-")[0] for e in envs]
    if args.game:
        wanted = {g.strip() for g in args.game.split(",")}
        ids = [g for g in ids if g in wanted]
    results = []
    for gid in ids:
        try:
            r = probe_game(arc, gid)
        except Exception as e:  # keep going, report the failure
            r = {"game": gid, "error": repr(e), "actions": {}, "clicks": [], "win_levels": -1, "available_actions": []}
        results.append(r)
        print(summarise(r) if "error" not in r else f"=== {gid} ERROR {r['error']}")
    Path("probe_results.json").write_text(json.dumps(results, indent=1))
    print("\nwrote probe_results.json")


if __name__ == "__main__":
    main()
