"""Run the ablation grid: each configuration x N seeds, unattended.

Usage (repo root):   .venv/bin/python scripts/run_ablation.py [--seeds 3] [--steps 1000] [--configs full,explorer]
Results: results/<config>_s<seed>.json (scorecards) and results/summary.txt
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

CONFIGS = {
    "full":            {},
    "no_colour_walls": {"ABL_COLOUR_WALLS": "0"},
    "no_arrow_prior":  {"ABL_ARROW_PRIOR": "0"},
    "no_volatility":   {"ABL_VOLATILITY": "0"},
    "explorer":        {"ABL_PROBE_NAV": "0"},           # v1-style diff explorer only
}


def run_one(cfg: str, env_extra: dict, seed: int, steps: int, out: Path) -> float | None:
    env = dict(os.environ, AGENT_SEED=str(seed), **env_extra)
    py = Path(".venv/bin/python")
    cmd = [str(py), "scripts/play_local.py", "--max-steps", str(steps)]
    print(f"[{cfg} seed={seed}] running ...", flush=True)
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    m = re.search(r"Aggregate scorecard score: ([0-9.]+)", proc.stdout)
    score = float(m.group(1)) if m else None
    if Path("scorecard.json").exists():
        shutil.copy("scorecard.json", out / f"{cfg}_s{seed}.json")
    if Path("agent_debug.log").exists():
        shutil.move("agent_debug.log", out / f"{cfg}_s{seed}.log")
    print(f"[{cfg} seed={seed}] score={score}", flush=True)
    if score is None:
        print(proc.stdout[-2000:], proc.stderr[-2000:])
    return score


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--configs", default=",".join(CONFIGS))
    args = p.parse_args()
    out = Path("results")
    out.mkdir(exist_ok=True)
    table = {}
    for cfg in args.configs.split(","):
        cfg = cfg.strip()
        scores = []
        for seed in range(1, args.seeds + 1):
            f = out / f"{cfg}_s{seed}.json"
            if f.exists():                       # resume: skip finished runs
                sc = json.load(open(f))["score"]
                print(f"[{cfg} seed={seed}] cached score={sc}")
                scores.append(sc)
                continue
            sc = run_one(cfg, CONFIGS[cfg], seed, args.steps, out)
            if sc is not None:
                scores.append(sc)
        table[cfg] = scores

    lines = ["config            n   mean    std    levels(sum over games, per seed)"]
    for cfg, scores in table.items():
        if not scores:
            continue
        levels = []
        for seed in range(1, args.seeds + 1):
            f = out / f"{cfg}_s{seed}.json"
            if f.exists():
                sc = json.load(open(f))
                levels.append(sum(e["levels_completed"] for e in sc["environments"]))
        std = statistics.stdev(scores) if len(scores) > 1 else 0.0
        lines.append(f"{cfg:16} {len(scores):2}  {statistics.mean(scores):6.3f} {std:6.3f}   {levels}")
    summary = "\n".join(lines)
    print("\n" + summary)
    (out / "summary.txt").write_text(summary)


if __name__ == "__main__":
    main()
