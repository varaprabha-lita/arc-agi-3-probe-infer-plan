# Probe, infer, plan: sample-efficient mechanics inference for ARC-AGI-3

Code, data and results for an ARC Prize 2026 Paper Track entry, anchored on the
ARC-AGI-3 Kaggle track (interactive agent games).

**Author:** Dr. Galiveeti Poornima, S-VYASA Deemed to be University, Bengaluru
(Kaggle: `galiveetipoornima`, team *VaraPrabha*).
**Licence:** MIT No Attribution (MIT-0).

## What the agent does

A single pure-NumPy class (`agent/my_agent.py`, no learned parameters, CPU only)
that treats every game as a mechanics-inference problem:

1. **Volatility mask.** Pixels that change on their own (timers, energy bars) are
   masked so they are never mistaken for the effect of an action.
2. **Probe.** Press each movement key once. If exactly one object translates
   rigidly, that object is the avatar; its displacement gives the logical cell
   size, the key→direction map (with an arrow-key prior) and the floor colour.
3. **Navigate.** Downsample to logical cells and run Dijkstra to candidate goal
   cells (floor cost 1, unknown cost 4, learned walls blocked). Every step is
   verified against the predicted position; a mismatch records a wall and the
   colour that blocked, and colours that block repeatedly are treated as walls
   everywhere.
4. **Goal-hypothesis ladder.** When "reach the goal cell" fails, the agent tries
   in turn: *select* (click another same-coloured block and park it in a hollow
   outline), *switch* (Space, then check which object now responds), *fire*,
   *push* (single-box Sokoban) and finally diff-aware *exploration*.
5. **Overlay planner** (collect games such as `re86`): flood-fill the controlled
   shape, find every lattice shift that lands a dot of the shape's colour on the
   shape, and walk the shift that covers the most dots. This plays `re86`
   levels 1–2 at exactly the human baseline (26/26 and 42/42 actions).

Cell size, avatar colours, floor, key map and wall colours transfer across
levels. Any exception drops to exploration so a submission never crashes.

## Why interaction efficiency is the whole game

The Kaggle metric (RHAE) scores each completed level as
`(human_actions / agent_actions)^2`, capped at 1.15, with *all* actions —
including exploration — counted. Exploration is therefore punished
quadratically, and clearing a level with a near-human action count matters far
more than clearing many levels slowly. The agent is designed around that.

## Results

Local, 25 public games, 1000 actions each (`results/scorecard_v10_local.json`):

| Agent | Local RHAE | Levels | Kaggle hidden (public LB) |
|---|---|---|---|
| random starter | 0.00 | 0 | — |
| v6 (probe + navigate) | 0.30 | — | 0.23 |
| v7c (+ goal-hypothesis ladder) | 0.50 | 9 | 0.32 |
| **v10 (+ overlay planner)** | **0.86** | **11** | **0.32** |

The overlay planner adds +0.36 locally and nothing on the hidden set: the
hidden games do not contain that mechanic. Everything the agent earns on the
hidden set comes from the general probe-and-navigate core.

Ablations (v10, 5 seeds, mean ± std; `results/ablation_v10.txt`):

| Configuration | RHAE | Levels / seed |
|---|---|---|
| Full agent | 0.81 ± 0.06 | 11, 10, 8, 11, 11 |
| no colour-wall inference | 0.73 ± 0.06 | 10, 9, 7, 10, 9 |
| no arrow-key prior | 0.74 ± 0.06 | 12, 10, 7, 10, 9 |
| no volatility mask | 0.81 ± 0.09 | 11, 10, 8, 11, 10 |
| Explorer only | 0.30 ± 0.06 | 7, 7, 4, 7, 5 |

Probing + planning is worth about +0.5 RHAE over exploration alone. Volatility
masking contributes nothing measurable once the planner exists.

## Reproducing

```
git clone https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter
cd ARC-AGI-3-Kaggle-Starter && make setup
cp <this repo>/agent/my_agent.py agent/
cp <this repo>/scripts/*.py scripts/
make play-local STEPS=1000                     # all 25 games, ~2 min
.venv/bin/python scripts/probe.py              # per-key board-change study
.venv/bin/python scripts/render_boards.py      # start-board PNGs -> boards/
.venv/bin/python scripts/run_ablation.py --seeds 5
```

Ablation switches are environment variables read by the agent:
`ABL_VOLATILITY`, `ABL_PROBE_NAV`, `ABL_ARROW_PRIOR`, `ABL_COLOUR_WALLS`
(set to `0` to disable); `AGENT_SEED` fixes randomness. The agent writes every
model, plan, block and fallback to `agent_debug.log`.

Kaggle submission: notebook `galiveetipoornima/arc-prize-2026-arc-agi-3-starter`,
version 3 (CPU).

## Layout

```
agent/my_agent.py        the submitted agent (v10)
scripts/                 probe study, board renderer, ablation harness
results/                 scorecards, probe study, ablation table, debug log
boards/                  start boards of the 25 public games
docs/agent_versions/     earlier versions v1–v9 for the record
```
