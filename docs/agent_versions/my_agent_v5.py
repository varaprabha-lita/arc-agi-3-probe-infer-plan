"""ARC-AGI-3 agent v2: probe → plan → verify, with an explorer fallback.

Pipeline per game:
  1. Volatility mask   pixels that change on their own (HUD, timers) are
                       ignored by every "did the board change?" test.
  2. Probe             press each of ACTION1-4 once; detect the object that
                       translates, its colours, the pixel cell size, and the
                       direction of each key (prior: 1=up 2=down 3=left 4=right).
  3. Navigate          downsample the board to logical cells, pick candidate
                       goal cells (rare colours), BFS the avatar to them,
                       verify each step, learn walls from blocked moves.
  4. Fallback          v1 diff-aware explorer (used for click-only games,
                       when no avatar is found, or when navigation stalls).

Knowledge that transfers across levels: cell size, avatar colours, key→direction map.
Any exception inside the smart path drops to the explorer so a submission never crashes.

Ablation switches are the USE_* class attributes.
"""
from __future__ import annotations

import hashlib
import math
import random
import time
from collections import defaultdict, deque
from typing import Any

import numpy as np

from arcengine import FrameData, GameAction, GameState
from agents.agent import Agent

PRIOR_DIR = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}   # (dy, dx)


class MyAgent(Agent):
    MAX_ACTIONS = 1000

    # ---- ablation switches
    USE_VOLATILITY = True
    USE_PROBE_NAV = True
    USE_ARROW_PRIOR = True

    NAV_BUDGET = 160          # actions in navigate mode per level before falling back
    MAX_CLICK_TARGETS = 12
    EPSILON = 0.08
    DEAD_AFTER = 2

    # ------------------------------------------------------------------ setup
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        import os
        seed_env = os.environ.get("AGENT_SEED")
        base = int(seed_env) if seed_env else int(time.time() * 1e6)
        self.rng = random.Random(base + sum(map(ord, self.game_id)))
        # volatility
        self.change_count = np.zeros((64, 64), dtype=np.int32)
        self.transitions = 0
        self.prev_grid: np.ndarray | None = None
        # transferable knowledge
        self.cell: int | None = None
        self.avatar_colours: set[int] = set()
        self.avatar_size = 0
        self.dir_map: dict[int, tuple[int, int]] = {}
        self.floor: int | None = None
        self.wall_colours: dict[int, int] = defaultdict(int)     # colour -> times it blocked us
        self.pass_colours: set[int] = set()                       # non-floor colours we walked onto
        # per-level state
        self.levels = 0
        self._new_level()
        # explorer memory
        self.stats: dict[str, dict[tuple, list[int]]] = defaultdict(dict)
        self.dead_sig: dict[tuple, list[int]] = defaultdict(lambda: [0, 0])
        self.last_hash: str | None = None
        self.last_key: tuple | None = None
        self.last_sig: tuple | None = None

    def _log(self, msg: str) -> None:
        try:
            with open("agent_debug.log", "a") as f:
                f.write(f"{self.game_id} L{self.levels} t{self.transitions}: {msg}\n")
        except Exception:
            pass

    def _new_level(self) -> None:
        self.mode = "probe" if self.USE_PROBE_NAV else "explore"
        self.probe_queue: list[int] = []
        self.probe_pending: int | None = None
        self.probe_results: dict[int, dict | None] = {}
        self.walls: set[tuple[int, int]] = set()
        self.tried_goals: set[tuple[int, int]] = set()
        self.plan: list[int] = []
        self.goal: tuple[int, int] | None = None
        self.expected: tuple[int, int] | None = None
        self.nav_actions = 0
        self.pending_interact = False
        self.avatar_pos: tuple[int, int] | None = None   # pixel bbox top-left
        self.reprobe_pending = False
        self.reprobes = 0
        self.goal_blocks = 0

    @property
    def name(self) -> str:
        return f"{super().name}.v5"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _grid(frame: FrameData) -> np.ndarray | None:
        if not frame.frame:
            return None
        return np.asarray(frame.frame[-1], dtype=np.int16)

    def _volatile(self) -> np.ndarray:
        if not self.USE_VOLATILITY or self.transitions < 4:
            return np.zeros((64, 64), dtype=bool)
        return self.change_count > 0.5 * self.transitions

    def _stable_hash(self, grid: np.ndarray) -> str:
        g = grid.copy()
        g[self._volatile()] = -1
        return hashlib.md5(g.tobytes()).hexdigest()

    def _components(self, grid: np.ndarray, colours: set[int] | None = None, max_size: int = 400):
        h, w = grid.shape
        seen = np.zeros_like(grid, dtype=bool)
        vol = self._volatile()
        counts = np.bincount(grid.ravel(), minlength=16)
        bg = int(counts.argmax())
        out = []
        for sy in range(h):
            for sx in range(w):
                if seen[sy, sx] or vol[sy, sx]:
                    continue
                c = int(grid[sy, sx])
                if c == bg or (colours is not None and c not in colours):
                    seen[sy, sx] = True
                    continue
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
                if len(cells) <= max_size:
                    ys = [p[0] for p in cells]
                    xs = [p[1] for p in cells]
                    out.append({"colour": c, "size": len(cells), "cells": cells,
                                "bbox": (min(ys), min(xs), max(ys), max(xs)), "counts": counts})
        return out

    def _translation(self, before: np.ndarray, after: np.ndarray) -> dict | None:
        """Detect an object that moved rigidly: returns colours, shift, size, new bbox."""
        vol = self._volatile()
        diff = (before != after) & ~vol
        if not diff.any():
            return None
        bg = int(np.bincount(before.ravel(), minlength=16).argmax())
        ys, xs = np.nonzero(diff)
        vanished: dict[int, list] = defaultdict(list)
        appeared: dict[int, list] = defaultdict(list)
        for y, x in zip(ys, xs):
            b, a = int(before[y, x]), int(after[y, x])
            if b != bg:
                vanished[b].append((y, x))
            if a != bg:
                appeared[a].append((y, x))
        shifts = {}
        for c in vanished:
            if c in appeared and len(vanished[c]) == len(appeared[c]):
                v = np.array(sorted(vanished[c]))
                a = np.array(sorted(appeared[c]))
                d = a - v
                if np.all(d == d[0]) and (d[0] != 0).any():
                    dy, dx = int(d[0][0]), int(d[0][1])
                    # a keypress moves an object a few pixels along one axis (allow 1 px sprite jitter)
                    if min(abs(dy), abs(dx)) <= 1 and 2 <= max(abs(dy), abs(dx)) <= 8 and len(v) <= 120:
                        if abs(dy) >= abs(dx):
                            dx = 0
                        else:
                            dy = 0
                        shifts[c] = (dy, dx)
        if not shifts:
            return None
        # majority shift
        best = max(set(shifts.values()), key=lambda s: sum(1 for v in shifts.values() if v == s))
        colours = {c for c, s in shifts.items() if s == best}
        # locate the moved object in `after`
        comps = self._components(after, colours)
        if not comps:
            return None
        comp = max(comps, key=lambda k: k["size"])
        # floor = what is revealed where the object used to be
        revealed = [int(after[y, x]) for c in colours for (y, x) in vanished[c] if int(after[y, x]) not in colours]
        floor = max(set(revealed), key=revealed.count) if revealed else bg
        return {"colours": colours, "shift": best, "size": comp["size"], "bbox": comp["bbox"], "floor": floor}

    def _find_avatar(self, grid: np.ndarray) -> tuple[int, int] | None:
        comps = self._components(grid, self.avatar_colours)
        comps = [c for c in comps if 0.5 * self.avatar_size <= c["size"] <= 1.5 * self.avatar_size]
        if not comps:
            return None
        ref = self.expected or self.avatar_pos
        if ref is None:
            comp = max(comps, key=lambda c: c["size"])
            return (comp["bbox"][0], comp["bbox"][1])
        dist = lambda c: max(abs(c["bbox"][0] - ref[0]), abs(c["bbox"][1] - ref[1]))
        comp = min(comps, key=dist)
        if self.avatar_pos is not None:
            d_last = max(abs(comp["bbox"][0] - self.avatar_pos[0]), abs(comp["bbox"][1] - self.avatar_pos[1]))
            if d_last > (self.cell or 1) + 1 and dist(comp) > 1:
                return None   # nothing near where the avatar should be: lost
        return (comp["bbox"][0], comp["bbox"][1])

    # logical grid: cell coordinate of a pixel, and colour sampled at cell centre
    def _cell_of(self, py: int, px: int) -> tuple[int, int]:
        return (py // self.cell, px // self.cell)

    def _logical(self, grid: np.ndarray) -> np.ndarray:
        s = self.cell
        n = 64 // s
        return grid[: n * s, : n * s].reshape(n, s, n, s)[:, s // 2, :, s // 2]

    def _plan_to(self, lg: np.ndarray, start: tuple[int, int], goal: tuple[int, int],
                 bg: int) -> list[tuple[int, int]] | None:
        import heapq
        n = lg.shape[0]
        dist = {start: 0}
        prev = {start: None}
        heap = [(0, start)]
        while heap:
            dcur, cur = heapq.heappop(heap)
            if cur == goal:
                path = []
                while cur is not None:
                    path.append(cur)
                    cur = prev[cur]
                return path[::-1]
            if dcur > dist.get(cur, 1e9):
                continue
            for d in PRIOR_DIR.values():
                nxt = (cur[0] + d[0], cur[1] + d[1])
                if not (0 <= nxt[0] < n and 0 <= nxt[1] < n) or nxt in self.walls:
                    continue
                c = int(lg[nxt])
                if nxt != goal and self.wall_colours.get(c, 0) >= 2:
                    continue                                 # this colour is a wall everywhere
                step = 1 if (c == bg or c in self.pass_colours) else 4
                nd = dcur + step
                if nd < dist.get(nxt, 1e9):
                    dist[nxt] = nd
                    prev[nxt] = cur
                    heapq.heappush(heap, (nd, nxt))
        return None

    def _step_action(self, d: tuple[int, int]) -> int | None:
        for a, dd in self.dir_map.items():
            if dd == d:
                return a
        return None

    # --------------------------------------------------------------- explorer
    def _click_targets(self, grid):
        comps = self._components(grid)
        counts = np.bincount(grid.ravel(), minlength=16)
        scored = []
        for c in comps:
            rarity = 1.0 / (1.0 + counts[c["colour"]])
            score = rarity * 1000.0 / (1.0 + math.log1p(c["size"]))
            y, x = c["cells"][c["size"] // 2]
            scored.append((score, x, y, (c["colour"], min(c["size"], 64) // 4)))
        scored.sort(reverse=True)
        return [(x, y, sig) for _, x, y, sig in scored[: self.MAX_CLICK_TARGETS]]

    def _score(self, h, key, sig):
        st = self.stats[h].get(key)
        if st is None:
            base = 2.0
        else:
            tries, changed = st
            base = 0.05 if (changed == 0 and tries >= self.DEAD_AFTER) else changed / tries + 1 / math.sqrt(tries)
        if sig is not None:
            t, ch = self.dead_sig[sig]
            if t >= 4 and ch == 0:
                base *= 0.1
        return base + self.rng.random() * 1e-3

    def _explore(self, grid: np.ndarray, latest: FrameData, changed_last: bool) -> GameAction:
        h = self._stable_hash(grid)
        if self.last_hash is not None and self.last_key is not None:
            st = self.stats[self.last_hash].setdefault(self.last_key, [0, 0])
            st[0] += 1
            st[1] += int(changed_last)
            if self.last_sig is not None:
                ds = self.dead_sig[self.last_sig]
                ds[0] += 1
                ds[1] += int(changed_last)
        legal = [a for a in latest.available_actions if a != 0] or [1, 2, 3, 4, 5, 6, 7]
        cands = []
        for a in legal:
            if a == 6:
                for x, y, sig in self._click_targets(grid):
                    cands.append((self._score(h, ("C", x, y), sig), 6, x, y, sig))
                cands.append((0.5, 6, self.rng.randrange(64), self.rng.randrange(64), None))
            else:
                cands.append((self._score(h, ("A", a), None), a, None, None, None))
        choice = self.rng.choice(cands) if self.rng.random() < self.EPSILON else max(cands, key=lambda c: c[0])
        _, a, x, y, sig = choice
        action = GameAction.from_id(a)
        if action.is_complex():
            action.set_data({"x": int(x), "y": int(y)})
            action.reasoning = {"mode": "explore", "click": [int(x), int(y)]}
            self.last_key, self.last_sig = ("C", int(x), int(y)), sig
        else:
            action.reasoning = {"mode": "explore", "action": a}
            self.last_key, self.last_sig = ("A", a), None
        self.last_hash = h
        return action

    # ---------------------------------------------------------------- driver
    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self.prev_grid = None
            self.last_hash = self.last_key = None
            self.plan, self.expected = [], None
            return GameAction.RESET

        grid = self._grid(latest_frame)
        if grid is None:
            return GameAction.RESET

        # level bookkeeping
        if latest_frame.levels_completed > self.levels:
            self.levels = latest_frame.levels_completed
            self._new_level()
            if self.cell and self.avatar_colours and self.dir_map:
                self.mode = "navigate"          # transfer: skip probing on later levels
            self.prev_grid = None

        # volatility statistics + "did anything real change?"
        changed_last = False
        if self.prev_grid is not None:
            d = self.prev_grid != grid
            changed_last = bool((d & ~self._volatile()).any())
            # changes explained by an object moving are not "volatility"
            try:
                tr = self._translation(self.prev_grid, grid)
            except Exception:
                tr = None
            dv = d.copy()
            if tr:
                for c in tr["colours"]:
                    dv &= ~((self.prev_grid == c) | (grid == c))
            self.change_count += dv
            self.transitions += 1
        before = self.prev_grid
        self.prev_grid = grid

        try:
            if self.mode == "probe":
                act = self._probe(grid, before, latest_frame)
                if act is not None:
                    return act
            if self.mode == "navigate":
                act = self._navigate(grid, before, latest_frame)
                if act is not None:
                    return act
        except Exception as e:  # never crash a submission
            self.mode = "explore"
            self.plan = []
            self._err = repr(e)
            import traceback
            self._log("EXCEPTION " + traceback.format_exc().replace("\n", " | "))
        return self._explore(grid, latest_frame, changed_last)

    # ----------------------------------------------------------------- probe
    def _probe(self, grid, before, latest) -> GameAction | None:
        keys = [a for a in latest.available_actions if a in (1, 2, 3, 4)]
        if not keys:
            self.mode = "explore"
            return None
        # record result of the pending probe
        if self.probe_pending is not None and before is not None:
            self.probe_results[self.probe_pending] = self._translation(before, grid)
            self.probe_pending = None
        if not self.probe_queue and not self.probe_results:
            self.probe_queue = list(keys)
        if self.probe_queue:
            a = self.probe_queue.pop(0)
            self.probe_pending = a
            act = GameAction.from_id(a)
            act.reasoning = {"mode": "probe", "action": a}
            return act
        # all probes done → interpret
        moved = {a: r for a, r in self.probe_results.items() if r}
        self._log("probe results " + str({a: (r["shift"], r["size"], sorted(r["colours"])) if r else None for a, r in self.probe_results.items()}))
        if not moved:
            self._log("FALLBACK: no key produced a translation")
            self.mode = "explore"
            return None
        # identity = (colours, size); prefer the object that moved in the most probes,
        # that obeyed the arrow prior, and that is not hugging the board edge (HUD boxes)
        from collections import Counter
        ident = lambda r: (frozenset(r["colours"]), r["size"])
        counts = Counter(ident(r) for r in moved.values())
        def quality(k):
            rs = [ (a, r) for a, r in moved.items() if ident(r) == k ]
            prior_ok = sum(1 for a, r in rs if (int(np.sign(r["shift"][0])), int(np.sign(r["shift"][1]))) == PRIOR_DIR.get(a))
            y0, x0, y1, x1 = rs[-1][1]["bbox"]
            edge = min(y0, x0, 63 - y1, 63 - x1) <= 4
            return (counts[k], prior_ok, not edge, -k[1])
        best_id = max(counts, key=quality)
        rs = {a: r for a, r in moved.items() if ident(r) == best_id}
        sizes = [max(abs(r["shift"][0]), abs(r["shift"][1])) for r in rs.values()]
        self.cell = max(1, min(Counter(sizes).most_common(), key=lambda kv: (-kv[1], kv[0]))[0])
        best = list(rs.values())[-1]
        self.avatar_colours = set(best["colours"])
        self.avatar_size = best["size"]
        floors = [r["floor"] for r in rs.values()]
        self.floor = max(set(floors), key=floors.count)
        for a in keys:
            r = rs.get(a)
            if r:
                dy, dx = r["shift"]
                self.dir_map[a] = (int(np.sign(dy)), int(np.sign(dx)))
            elif self.USE_ARROW_PRIOR:
                self.dir_map[a] = PRIOR_DIR[a]
        self.avatar_pos = best["bbox"][:2]
        self._log(f"model cell={self.cell} avatar={sorted(self.avatar_colours)} size={self.avatar_size} floor={self.floor} dirs={self.dir_map} pos={self.avatar_pos}")
        self.mode = "navigate"
        return self._navigate(grid, before, latest)

    # -------------------------------------------------------------- navigate
    def _navigate(self, grid, before, latest) -> GameAction | None:
        if self.nav_actions >= self.NAV_BUDGET:
            self._log(f"FALLBACK: nav budget exhausted, walls={len(self.walls)} goals_tried={len(self.tried_goals)}")
            self.mode = "explore"
            return None
        # result of a re-identification move
        if self.reprobe_pending:
            self.reprobe_pending = False
            tr = self._translation(before, grid) if before is not None else None
            if tr and set(tr["colours"]) == self.avatar_colours:
                self.avatar_pos = tr["bbox"][:2]
                self.expected = None
                self._log(f"re-identified avatar at {self.avatar_pos}")
            else:
                self._log("FALLBACK: re-identification failed")
                self.mode = "explore"
                return None
        pos = self._find_avatar(grid)
        if pos is None:
            if self.reprobes < 3 and self.dir_map:
                self.reprobes += 1
                self.reprobe_pending = True
                self.plan, self.expected = [], None
                a = self.rng.choice(list(self.dir_map))
                self.nav_actions += 1
                self._log(f"avatar lost (last={self.avatar_pos}); re-probing with {a}")
                act = GameAction.from_id(a)
                act.reasoning = {"mode": "reprobe"}
                return act
            self._log(f"FALLBACK: avatar not found (expected={self.expected}, last={self.avatar_pos}, nav_actions={self.nav_actions})")
            self.mode = "explore"
            return None
        lg = self._logical(grid)
        bg = self.floor if self.floor is not None else int(np.bincount(lg.ravel()).argmax())
        # verify last step
        if self.expected is not None:
            off = max(abs(pos[0] - self.expected[0]), abs(pos[1] - self.expected[1]))
            ec = getattr(self, "expected_colour", None)
            if off > 1:
                # blocked: the cell we tried to enter is a wall
                self.walls.add(self._cell_of(*self.expected))
                if ec is not None and ec != bg:
                    self.wall_colours[ec] += 1
                self._log(f"blocked: expected {self.expected} got {pos}; wall {self._cell_of(*self.expected)} colour {ec} (x{self.wall_colours.get(ec, 0)})")
                self.plan = []
                self.goal_blocks += 1
                if self.goal is not None and self.goal_blocks >= 3:
                    self._log(f"giving up on goal {self.goal}")
                    self.tried_goals.add(self.goal)
                    self.goal = None
                    self.goal_blocks = 0
            else:
                pos = self.expected      # arrived (allow 1 px sprite jitter)
                if ec is not None and ec != bg and ec not in self.pass_colours:
                    self.pass_colours.add(ec)
                    self._log(f"colour {ec} is passable")
            self.expected = None
        self.avatar_pos = pos
        here = self._cell_of(*pos)
        big = {int(c) for c, n in enumerate(np.bincount(lg.ravel(), minlength=16)) if n > 0.4 * lg.size}
        # HUD bands: a single non-floor colour filling most of a logical row/column
        hud = set()
        n = lg.shape[0]
        for i in range(n):
            for line, cells in ((lg[i, :], [(i, j) for j in range(n)]), (lg[:, i], [(j, i) for j in range(n)])):
                cnts = np.bincount(line, minlength=16)
                c = int(cnts.argmax())
                if c != bg and cnts[c] >= 0.75 * n:
                    hud.update(cc for cc in cells if int(lg[cc]) == c)

        # arrived at goal with nothing happening → optional interact, then next goal
        if self.goal is not None and here == self.goal and not self.plan:
            self.tried_goals.add(self.goal)
            self.goal = None
            if 5 in latest.available_actions and not self.pending_interact:
                self.pending_interact = True
                self.nav_actions += 1
                act = GameAction.ACTION5
                act.reasoning = {"mode": "interact"}
                return act
            self.pending_interact = False

        if not self.plan:
            # candidate goals: non-background, non-avatar cells, rarest colours first, nearest first
            counts = np.bincount(lg.ravel(), minlength=16)
            cands = []
            for cy in range(lg.shape[0]):
                for cx in range(lg.shape[1]):
                    c = int(lg[cy, cx])
                    if c == bg or c in big or c in self.avatar_colours or (cy, cx) in self.tried_goals or (cy, cx) in self.walls or (cy, cx) in hud:
                        continue
                    if self.wall_colours.get(c, 0) >= 3:
                        continue                              # a colour that blocks us repeatedly is not a goal either
                    if self._volatile()[cy * self.cell + self.cell // 2, cx * self.cell + self.cell // 2]:
                        continue
                    dist = abs(cy - here[0]) + abs(cx - here[1])
                    cands.append((counts[c], dist, (cy, cx)))
            cands.sort()
            for _, _, g in cands[:40]:
                path = self._plan_to(lg, here, g, bg)
                if path and len(path) > 1:
                    self.goal = g
                    self.goal_blocks = 0
                    self.plan = path[1:]
                    break
            if not self.plan:
                self._log(f"FALLBACK: no reachable goal (cands={len(cands)}, walls={len(self.walls)}, tried={len(self.tried_goals)}, here={here}, bg={bg})")
                self.mode = "explore"
                return None
            self._log(f"plan to {self.goal} colour={int(lg[self.goal])} len={len(self.plan)} from {here}")

        nxt = self.plan.pop(0)
        d = (nxt[0] - here[0], nxt[1] - here[1])
        a = self._step_action(d)
        if a is None or a not in latest.available_actions:
            self.walls.add(nxt)
            self.plan = []
            return self._navigate(grid, before, latest) if self.nav_actions < self.NAV_BUDGET else None
        self.expected = (pos[0] + d[0] * self.cell, pos[1] + d[1] * self.cell)
        self.expected_colour = int(lg[nxt])
        self.nav_actions += 1
        act = GameAction.from_id(a)
        act.reasoning = {"mode": "navigate", "to": list(nxt), "goal": list(self.goal)}
        return act
