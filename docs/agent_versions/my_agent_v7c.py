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

    # ---- ablation switches (override with env vars ABL_VOLATILITY / ABL_PROBE_NAV / ABL_ARROW_PRIOR / ABL_COLOUR_WALLS = 0|1)
    import os as _os
    USE_VOLATILITY = _os.environ.get("ABL_VOLATILITY", "1") == "1"
    USE_PROBE_NAV = _os.environ.get("ABL_PROBE_NAV", "1") == "1"
    USE_ARROW_PRIOR = _os.environ.get("ABL_ARROW_PRIOR", "1") == "1"
    USE_COLOUR_WALLS = _os.environ.get("ABL_COLOUR_WALLS", "1") == "1"

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
        self.box_colours: set[int] = set()                        # colours that moved when we walked into them
        self.fire_axis: str | None = None                         # "col" or "row" once a shot has visibly worked
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
        self.hyp = "reach"                    # reach -> select -> switch -> fire -> push -> explore
        self.selected: set[tuple[int, int]] = set()   # blocks already clicked this level (pixel bbox)
        self.parked: set[tuple[int, int]] = set()     # outline cells we have already put an object into
        self.goal_is_outline = False
        self.giveups: dict[tuple[int, int], int] = defaultdict(int)
        self.switches = 0
        self.tried_fire: set[tuple[int, int]] = set()
        self.goal_colour_at: dict[tuple[int, int], int] = {}
        self.seen_targets: dict[tuple[int, int], int] = {}     # every rare-colour cell seen this level -> colour
        self.fire_target: tuple | None = None      # (target, axis) of the planned shot
        self.fire_pending: tuple | None = None     # set when ACTION5 is actually sent
        self.expected_cell: tuple[int, int] | None = None
        self.expected_dir: tuple[int, int] | None = None

    @property
    def name(self) -> str:
        return f"{super().name}.v7c"

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
        cand = [(dy, 0) for dy in range(-8, 9) if abs(dy) >= 2] + [(0, dx) for dx in range(-8, 9) if abs(dx) >= 2]
        for c in vanished:
            if c not in appeared:
                continue
            nv, na = len(vanished[c]), len(appeared[c])
            if nv > 120 or na > 120 or min(nv, na) < 0.8 * max(nv, na):
                continue
            best, best_n = None, 0
            for dy, dx in cand:
                n_v = sum(1 for (y, x) in vanished[c]
                          if 0 <= y + dy < 64 and 0 <= x + dx < 64 and after[y + dy, x + dx] == c)
                n_a = sum(1 for (y, x) in appeared[c]
                          if 0 <= y - dy < 64 and 0 <= x - dx < 64 and before[y - dy, x - dx] == c)
                n = min(n_v / nv, n_a / na)
                if n > best_n:
                    best, best_n = (dy, dx), n
            if best is not None and best_n >= 0.8:
                shifts[c] = best
        if not shifts:
            return None
        # majority shift
        best = max(set(shifts.values()), key=lambda s: sum(1 for v in shifts.values() if v == s))
        colours = {c for c, s in shifts.items() if s == best}
        # locate the moved object in `after`
        comps = self._components(after, colours)
        if not comps:
            return None
        moved_cells = {tuple(p) for c in colours for p in appeared[c]}
        hit = [k for k in comps if moved_cells & set(k["cells"])]
        comp = max(hit or comps, key=lambda k: k["size"])
        n_changed = sum(len(vanished[c]) for c in colours)
        if comp["size"] > 120 or n_changed < 0.3 * comp["size"]:
            return None          # a big static region (floor/wall) with a few pixels changing is not a mover
        # floor = what is revealed where the object used to be
        revealed = [int(after[y, x]) for c in colours for (y, x) in vanished[c] if int(after[y, x]) not in colours]
        floor = max(set(revealed), key=revealed.count) if revealed else bg
        return {"colours": colours, "shift": best, "size": comp["size"], "bbox": comp["bbox"], "floor": floor}

    def _find_avatar(self, grid: np.ndarray) -> tuple[int, int] | None:
        allc = self._components(grid, self.avatar_colours)
        comps = [c for c in allc if 0.5 * self.avatar_size <= c["size"] <= 1.5 * self.avatar_size]
        ref = self.expected or self.avatar_pos
        if not comps:
            # the avatar may have merged with a same-coloured neighbour: accept a bigger blob that contains ref
            if ref is not None:
                for c in allc:
                    y0, x0, y1, x1 = c["bbox"]
                    if c["size"] > 1.5 * self.avatar_size and y0 <= ref[0] <= y1 and x0 <= ref[1] <= x1:
                        return (int(ref[0]), int(ref[1]))
            return None
        if ref is None:
            comp = max(comps, key=lambda c: c["size"])
            return (comp["bbox"][0], comp["bbox"][1])
        dist = lambda c: max(abs(c["bbox"][0] - ref[0]), abs(c["bbox"][1] - ref[1]))
        comp = min(comps, key=dist)
        if self.avatar_pos is not None:
            d_last = max(abs(comp["bbox"][0] - self.avatar_pos[0]), abs(comp["bbox"][1] - self.avatar_pos[1]))
            if d_last > (self.cell or 1) + 1 and dist(comp) > 1:
                return None   # nothing near where the avatar should be: lost
        self.avatar_bbox = comp["bbox"]
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
                if self.USE_COLOUR_WALLS and nxt != goal and self.wall_colours.get(c, 0) >= 2:
                    continue                                 # this colour is a wall everywhere
                step = 1 if (c == bg or c in self.pass_colours) else 4
                nd = dcur + step
                if nd < dist.get(nxt, 1e9):
                    dist[nxt] = nd
                    prev[nxt] = cur
                    heapq.heappush(heap, (nd, nxt))
        return None

    def _push_plan(self, lg, here, bg, targets):
        """Plan pushes of one box (colour in box_colours) onto one target cell.
        Returns the avatar cell path for the FIRST push: walk to the cell behind the box, then step into it."""
        n = lg.shape[0]
        boxes = [(cy, cx) for cy in range(n) for cx in range(n) if int(lg[cy, cx]) in self.box_colours]
        def colour_at(t):
            c = int(lg[t])
            return self.seen_targets.get(t, self.goal_colour_at.get(t, c)) if (t == here or c in self.avatar_colours) else c
        targets = [t for t in targets if colour_at(t) not in self.box_colours and colour_at(t) != bg
                   and colour_at(t) not in self.avatar_colours]
        if not boxes or not targets:
            return None

        def free(c, box):
            return (0 <= c[0] < n and 0 <= c[1] < n and c != box and c not in self.walls
                    and c not in boxes and self.wall_colours.get(int(lg[c]), 0) < 2)

        def reach(start, box):
            seen = {start}
            q = deque([start])
            while q:
                cur = q.popleft()
                for d in PRIOR_DIR.values():
                    nxt = (cur[0] + d[0], cur[1] + d[1])
                    if nxt not in seen and free(nxt, box):
                        seen.add(nxt)
                        q.append(nxt)
            return seen

        boxes.sort(key=lambda b: abs(b[0] - here[0]) + abs(b[1] - here[1]))
        for box in boxes[:3]:
            start = (box, here)
            prev = {(box, min(reach(here, box))): None}
            q = deque([start])
            found = None
            steps = 0
            while q and steps < 4000:
                steps += 1
                b, av = q.popleft()
                if b in targets:
                    found = (b, av)
                    break
                r = reach(av, b)
                for d in PRIOR_DIR.values():
                    behind = (b[0] - d[0], b[1] - d[1])
                    ahead = (b[0] + d[0], b[1] + d[1])
                    if behind in r and free(ahead, b):
                        key = (ahead, min(reach(b, ahead)))
                        if key not in prev:
                            prev[key] = ((b, min(r)), d, av)
                            q.append((ahead, b))
            if found is None:
                continue
            # walk back to the first push
            key = (found[0], min(reach(found[1], found[0])))
            first = None
            while prev.get(key) is not None:
                pb, d, av = prev[key]
                first = (pb[0], d)
                key = pb
            if first is None:
                continue
            box0, d = first
            behind = (box0[0] - d[0], box0[1] - d[1])
            path = self._plan_to(lg, here, behind, bg) if behind != here else [here]
            if path is None:
                continue
            self._log(f"push plan: box {box0} dir {d} via {behind}")
            return path[1:] + [box0]
        return None

    def _outline_cells(self, grid: np.ndarray, bg: int) -> dict[tuple[int, int], int]:
        """Logical cells whose centre is floor but whose border pixels are one other colour (hollow frames)."""
        s = self.cell
        if s < 3:
            return {}
        n = 64 // s
        out = {}
        for cy in range(n):
            for cx in range(n):
                block = grid[cy * s:(cy + 1) * s, cx * s:(cx + 1) * s]
                if int(block[s // 2, s // 2]) != bg:
                    continue
                ring = np.concatenate([block[0, :], block[-1, :], block[:, 0], block[:, -1]])
                cnt = np.bincount(ring, minlength=16)
                c = int(cnt.argmax())
                if c != bg and cnt[c] >= 0.75 * ring.size and c not in self.avatar_colours:
                    out[(cy, cx)] = c
        return out

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
        # did the last shot remove its target?
        if self.fire_pending is not None and before is not None:
            t, axis = self.fire_pending
            self.fire_pending = None
            if int(self._logical(before)[t]) != int(self._logical(grid)[t]):
                self.fire_axis = axis
                self.seen_targets.pop(t, None)
                self._log(f"shot from {axis} removed target {t}")
        # a select/switch was just sent: now press one key so the newly controlled object reveals itself
        if self.reprobe_pending and not getattr(self, "reprobe_key_sent", False):
            self.reprobe_key_sent = True
            a = self.rng.choice(list(self.dir_map)) if self.dir_map else 1
            self.nav_actions += 1
            act = GameAction.from_id(a)
            act.reasoning = {"mode": "reprobe"}
            return act
        # result of a re-identification move
        if self.reprobe_pending:
            self.reprobe_pending = False
            self.reprobe_key_sent = False
            tr = self._translation(before, grid) if before is not None else None
            if tr:
                moved_far = self.avatar_pos is not None and max(abs(tr["bbox"][0] - self.avatar_pos[0]), abs(tr["bbox"][1] - self.avatar_pos[1])) > self.cell + 1
                if moved_far and set(tr["colours"]) == self.avatar_colours:
                    self.tried_goals.clear()
                    self.hyp = "reach"
                    self._log("control moved to another object of the same colours")
                if set(tr["colours"]) != self.avatar_colours or tr["size"] != self.avatar_size:
                    self.avatar_colours = set(tr["colours"])
                    self.avatar_size = tr["size"]
                    self.tried_goals.clear()
                    self.hyp = "reach"
                    self._log(f"control is now object colours={sorted(self.avatar_colours)} size={self.avatar_size}")
                self.avatar_pos = tr["bbox"][:2]
                self.expected = None
                self._log(f"re-identified avatar at {self.avatar_pos}")
            else:
                self.reprobes += 1
                if self.reprobes <= 4:
                    self._log("re-identification found no mover; continuing hypothesis ladder")
                    self.plan, self.expected = [], None
                    self.hyp = "select" if self.hyp == "reach" else self.hyp
                else:
                    self._log("FALLBACK: re-identification failed")
                    self.mode = "explore"
                    return None
        pos = self._find_avatar(grid)
        if pos is None:
            if self.reprobes < 3 and self.dir_map:
                self.reprobes += 1
                self.reprobe_pending = True
                self.reprobe_key_sent = True
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
            if ec in self.avatar_colours:
                ec = None
            if off > 1:
                # blocked: the cell we tried to enter is a wall
                wall_cell = self.expected_cell or self._cell_of(*self.expected)
                if wall_cell != self.goal:
                    self.walls.add(wall_cell)
                if ec is not None and ec != bg:
                    self.wall_colours[ec] += 1
                self._log(f"blocked: expected {self.expected} got {pos}; wall {wall_cell} colour {ec} (x{self.wall_colours.get(ec, 0)})")
                self.plan = []
                self.goal_blocks += 1
                if self.goal is not None and self.goal_blocks >= 3:
                    self.giveups[self.goal] += 1
                    self._log(f"giving up on goal {self.goal} (x{self.giveups[self.goal]})")
                    self.tried_goals.add(self.goal)
                    self.goal = None
                    self.goal_blocks = 0
            else:
                pos = self.expected      # arrived (allow 1 px sprite jitter)
                pushed = False
                if ec is not None and ec != bg and before is not None and self.expected_cell and self.expected_dir:
                    by, bx = self.expected_cell[0] + self.expected_dir[0], self.expected_cell[1] + self.expected_dir[1]
                    if 0 <= by < lg.shape[0] and 0 <= bx < lg.shape[1]:
                        lgb = self._logical(before)
                        if int(lgb[by, bx]) != ec and int(lg[by, bx]) == ec:
                            pushed = True
                            if ec not in self.box_colours:
                                self.box_colours.add(ec)
                                self._log(f"colour {ec} is a pushable box")
                if ec is not None and ec != bg and not pushed and ec not in self.pass_colours:
                    self.pass_colours.add(ec)
                    self._log(f"colour {ec} is passable")
            self.expected = None
        self.avatar_pos = pos
        bb = getattr(self, "avatar_bbox", None)
        if bb is not None and abs(bb[0] - pos[0]) <= 1 and abs(bb[1] - pos[1]) <= 1:
            here = self._cell_of((bb[0] + bb[2]) // 2, (bb[1] + bb[3]) // 2)
        else:
            here = self._cell_of(*pos)
        big = {int(c) for c, n in enumerate(np.bincount(lg.ravel(), minlength=16)) if n > 0.4 * lg.size}
        # HUD bands: a single non-floor colour filling most of a logical row/column
        hud = set()
        n = lg.shape[0]
        for i in range(n):
            for line, cells in ((lg[i, :], [(i, j) for j in range(n)]), (lg[:, i], [(j, i) for j in range(n)])):
                cnts = np.bincount(line, minlength=16)
                c = int(cnts.argmax())
                frac = 0.4 if (i <= 1 or i >= n - 2) else 0.75     # outer rows/cols: even a partial bar is a HUD
                if c != bg and cnts[c] >= frac * n:
                    hud.update(cc for cc in cells if int(lg[cc]) == c)

        # arrived at goal with nothing happening → optional interact, then next goal
        if self.goal is not None and here == self.goal and not self.plan:
            if self.hyp != "push":
                self.tried_goals.add(self.goal)
            for g, n in list(self.giveups.items()):
                if n < 2 and g != self.goal:
                    self.tried_goals.discard(g)
            if self.goal_is_outline:
                self.parked.add(self.goal)
                self.goal_is_outline = False
                self.goal = None
                self.hyp = "select"
                self._log(f"parked object in outline {here}; selecting another")
            self.goal = None
            if 5 in latest.available_actions and not self.pending_interact:
                self.pending_interact = True
                self.nav_actions += 1
                self.fire_pending, self.fire_target = self.fire_target, None
                # ACTION5 may hand control to another object: verify with one key press afterwards
                self.reprobe_pending, self.reprobe_key_sent = True, False
                self.plan, self.expected = [], None
                act = GameAction.ACTION5
                act.reasoning = {"mode": "interact"}
                return act
            self.pending_interact = False

        if not self.plan:
            # candidate goals: non-background, non-avatar cells, rarest colours first, nearest first
            counts = np.bincount(lg.ravel(), minlength=16)
            cands = []
            # cells covered by the avatar itself (bbox in logical cells, +1 margin for its own outline)
            ay0, ax0 = self._cell_of(*pos)
            span = max(1, int(round((self.avatar_size ** 0.5) / self.cell)))
            own = {(y, x) for y in range(ay0 - 1, ay0 + span + 1) for x in range(ax0 - 1, ax0 + span + 1)}
            outline = self._outline_cells(grid, bg)
            peer = set()
            for c in self._components(grid, self.avatar_colours):
                if c["size"] >= 0.5 * self.avatar_size:
                    for (py, px) in c["cells"]:
                        peer.add(self._cell_of(py, px))
            for cy in range(lg.shape[0]):
                for cx in range(lg.shape[1]):
                    c = int(lg[cy, cx])
                    if (cy, cx) in outline:
                        c = outline[(cy, cx)]
                    if c == bg or c in big or (cy, cx) in own or (cy, cx) in hud or (cy, cx) in peer or (cy, cx) in self.parked:
                        continue
                    if c not in self.box_colours:
                        self.seen_targets[(cy, cx)] = c
                    if (cy, cx) in self.tried_goals or (cy, cx) in self.walls:
                        continue
                    if self.USE_COLOUR_WALLS and self.wall_colours.get(c, 0) >= 3:
                        continue                              # a colour that blocks us repeatedly is not a goal either
                    if c in self.box_colours:
                        continue                              # boxes are things to push, not places to go
                    if self._volatile()[cy * self.cell + self.cell // 2, cx * self.cell + self.cell // 2]:
                        continue
                    dist = abs(cy - here[0]) + abs(cx - here[1])
                    prio = 0 if c in self.avatar_colours else 1
                    cands.append((prio, counts[c], dist, (cy, cx)))
            cands.sort()
            if self.hyp == "reach" and not self.plan:
                pass
            if self.hyp == "reach":
                for _, _, _, g in cands[:40]:
                    path = self._plan_to(lg, here, g, bg)
                    if path and len(path) > 1:
                        self.goal = g
                        self.pending_interact = False
                        self.goal_is_outline = g in outline
                        self.goal_colour_at[g] = int(lg[g])
                        self.goal_blocks = 0
                        self.plan = path[1:]
                        break
                if not self.plan:
                    self.hyp = "select"
                    self._log("hypothesis reach exhausted -> select")
            if not self.plan and self.hyp == "select":
                # click another block of the avatar's colours and see whether control moves to it
                act = None
                if 6 in latest.available_actions:
                    others = [c for c in self._components(grid, self.avatar_colours)
                              if (c["bbox"][0], c["bbox"][1]) != tuple(pos) and (c["bbox"][0], c["bbox"][1]) not in self.selected
                              and 0.3 * self.avatar_size <= c["size"] <= 6 * self.avatar_size
                              and self._cell_of(c["bbox"][0], c["bbox"][1]) not in self.parked]
                    if others:
                        c = min(others, key=lambda c: abs(c["bbox"][0] - pos[0]) + abs(c["bbox"][1] - pos[1]))
                        self.selected.add((c["bbox"][0], c["bbox"][1]))
                        y, x = c["cells"][c["size"] // 2]
                        act = GameAction.ACTION6
                        act.set_data({"x": int(x), "y": int(y)})
                        act.reasoning = {"mode": "select", "click": [int(x), int(y)]}
                        self.reprobe_pending = True          # next step: press a key, see what moves
                        self.avatar_pos = (c["bbox"][0], c["bbox"][1])
                        self.plan, self.expected = [], None
                        self.nav_actions += 1
                        self._log(f"select: clicking block at ({x},{y})")
                if act is not None:
                    self.hyp = "reach"
                    return act
                self.hyp = "switch"
                self._log("hypothesis select exhausted -> switch")
            if not self.plan and self.hyp == "switch":
                if 5 in latest.available_actions and self.switches < 6:
                    self.switches += 1
                    self.reprobe_pending = True
                    self.plan, self.expected = [], None
                    self.nav_actions += 1
                    self.hyp = "reach"
                    self._log("switch: pressing ACTION5 to change the controlled object")
                    act = GameAction.ACTION5
                    act.reasoning = {"mode": "switch"}
                    return act
                self.hyp = "fire" if 5 in latest.available_actions else ("push" if self.box_colours else "explore")
                self._log(f"hypothesis switch exhausted -> {self.hyp}")
            if not self.plan and self.hyp == "fire":
                # for each target, stand on the nearest floor cell in its column (then row) and press ACTION5
                best = None
                axes = [self.fire_axis] + [a for a in ("col", "row") if a != self.fire_axis] if self.fire_axis else ["col", "row"]
                for axis in axes:
                    for t in sorted(self.seen_targets):
                        if (t, axis) in self.tried_fire:
                            continue
                        cells = ([(cy, t[1]) for cy in range(lg.shape[0])] if axis == "col"
                                 else [(t[0], cx) for cx in range(lg.shape[1])])
                        cells = [c for c in cells if int(lg[c]) == bg and c not in self.walls]
                        if not cells:
                            self.tried_fire.add((t, axis))
                            continue
                        c = min(cells, key=lambda c: abs(c[0] - here[0]) + abs(c[1] - here[1]))
                        d = abs(c[0] - here[0]) + abs(c[1] - here[1])
                        if best is None or d < best[0]:
                            best = (d, c, t, axis)
                    if best is not None:
                        break
                if best is not None:
                    _, c, t, axis = best
                    self.tried_fire.add((t, axis))
                    path = self._plan_to(lg, here, c, bg)
                    if path is not None:
                        self.goal = c
                        self.goal_blocks = 0
                        self.plan = path[1:]
                        self.pending_interact = False
                        self.fire_target = (t, axis)
                        if not self.plan:                       # already aligned: fire now
                            self.nav_actions += 1
                            self.goal = None
                            self.fire_pending, self.fire_target = self.fire_target, None
                            act = GameAction.ACTION5
                            act.reasoning = {"mode": "fire"}
                            return act
                if not self.plan:
                    self.hyp = "push" if self.box_colours else "explore"
                    self._log(f"hypothesis fire exhausted -> {self.hyp}")
            if not self.plan and self.hyp == "push":
                plan = self._push_plan(lg, here, bg, sorted(self.seen_targets))
                if plan:
                    self.goal = plan[-1]
                    self.goal_blocks = 0
                    self.plan = plan
                else:
                    self.hyp = "explore"
                    self._log("hypothesis push exhausted -> explore")
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
        self.expected_cell, self.expected_dir = nxt, d
        self.nav_actions += 1
        act = GameAction.from_id(a)
        act.reasoning = {"mode": "navigate", "to": list(nxt), "goal": list(self.goal)}
        return act
