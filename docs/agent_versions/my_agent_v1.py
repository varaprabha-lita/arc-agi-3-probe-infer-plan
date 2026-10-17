"""ARC-AGI-3 agent v1: diff-aware explorer.

Idea: the agent knows nothing about the game, so it treats every
(board, action) pair as an experiment. It remembers whether each action
changed the board from each state, prefers untried actions, avoids
actions that repeatedly do nothing, and aims clicks at small,
rare-coloured objects rather than random pixels.

Contract (enforced by the ARC-AGI-3-Agents framework):
  - Subclass `agents.agent.Agent`, class named `MyAgent`.
  - Implement `is_done(frames, latest_frame) -> bool`.
  - Implement `choose_action(frames, latest_frame) -> GameAction`.
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


class MyAgent(Agent):
    """Diff-aware explorer with object-targeted clicks."""

    MAX_ACTIONS = 400        # per-game cap; framework also enforces global limits
    MAX_CLICK_TARGETS = 12   # object centroids considered per state
    EPSILON = 0.08           # small chance of a random legal action
    DEAD_AFTER = 2           # tries with no effect before an action is "dead" here

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        seed = int(time.time() * 1_000_000) + hash(self.game_id) % 1_000_000
        self.rng = random.Random(seed)
        # state_hash -> {action_key: [tries, times_it_changed_board]}
        self.stats: dict[str, dict[tuple, list[int]]] = defaultdict(dict)
        self.last_hash: str | None = None
        self.last_key: tuple | None = None
        self.last_levels = 0
        # click signatures (colour, size) that never did anything
        self.dead_sig_tries: dict[tuple, list[int]] = defaultdict(lambda: [0, 0])
        self.last_click_sig: tuple | None = None

    @property
    def name(self) -> str:
        return f"{super().name}.explorer_v1"

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _grid(frame: FrameData) -> np.ndarray | None:
        if not frame.frame:
            return None
        return np.asarray(frame.frame[-1], dtype=np.int16)

    @staticmethod
    def _hash(grid: np.ndarray) -> str:
        return hashlib.md5(grid.tobytes()).hexdigest()

    def _click_targets(self, grid: np.ndarray) -> list[tuple[int, int, tuple]]:
        """Return up to MAX_CLICK_TARGETS (x, y, signature) for small objects.

        Objects = 4-connected same-colour components. Ranked by rarity of the
        colour and smallness of the object (button-like things first).
        """
        h, w = grid.shape
        seen = np.zeros_like(grid, dtype=bool)
        colour_counts = np.bincount(grid.ravel(), minlength=16)
        bg = int(colour_counts.argmax())
        comps: list[tuple[float, int, int, tuple]] = []

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
                if size > 400:          # huge regions are backgrounds/frames, not buttons
                    continue
                cy = int(round(sum(p[0] for p in cells) / size))
                cx = int(round(sum(p[1] for p in cells) / size))
                # make sure the centroid is actually on the object
                if grid[cy, cx] != c:
                    cy, cx = cells[size // 2]
                rarity = 1.0 / (1.0 + colour_counts[c])
                score = rarity * 1000.0 / (1.0 + math.log1p(size))
                sig = (c, min(size, 64) // 4)   # coarse size bucket
                comps.append((score, cx, cy, sig))

        comps.sort(reverse=True)
        return [(x, y, sig) for _, x, y, sig in comps[: self.MAX_CLICK_TARGETS]]

    def _score(self, state_hash: str, key: tuple, sig: tuple | None) -> float:
        st = self.stats[state_hash].get(key)
        if st is None:
            base = 2.0                                    # untried here: try it
        else:
            tries, changed = st
            if changed == 0 and tries >= self.DEAD_AFTER:
                base = 0.05                               # dead in this state
            else:
                base = changed / tries + 1.0 / math.sqrt(tries)
        if sig is not None:
            t, ch = self.dead_sig_tries[sig]
            if t >= 4 and ch == 0:
                base *= 0.1                               # this kind of object never reacts
        return base + self.rng.random() * 1e-3            # tie-break

    # ---------------------------------------------------------------- contract
    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self.last_hash = None
            self.last_key = None
            return GameAction.RESET

        grid = self._grid(latest_frame)
        if grid is None:
            return GameAction.RESET
        h = self._hash(grid)

        # -- record outcome of the previous experiment
        level_up = latest_frame.levels_completed > self.last_levels
        if self.last_hash is not None and self.last_key is not None:
            changed = (h != self.last_hash) or level_up
            st = self.stats[self.last_hash].setdefault(self.last_key, [0, 0])
            st[0] += 1
            st[1] += int(changed)
            if self.last_click_sig is not None:
                ds = self.dead_sig_tries[self.last_click_sig]
                ds[0] += 1
                ds[1] += int(changed)
        self.last_levels = latest_frame.levels_completed

        # -- legal actions
        legal = [a for a in latest_frame.available_actions if a != 0]
        if not legal:
            legal = [1, 2, 3, 4, 5, 6, 7]

        # -- build candidate list: (score, action_int, x, y, sig)
        candidates: list[tuple[float, int, int | None, int | None, tuple | None]] = []
        for a in legal:
            if a == 6:
                for x, y, sig in self._click_targets(grid):
                    key = ("C", x, y)
                    candidates.append((self._score(h, key, sig), 6, x, y, sig))
                # one random fallback click so clicks are never fully dead
                rx, ry = self.rng.randrange(64), self.rng.randrange(64)
                candidates.append((0.5, 6, rx, ry, None))
            else:
                key = ("A", a)
                candidates.append((self._score(h, key, None), a, None, None, None))

        if self.rng.random() < self.EPSILON:
            choice = self.rng.choice(candidates)
        else:
            choice = max(candidates, key=lambda c: c[0])
        _, a, x, y, sig = choice

        action = GameAction.from_id(a)
        if action.is_complex():
            action.set_data({"x": int(x), "y": int(y)})
            action.reasoning = {"why": "object click", "x": int(x), "y": int(y)}
            self.last_key = ("C", int(x), int(y))
            self.last_click_sig = sig
        else:
            action.reasoning = f"explore action {a}"
            self.last_key = ("A", a)
            self.last_click_sig = None
        self.last_hash = h
        return action
