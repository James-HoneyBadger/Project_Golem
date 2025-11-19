#!/usr/bin/env python3
"""
Cellular Automaton GUI
Supports multiple modes:
- Conway's Game of Life
- High Life (B36/S23)
- Immigration Game (multi-color)
- Rainbow Game (6 colors)
- Langton's Ant
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import numpy as np
from scipy import signal
from abc import ABC, abstractmethod
import json
import os
import time
from collections import deque

# Optional Pillow for export features
try:
    from PIL import Image, ImageDraw

    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False


class CellularAutomaton(ABC):
    """Base class for cellular automaton implementations"""

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.reset()

    @abstractmethod
    def reset(self):
        """Reset the automaton to initial state"""
        pass

    @abstractmethod
    def step(self):
        """Perform one step of the simulation"""
        pass

    @abstractmethod
    def get_grid(self):
        """Return the current grid state for rendering"""
        pass

    @abstractmethod
    def handle_click(self, x, y):
        """Handle mouse click at grid position (x, y)"""
        pass


class ConwayGameOfLife(CellularAutomaton):
    """Conway's Game of Life implementation"""

    def reset(self):
        self.grid = np.zeros((self.height, self.width), dtype=int)

    def load_pattern(self, pattern_name):
        """Load a predefined pattern onto the grid"""
        self.grid = np.zeros((self.height, self.width), dtype=int)
        center_x = self.width // 2
        center_y = self.height // 2

        if pattern_name == "Classic Mix":
            self._add_classic_mix(center_x, center_y)
        elif pattern_name == "Glider Gun":
            self._add_glider_gun(center_x, center_y)
        elif pattern_name == "Puffers":
            self._add_puffers(center_x, center_y)
        elif pattern_name == "Oscillators":
            self._add_oscillators(center_x, center_y)
        elif pattern_name == "Spaceships":
            self._add_spaceships(center_x, center_y)
        elif pattern_name == "Random Soup":
            self._add_random_soup()
        elif pattern_name == "R-Pentomino":
            self._add_r_pentomino(center_x, center_y)
        elif pattern_name == "Acorn":
            self._add_acorn(center_x, center_y)

    def _add_classic_mix(self, center_x, center_y):
        """Add interesting default patterns to the grid"""
        # Glider (top-left area)
        glider = [(1, 0), (2, 1), (0, 2), (1, 2), (2, 2)]
        for dx, dy in glider:
            x, y = center_x - 30 + dx, center_y - 25 + dy
            if 0 <= x < self.width and 0 <= y < self.height:
                self.grid[y, x] = 1

        # Blinker (top area)
        blinker = [(0, 0), (1, 0), (2, 0)]
        for dx, dy in blinker:
            x, y = center_x + dx - 1, center_y - 30 + dy
            if 0 <= x < self.width and 0 <= y < self.height:
                self.grid[y, x] = 1

        # Toad (center-left)
        toad = [(1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1)]
        for dx, dy in toad:
            x, y = center_x - 25 + dx, center_y + dy - 1
            if 0 <= x < self.width and 0 <= y < self.height:
                self.grid[y, x] = 1

        # Lightweight spaceship (LWSS)
        lwss = [
            (1, 0),
            (4, 0),
            (0, 1),
            (0, 2),
            (4, 2),
            (0, 3),
            (1, 3),
            (2, 3),
            (3, 3),
        ]
        for dx, dy in lwss:
            x, y = center_x + 15 + dx, center_y - 15 + dy
            if 0 <= x < self.width and 0 <= y < self.height:
                self.grid[y, x] = 1

        # Block (stable pattern)
        block = [(0, 0), (1, 0), (0, 1), (1, 1)]
        for dx, dy in block:
            x, y = center_x - 30 + dx, center_y + 20 + dy
            if 0 <= x < self.width and 0 <= y < self.height:
                self.grid[y, x] = 1

        # Beehive (stable pattern)
        beehive = [(1, 0), (2, 0), (0, 1), (3, 1), (1, 2), (2, 2)]
        for dx, dy in beehive:
            x, y = center_x + 20 + dx, center_y + 20 + dy
            if 0 <= x < self.width and 0 <= y < self.height:
                self.grid[y, x] = 1

    def _add_glider_gun(self, center_x, center_y):
        """Gosper's Glider Gun - produces gliders indefinitely"""
        gun = [
            (0, 4),
            (0, 5),
            (1, 4),
            (1, 5),
            (10, 4),
            (10, 5),
            (10, 6),
            (11, 3),
            (11, 7),
            (12, 2),
            (12, 8),
            (13, 2),
            (13, 8),
            (14, 5),
            (15, 3),
            (15, 7),
            (16, 4),
            (16, 5),
            (16, 6),
            (17, 5),
            (20, 2),
            (20, 3),
            (20, 4),
            (21, 2),
            (21, 3),
            (21, 4),
            (22, 1),
            (22, 5),
            (24, 0),
            (24, 1),
            (24, 5),
            (24, 6),
            (34, 2),
            (34, 3),
            (35, 2),
            (35, 3),
        ]
        for dx, dy in gun:
            x, y = center_x - 18 + dx, center_y - 5 + dy
            if 0 <= x < self.width and 0 <= y < self.height:
                self.grid[y, x] = 1

    def _add_puffers(self, center_x, center_y):
        """Add puffer trains that leave debris"""
        # Puffer train
        puffer = [
            (0, 0),
            (2, 0),
            (3, 1),
            (3, 2),
            (0, 3),
            (3, 3),
            (1, 4),
            (2, 4),
            (3, 4),
            (6, 1),
            (6, 2),
            (6, 3),
            (7, 0),
            (7, 2),
            (8, 1),
            (8, 2),
            (11, 0),
            (13, 0),
            (14, 1),
            (14, 2),
            (11, 3),
            (14, 3),
            (12, 4),
            (13, 4),
            (14, 4),
        ]
        for dx, dy in puffer:
            x, y = center_x - 7 + dx, center_y - 2 + dy
            if 0 <= x < self.width and 0 <= y < self.height:
                self.grid[y, x] = 1

    def _add_oscillators(self, center_x, center_y):
        """Collection of various oscillators"""
        # Blinker (period 2)
        blinker = [(0, 0), (1, 0), (2, 0)]
        for dx, dy in blinker:
            x, y = center_x - 30 + dx, center_y - 20 + dy
            if 0 <= x < self.width and 0 <= y < self.height:
                self.grid[y, x] = 1

        # Toad (period 2)
        toad = [(1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1)]
        for dx, dy in toad:
            x, y = center_x - 20 + dx, center_y - 20 + dy
            if 0 <= x < self.width and 0 <= y < self.height:
                self.grid[y, x] = 1

        # Beacon (period 2)
        beacon = [(0, 0), (1, 0), (0, 1), (3, 2), (2, 3), (3, 3)]
        for dx, dy in beacon:
            x, y = center_x - 5 + dx, center_y - 20 + dy
            if 0 <= x < self.width and 0 <= y < self.height:
                self.grid[y, x] = 1

        # Pulsar (period 3)
        pulsar_pattern = [
            (2, 0),
            (3, 0),
            (4, 0),
            (8, 0),
            (9, 0),
            (10, 0),
            (0, 2),
            (5, 2),
            (7, 2),
            (12, 2),
            (0, 3),
            (5, 3),
            (7, 3),
            (12, 3),
            (0, 4),
            (5, 4),
            (7, 4),
            (12, 4),
            (2, 5),
            (3, 5),
            (4, 5),
            (8, 5),
            (9, 5),
            (10, 5),
            (2, 7),
            (3, 7),
            (4, 7),
            (8, 7),
            (9, 7),
            (10, 7),
            (0, 8),
            (5, 8),
            (7, 8),
            (12, 8),
            (0, 9),
            (5, 9),
            (7, 9),
            (12, 9),
            (0, 10),
            (5, 10),
            (7, 10),
            (12, 10),
            (2, 12),
            (3, 12),
            (4, 12),
            (8, 12),
            (9, 12),
            (10, 12),
        ]
        for dx, dy in pulsar_pattern:
            x, y = center_x - 6 + dx, center_y + dy - 1
            if 0 <= x < self.width and 0 <= y < self.height:
                self.grid[y, x] = 1

    def _add_spaceships(self, center_x, center_y):
        """Collection of various spaceships"""
        # Glider
        glider = [(1, 0), (2, 1), (0, 2), (1, 2), (2, 2)]
        for dx, dy in glider:
            x, y = center_x - 30 + dx, center_y - 20 + dy
            if 0 <= x < self.width and 0 <= y < self.height:
                self.grid[y, x] = 1

        # LWSS
        lwss = [
            (1, 0),
            (4, 0),
            (0, 1),
            (0, 2),
            (4, 2),
            (0, 3),
            (1, 3),
            (2, 3),
            (3, 3),
        ]
        for dx, dy in lwss:
            x, y = center_x - 15 + dx, center_y - 20 + dy
            if 0 <= x < self.width and 0 <= y < self.height:
                self.grid[y, x] = 1

        # MWSS (Middleweight Spaceship)
        mwss = [
            (2, 0),
            (0, 1),
            (4, 1),
            (0, 2),
            (0, 3),
            (4, 3),
            (0, 4),
            (1, 4),
            (2, 4),
            (3, 4),
            (4, 4),
        ]
        for dx, dy in mwss:
            x, y = center_x + dx, center_y - 20 + dy
            if 0 <= x < self.width and 0 <= y < self.height:
                self.grid[y, x] = 1

        # HWSS (Heavyweight Spaceship)
        hwss = [
            (2, 0),
            (3, 0),
            (0, 1),
            (5, 1),
            (0, 2),
            (0, 3),
            (5, 3),
            (0, 4),
            (1, 4),
            (2, 4),
            (3, 4),
            (4, 4),
            (5, 4),
        ]
        for dx, dy in hwss:
            x, y = center_x + 15 + dx, center_y - 20 + dy
            if 0 <= x < self.width and 0 <= y < self.height:
                self.grid[y, x] = 1

    def _add_random_soup(self):
        """Add random cells for interesting evolution"""
        for i in range(self.height):
            for j in range(self.width):
                if np.random.random() < 0.15:  # 15% chance
                    self.grid[i, j] = 1

    def _add_r_pentomino(self, center_x, center_y):
        """R-pentomino - evolves for 1103 generations"""
        r_pentomino = [(1, 0), (2, 0), (0, 1), (1, 1), (1, 2)]
        for dx, dy in r_pentomino:
            x, y = center_x + dx - 1, center_y + dy - 1
            if 0 <= x < self.width and 0 <= y < self.height:
                self.grid[y, x] = 1

    def _add_acorn(self, center_x, center_y):
        """Acorn pattern - takes 5206 generations to stabilize"""
        acorn = [(1, 0), (3, 1), (0, 2), (1, 2), (4, 2), (5, 2), (6, 2)]
        for dx, dy in acorn:
            x, y = center_x + dx - 3, center_y + dy - 1
            if 0 <= x < self.width and 0 <= y < self.height:
                self.grid[y, x] = 1

    def step(self):
        """Optimized step using convolution for neighbor counting"""
        # Convolution kernel for counting neighbors
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])

        # Count neighbors using convolution with wrap-around boundaries
        neighbors = signal.convolve2d(
            self.grid,
            kernel,
            mode="same",
            boundary="wrap",
        )

        # Apply Conway's rules vectorized
        # Birth: dead cell with 3 neighbors becomes alive
        # Survival: live cell with 2 or 3 neighbors stays alive
        self.grid = ((self.grid == 1) & ((neighbors == 2) | (neighbors == 3))) | (
            (self.grid == 0) & (neighbors == 3)
        )
        self.grid = self.grid.astype(int)

    def get_grid(self):
        return self.grid

    def handle_click(self, x, y):
        """Toggle cell state"""
        self.grid[y, x] = 1 - self.grid[y, x]


class LangtonsAnt(CellularAutomaton):
    """Langton's Ant implementation"""

    def reset(self):
        self.grid = np.zeros((self.height, self.width), dtype=int)
        # Start ant in the middle, facing up
        self.ant_x = self.width // 2
        self.ant_y = self.height // 2
        self.ant_dir = 0  # 0=North, 1=East, 2=South, 3=West

    def step(self):
        # Current cell color
        current_color = self.grid[self.ant_y, self.ant_x]

        # Flip the color of the current square
        self.grid[self.ant_y, self.ant_x] = 1 - current_color

        # Turn: if white (0), turn right; if black (1), turn left
        if current_color == 0:
            self.ant_dir = (self.ant_dir + 1) % 4  # Turn right
        else:
            self.ant_dir = (self.ant_dir - 1) % 4  # Turn left

        # Move forward
        if self.ant_dir == 0:  # North
            self.ant_y = (self.ant_y - 1) % self.height
        elif self.ant_dir == 1:  # East
            self.ant_x = (self.ant_x + 1) % self.width
        elif self.ant_dir == 2:  # South
            self.ant_y = (self.ant_y + 1) % self.height
        else:  # West
            self.ant_x = (self.ant_x - 1) % self.width

    def get_grid(self):
        # Create a copy with the ant position highlighted
        display_grid = self.grid.copy()
        display_grid[self.ant_y, self.ant_x] = 2  # Special value for ant
        return display_grid

    def handle_click(self, x, y):
        """Move ant to clicked position"""
        self.ant_x = x
        self.ant_y = y


class HighLife(CellularAutomaton):
    """High Life - B36/S23 (replicators possible)"""

    def reset(self):
        self.grid = np.zeros((self.height, self.width), dtype=int)

    def load_pattern(self, pattern_name):
        """Load a predefined pattern onto the grid"""
        self.grid = np.zeros((self.height, self.width), dtype=int)
        center_x = self.width // 2
        center_y = self.height // 2

        if pattern_name == "Replicator":
            self._add_replicator(center_x, center_y)
        elif pattern_name == "Random Soup":
            self._add_random_soup()

    def _add_replicator(self, center_x, center_y):
        """Add a replicator pattern unique to HighLife"""
        replicator = [(1, 0), (0, 1), (1, 1), (2, 1), (0, 2), (2, 2), (1, 3)]
        for dx, dy in replicator:
            x, y = center_x + dx - 1, center_y + dy - 1
            if 0 <= x < self.width and 0 <= y < self.height:
                self.grid[y, x] = 1

    def _add_random_soup(self):
        """Add random cells for interesting evolution"""
        for i in range(self.height):
            for j in range(self.width):
                if np.random.random() < 0.15:
                    self.grid[i, j] = 1

    def step(self):
        """Optimized step using convolution for neighbor counting"""
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])

        neighbors = signal.convolve2d(
            self.grid,
            kernel,
            mode="same",
            boundary="wrap",
        )

        # HighLife rules: B36/S23
        # Birth on 3 or 6 neighbors, Survival on 2 or 3
        birth = (self.grid == 0) & ((neighbors == 3) | (neighbors == 6))
        survival = (self.grid == 1) & ((neighbors == 2) | (neighbors == 3))
        self.grid = (birth | survival).astype(int)

    def get_grid(self):
        return self.grid

    def handle_click(self, x, y):
        """Toggle cell state"""
        self.grid[y, x] = 1 - self.grid[y, x]


def parse_bs(rule_str):
    """Parse B/S rule notation like 'B3/S23' -> (birth_set, survival_set).

    Returns two Python sets of neighbor counts.
    """
    rule = rule_str.upper().replace(" ", "")
    b_part, s_part = set(), set()
    if "B" in rule:
        try:
            b_idx = rule.index("B")
            # Find end of B section
            s_idx = rule.index("S") if "S" in rule else len(rule)
            b_digits = "".join(ch for ch in rule[b_idx + 1 : s_idx] if ch.isdigit())
            b_part = {int(ch) for ch in b_digits}
        except Exception:
            b_part = set()
    if "S" in rule:
        try:
            s_idx = rule.index("S")
            # Find end of S section
            end = len(rule)
            s_digits = "".join(ch for ch in rule[s_idx + 1 : end] if ch.isdigit())
            s_part = {int(ch) for ch in s_digits}
        except Exception:
            s_part = set()
    return b_part, s_part


class LifeLikeAutomaton(CellularAutomaton):
    """Generic life-like CA with B/S rules."""

    def __init__(self, width, height, birth=None, survival=None):
        self.birth = set(birth or {3})
        self.survival = set(survival or {2, 3})
        super().__init__(width, height)

    def set_rules(self, birth, survival):
        self.birth = set(birth)
        self.survival = set(survival)

    def reset(self):
        self.grid = np.zeros((self.height, self.width), dtype=int)

    def load_pattern(self, pattern_name):
        self.grid = np.zeros((self.height, self.width), dtype=int)
        if pattern_name == "Random Soup":
            for i in range(self.height):
                for j in range(self.width):
                    if np.random.random() < 0.15:
                        self.grid[i, j] = 1

    def step(self):
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        neighbors = signal.convolve2d(
            self.grid,
            kernel,
            mode="same",
            boundary="wrap",
        )
        # Compute birth/survival with any-of logic
        birth_any = (
            np.logical_or.reduce([(neighbors == n) for n in self.birth])
            if self.birth
            else False
        )
        surv_any = (
            np.logical_or.reduce([(neighbors == n) for n in self.survival])
            if self.survival
            else False
        )
        birth = (self.grid == 0) & birth_any
        survival = (self.grid == 1) & surv_any
        self.grid = (birth | survival).astype(int)

    def get_grid(self):
        return self.grid

    def handle_click(self, x, y):
        self.grid[y, x] = 1 - self.grid[y, x]


class ImmigrationGame(CellularAutomaton):
    """Immigration Game - Multi-state automaton with 4 colors"""

    def reset(self):
        self.grid = np.zeros((self.height, self.width), dtype=int)

    def load_pattern(self, pattern_name):
        """Load a predefined pattern onto the grid"""
        self.grid = np.zeros((self.height, self.width), dtype=int)
        center_x = self.width // 2
        center_y = self.height // 2

        if pattern_name == "Color Mix":
            self._add_color_mix(center_x, center_y)
        elif pattern_name == "Random Soup":
            self._add_random_soup()

    def _add_color_mix(self, center_x, center_y):
        """Add patterns with different colors"""
        # State 1 glider
        glider1 = [(1, 0), (2, 1), (0, 2), (1, 2), (2, 2)]
        for dx, dy in glider1:
            x, y = center_x - 20 + dx, center_y - 15 + dy
            if 0 <= x < self.width and 0 <= y < self.height:
                self.grid[y, x] = 1

        # State 2 blinker
        blinker2 = [(0, 0), (1, 0), (2, 0)]
        for dx, dy in blinker2:
            x, y = center_x + dx - 1, center_y - 15 + dy
            if 0 <= x < self.width and 0 <= y < self.height:
                self.grid[y, x] = 2

        # State 3 block
        block3 = [(0, 0), (1, 0), (0, 1), (1, 1)]
        for dx, dy in block3:
            x, y = center_x + 10 + dx, center_y - 15 + dy
            if 0 <= x < self.width and 0 <= y < self.height:
                self.grid[y, x] = 3

    def _add_random_soup(self):
        """Add random colored cells"""
        for i in range(self.height):
            for j in range(self.width):
                if np.random.random() < 0.15:
                    self.grid[i, j] = np.random.randint(1, 4)

    def step(self):
        new_grid = np.zeros_like(self.grid)

        for i in range(self.height):
            for j in range(self.width):
                # Count neighbors by state
                neighbor_count = 0
                neighbor_states = []

                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = (i + di) % self.height, (j + dj) % self.width
                        if self.grid[ni, nj] > 0:
                            neighbor_count += 1
                            neighbor_states.append(self.grid[ni, nj])

                # Immigration rules: same as Conway's but color is successor
                current_state = self.grid[i, j]

                if current_state > 0:  # Cell is alive
                    if neighbor_count in [2, 3]:
                        new_grid[i, j] = current_state
                else:  # Cell is dead
                    if neighbor_count == 3:
                        # New cell takes the "next" state of its parents
                        if neighbor_states:
                            avg_state = int(np.mean(neighbor_states))
                            new_grid[i, j] = (avg_state % 3) + 1

        self.grid = new_grid

    def get_grid(self):
        return self.grid

    def handle_click(self, x, y):
        """Cycle through cell states"""
        self.grid[y, x] = (self.grid[y, x] + 1) % 4


class RainbowGame(CellularAutomaton):
    """Rainbow Game - Multi-color variant with 6 colors"""

    def reset(self):
        self.grid = np.zeros((self.height, self.width), dtype=int)

    def load_pattern(self, pattern_name):
        """Load a predefined pattern onto the grid"""
        self.grid = np.zeros((self.height, self.width), dtype=int)
        center_x = self.width // 2
        center_y = self.height // 2

        if pattern_name == "Rainbow Mix":
            self._add_rainbow_mix(center_x, center_y)
        elif pattern_name == "Random Soup":
            self._add_random_soup()

    def _add_rainbow_mix(self, center_x, center_y):
        """Add patterns with different rainbow colors"""
        patterns = [
            # Red glider (state 1)
            ([(1, 0), (2, 1), (0, 2), (1, 2), (2, 2)], -30, -20, 1),
            # Orange blinker (state 2)
            ([(0, 0), (1, 0), (2, 0)], -15, -20, 2),
            # Yellow toad (state 3)
            ([(1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1)], 0, -20, 3),
            # Green block (state 4)
            ([(0, 0), (1, 0), (0, 1), (1, 1)], 15, -20, 4),
            # Blue beehive (state 5)
            ([(1, 0), (2, 0), (0, 1), (3, 1), (1, 2), (2, 2)], 25, -20, 5),
            # Purple beacon (state 6)
            ([(0, 0), (1, 0), (0, 1), (3, 2), (2, 3), (3, 3)], -20, 0, 6),
        ]

        for pattern, offset_x, offset_y, state in patterns:
            for dx, dy in pattern:
                x, y = center_x + offset_x + dx, center_y + offset_y + dy
                if 0 <= x < self.width and 0 <= y < self.height:
                    self.grid[y, x] = state

    def _add_random_soup(self):
        """Add random colored cells"""
        for i in range(self.height):
            for j in range(self.width):
                if np.random.random() < 0.15:
                    self.grid[i, j] = np.random.randint(1, 7)

    def step(self):
        new_grid = np.zeros_like(self.grid)

        for i in range(self.height):
            for j in range(self.width):
                # Count neighbors and track their colors
                neighbor_count = 0
                neighbor_colors = []

                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = (i + di) % self.height, (j + dj) % self.width
                        if self.grid[ni, nj] > 0:
                            neighbor_count += 1
                            neighbor_colors.append(self.grid[ni, nj])

                current_state = self.grid[i, j]

                if current_state > 0:  # Cell is alive
                    if neighbor_count in [2, 3]:
                        new_grid[i, j] = current_state
                else:  # Cell is dead
                    if neighbor_count == 3:
                        # New cell is average of neighbor colors
                        if neighbor_colors:
                            new_grid[i, j] = int(np.mean(neighbor_colors))

        self.grid = new_grid

    def get_grid(self):
        return self.grid

    def handle_click(self, x, y):
        """Cycle through rainbow colors"""
        self.grid[y, x] = (self.grid[y, x] + 1) % 7


class CellularAutomatonGUI:
    """Main GUI application for cellular automaton simulation"""

    def __init__(self, root):
        self.root = root
        self.root.title("Cellular Automaton Studio")

        # Modern color scheme
        self.colors = {
            "bg": "#1e1e1e",
            "fg": "#e0e0e0",
            "accent": "#0078d4",
            "accent_hover": "#1984d8",
            "secondary": "#2d2d30",
            "border": "#3e3e42",
            "success": "#4caf50",
            "warning": "#ff9800",
            "danger": "#f44336",
            "canvas_bg": "#252525",
        }

        # Configure window
        self.root.configure(bg=self.colors["bg"])
        self.root.geometry("1400x900")

        # Setup modern ttk style
        self.setup_styles()

        # Configuration
        self.grid_width = 100
        self.grid_height = 80
        self.cell_size = 8
        self.update_delay = 50  # milliseconds

        # State
        self.running = False
        self.current_automaton = None
        self.cell_rects = {}  # Cache of canvas rectangle IDs
        self.prev_grid = None  # Previous grid state for optimization

        # FPS tracking
        self.last_update_time = time.time()
        self.fps = 0
        self.frame_times = deque(maxlen=30)
        self.population_history = deque(maxlen=500)
        self.stats_window = None
        self.stats_canvas = None
        self.stats_canvas_w = 400
        self.stats_canvas_h = 120

        # History/replay system
        self.history = deque(maxlen=1000)
        self.history_position = -1
        self.recording_history = True

        # Drawing tools
        self.draw_mode = "toggle"  # toggle, pen, eraser, line, rect
        self.draw_start = None  # start point for line/rect
        self.last_draw_pos = None  # last point for continuous draw/drag
        self.show_grid_lines = True
        # Symmetry modes: none, horizontal, vertical, both, rotational
        self.symmetry_mode = "none"

        # Grid configuration
        self.zoom_level = 1.0
        self.pan_offset_x = 0
        self.pan_offset_y = 0

    def setup_styles(self):
        """Configure ttk styles for modern appearance"""
        style = ttk.Style()
        style.theme_use("clam")

        # Configure colors
        style.configure(
            ".",
            background=self.colors["secondary"],
            foreground=self.colors["fg"],
            fieldbackground=self.colors["secondary"],
            bordercolor=self.colors["border"],
            darkcolor=self.colors["secondary"],
            lightcolor=self.colors["border"],
            troughcolor=self.colors["bg"],
            selectbackground=self.colors["accent"],
            selectforeground="white",
        )

        # Modern button style
        style.configure(
            "TButton",
            background=self.colors["accent"],
            foreground="white",
            borderwidth=0,
            focuscolor="none",
            padding=(12, 6),
            font=("Segoe UI", 9),
        )
        style.map(
            "TButton",
            background=[
                ("active", self.colors["accent_hover"]),
                ("pressed", "#1878c8"),
            ],
            foreground=[("disabled", "#888888")],
        )

        # Action button (Start/Stop)
        style.configure(
            "Action.TButton",
            background=self.colors["success"],
            foreground="white",
            padding=(16, 8),
            font=("Segoe UI", 10, "bold"),
        )
        style.map(
            "Action.TButton", background=[("active", "#66bb6a"), ("pressed", "#43a047")]
        )

        # Danger button
        style.configure(
            "Danger.TButton", background=self.colors["danger"], foreground="white"
        )
        style.map(
            "Danger.TButton", background=[("active", "#f55a4e"), ("pressed", "#e53935")]
        )

        # Small icon button
        style.configure("Icon.TButton", padding=(8, 4), font=("Segoe UI", 9))

        # Label style
        style.configure(
            "TLabel",
            background=self.colors["secondary"],
            foreground=self.colors["fg"],
            font=("Segoe UI", 9),
        )

        # Title label
        style.configure(
            "Title.TLabel",
            font=("Segoe UI", 11, "bold"),
            foreground=self.colors["accent"],
        )

        # Combobox
        style.configure(
            "TCombobox",
            fieldbackground=self.colors["bg"],
            background=self.colors["secondary"],
            foreground=self.colors["fg"],
            arrowcolor=self.colors["fg"],
            borderwidth=1,
            relief="flat",
        )

        # Entry
        style.configure(
            "TEntry",
            fieldbackground=self.colors["bg"],
            foreground=self.colors["fg"],
            borderwidth=1,
            relief="flat",
        )

        # Frame
        style.configure("TFrame", background=self.colors["secondary"], borderwidth=0)

        # Card frame (raised panels)
        style.configure(
            "Card.TFrame",
            background=self.colors["secondary"],
            relief="flat",
            borderwidth=1,
        )

        # Spinbox
        style.configure(
            "TSpinbox",
            fieldbackground=self.colors["bg"],
            background=self.colors["secondary"],
            foreground=self.colors["fg"],
            arrowcolor=self.colors["fg"],
            borderwidth=1,
        )

        # Create UI
        self.create_widgets()

        # Setup keyboard shortcuts
        self.setup_keyboard_shortcuts()

        # Initialize with Conway's Game of Life
        self.switch_mode("Conway's Game of Life")

    def setup_keyboard_shortcuts(self):
        """Setup keyboard shortcuts for common actions"""
        self.root.bind("<space>", lambda e: self.toggle_simulation())
        self.root.bind("r", lambda e: self.reset_simulation())
        self.root.bind("R", lambda e: self.reset_simulation())
        self.root.bind("c", lambda e: self.clear_grid())
        self.root.bind("C", lambda e: self.clear_grid())
        self.root.bind("s", lambda e: self.step_once())
        self.root.bind("S", lambda e: self.step_once())

        # Number keys for mode selection
        self.root.bind(
            "1",
            lambda e: self._quick_mode("Conway's Game of Life"),
        )
        self.root.bind("2", lambda e: self._quick_mode("High Life"))
        self.root.bind("3", lambda e: self._quick_mode("Immigration Game"))
        self.root.bind("4", lambda e: self._quick_mode("Rainbow Game"))
        self.root.bind("5", lambda e: self._quick_mode("Langton's Ant"))

        # Save/Load shortcuts
        self.root.bind("<Control-s>", lambda e: self.save_pattern())
        self.root.bind("<Control-o>", lambda e: self.load_saved_pattern())

        # History navigation shortcuts
        self.root.bind("<Left>", lambda e: self.rewind_history())
        self.root.bind("<Right>", lambda e: self.forward_history())

        # Grid toggle shortcut
        self.root.bind("g", lambda e: self.toggle_grid_lines())
        self.root.bind("G", lambda e: self.toggle_grid_lines())

    def _quick_mode(self, mode_name):
        """Quick mode switch via keyboard"""
        self.mode_var.set(mode_name)
        self.switch_mode(mode_name)

    def create_widgets(self):
        # Main container with padding
        main_container = ttk.Frame(self.root, style="TFrame")
        main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Left sidebar for controls
        sidebar = ttk.Frame(main_container, style="Card.TFrame", padding=15)
        sidebar.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.W), padx=(0, 10))

        # Title
        title_label = ttk.Label(
            sidebar, text="⚡ Cellular Automaton Studio", style="Title.TLabel"
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 15))

        # Mode selection section
        row = 1
        ttk.Label(sidebar, text="SIMULATION MODE").grid(
            row=row, column=0, columnspan=2, sticky=tk.W, pady=(10, 5)
        )
        row += 1

        self.mode_var = tk.StringVar(value="Conway's Game of Life")
        mode_combo = ttk.Combobox(
            sidebar,
            textvariable=self.mode_var,
            values=[
                "Conway's Game of Life",
                "High Life",
                "Immigration Game",
                "Rainbow Game",
                "Langton's Ant",
                "Custom Rules",
            ],
            state="readonly",
            width=28,
        )
        mode_combo.grid(
            row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 15)
        )
        mode_combo.bind(
            "<<ComboboxSelected>>",
            lambda e: self.switch_mode(self.mode_var.get()),
        )
        row += 1

        # Playback controls
        ttk.Label(sidebar, text="PLAYBACK").grid(
            row=row, column=0, columnspan=2, sticky=tk.W, pady=(10, 5)
        )
        row += 1

        # Start/Stop button (prominent)
        self.start_button = ttk.Button(
            sidebar,
            text="▶ START",
            command=self.toggle_simulation,
            style="Action.TButton",
        )
        self.start_button.grid(
            row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 8)
        )
        row += 1

        # Secondary playback buttons
        playback_frame = ttk.Frame(sidebar)
        playback_frame.grid(
            row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 15)
        )
        row += 1

        ttk.Button(playback_frame, text="Step", command=self.step_once, width=8).grid(
            row=0, column=0, padx=(0, 4)
        )
        ttk.Button(
            playback_frame, text="Reset", command=self.reset_simulation, width=8
        ).grid(row=0, column=1, padx=(0, 4))
        ttk.Button(
            playback_frame,
            text="Clear",
            command=self.clear_grid,
            style="Danger.TButton",
            width=8,
        ).grid(row=0, column=2)

        # History controls
        ttk.Label(sidebar, text="HISTORY").grid(
            row=row, column=0, columnspan=2, sticky=tk.W, pady=(10, 5)
        )
        row += 1

        history_frame = ttk.Frame(sidebar)
        history_frame.grid(
            row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 15)
        )
        row += 1

        ttk.Button(
            history_frame,
            text="◀",
            command=self.rewind_history,
            width=5,
            style="Icon.TButton",
        ).grid(row=0, column=0, padx=(0, 2))
        ttk.Button(
            history_frame,
            text="▶",
            command=self.forward_history,
            width=5,
            style="Icon.TButton",
        ).grid(row=0, column=1, padx=(0, 10))
        ttk.Button(
            history_frame, text="Grid", command=self.toggle_grid_lines, width=6
        ).grid(row=0, column=2, padx=(0, 2))
        ttk.Button(
            history_frame, text="Stats", command=self.open_stats_panel, width=6
        ).grid(row=0, column=3)

        # Pattern selector
        ttk.Label(sidebar, text="PATTERN").grid(
            row=row, column=0, columnspan=2, sticky=tk.W, pady=(10, 5)
        )
        row += 1

        self.pattern_var = tk.StringVar(value="Empty")
        self.pattern_combo = ttk.Combobox(
            sidebar,
            textvariable=self.pattern_var,
            values=[
                "Empty",
                "Classic Mix",
                "Glider Gun",
                "Spaceships",
                "Oscillators",
                "Puffers",
                "R-Pentomino",
                "Acorn",
                "Random Soup",
            ],
            state="readonly",
            width=28,
        )
        self.pattern_combo.grid(
            row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 8)
        )
        row += 1

        self.load_pattern_button = ttk.Button(
            sidebar,
            text="Load Pattern",
            command=self.load_pattern_handler,
        )
        self.load_pattern_button.grid(
            row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 15)
        )
        row += 1

        # File operations section
        ttk.Label(sidebar, text="FILE").grid(
            row=row, column=0, columnspan=2, sticky=tk.W, pady=(10, 5)
        )
        row += 1
        ttk.Button(
            control_frame,
            text="Export GIF",
            command=self.export_gif,
        ).grid(row=1, column=6, padx=5, pady=5)

        # History controls
        ttk.Button(
            control_frame,
            text="◀",
            command=self.rewind_history,
            width=3,
        ).grid(row=0, column=8, padx=2)
        ttk.Button(
            control_frame,
            text="▶",
            command=self.forward_history,
            width=3,
        ).grid(row=0, column=9, padx=2)

        # Grid lines toggle & Stats button
        ttk.Button(
            control_frame, text="Grid", command=self.toggle_grid_lines, width=4
        ).grid(row=0, column=10, padx=2)
        ttk.Button(
            control_frame, text="Stats", command=self.open_stats_panel, width=5
        ).grid(row=0, column=11, padx=2)

        # Pattern selector (for Conway's Game of Life)
        ttk.Label(control_frame, text="Pattern:").grid(row=1, column=0, padx=5, pady=5)
        self.pattern_var = tk.StringVar(value="Empty")
        self.pattern_combo = ttk.Combobox(
            control_frame,
            textvariable=self.pattern_var,
            values=[
                "Empty",
                "Classic Mix",
                "Glider Gun",
                "Spaceships",
                "Oscillators",
                "Puffers",
                "R-Pentomino",
                "Acorn",
                "Random Soup",
            ],
            state="readonly",
            width=20,
        )
        self.pattern_combo.grid(row=1, column=1, padx=5, pady=5)

        self.load_pattern_button = ttk.Button(
            control_frame,
            text="Load Pattern",
            command=self.load_pattern_handler,
        )
        self.load_pattern_button.grid(row=1, column=2, padx=5, pady=5)

        # File operations
        ttk.Button(control_frame, text="Save", command=self.save_pattern).grid(
            row=1, column=3, padx=5, pady=5
        )
        ttk.Button(
            control_frame, text="Load File", command=self.load_saved_pattern
        ).grid(row=1, column=4, padx=5, pady=5)

        # Custom rules controls (row 4)
        ttk.Label(control_frame, text="Rules (B/S):").grid(
            row=4, column=0, padx=5, pady=5
        )
        self.rules_var = tk.StringVar(value="B3/S23")
        self.rules_entry = ttk.Entry(
            control_frame,
            textvariable=self.rules_var,
            width=12,
        )
        self.rules_entry.grid(row=4, column=1, padx=5, pady=5, sticky=tk.W)
        ttk.Button(
            control_frame,
            text="Apply Rules",
            command=self.apply_custom_rules,
        ).grid(row=4, column=2, padx=5, pady=5)

        # Grid size controls (row 2)
        ttk.Label(control_frame, text="Grid Size:").grid(
            row=2, column=0, padx=5, pady=5
        )

        self.size_preset_var = tk.StringVar(value="Medium")
        size_presets = [
            "Small (50×40)",
            "Medium (100×80)",
            "Large (200×160)",
            "Custom",
        ]
        self.size_combo = ttk.Combobox(
            control_frame,
            textvariable=self.size_preset_var,
            values=size_presets,
            state="readonly",
            width=18,
        )
        self.size_combo.grid(row=2, column=1, padx=5, pady=5)
        self.size_combo.bind(
            "<<ComboboxSelected>>",
            self.apply_grid_size_preset,
        )

        # Custom size inputs
        ttk.Label(control_frame, text="W:").grid(row=2, column=2, sticky=tk.E)
        self.custom_width_var = tk.IntVar(value=self.grid_width)
        self.width_spinbox = ttk.Spinbox(
            control_frame,
            from_=10,
            to=500,
            textvariable=self.custom_width_var,
            width=6,
        )
        self.width_spinbox.grid(row=2, column=3, padx=(0, 5), pady=5, sticky=tk.W)

        ttk.Label(control_frame, text="H:").grid(row=2, column=4, sticky=tk.E)
        self.custom_height_var = tk.IntVar(value=self.grid_height)
        self.height_spinbox = ttk.Spinbox(
            control_frame,
            from_=10,
            to=500,
            textvariable=self.custom_height_var,
            width=6,
        )
        self.height_spinbox.grid(row=2, column=5, padx=(0, 5), pady=5, sticky=tk.W)

        ttk.Button(
            control_frame, text="Apply", command=self.apply_custom_grid_size
        ).grid(row=2, column=6, padx=5, pady=5)
        # Drawing tools and symmetry (row 3)
        ttk.Label(control_frame, text="Draw Mode:").grid(
            row=3, column=0, padx=5, pady=5
        )

        draw_modes_frame = ttk.Frame(control_frame)
        draw_modes_frame.grid(
            row=3, column=1, columnspan=3, padx=5, pady=5, sticky=tk.W
        )

        self.draw_mode_var = tk.StringVar(value="toggle")
        ttk.Radiobutton(
            draw_modes_frame,
            text="Toggle",
            variable=self.draw_mode_var,
            value="toggle",
            command=self.update_draw_mode,
        ).pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(
            draw_modes_frame,
            text="Pen",
            variable=self.draw_mode_var,
            value="pen",
            command=self.update_draw_mode,
        ).pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(
            draw_modes_frame,
            text="Eraser",
            variable=self.draw_mode_var,
            value="eraser",
            command=self.update_draw_mode,
        ).pack(side=tk.LEFT, padx=2)

        ttk.Label(control_frame, text="Symmetry:").grid(
            row=3, column=4, padx=5, pady=5, sticky=tk.E
        )

        self.symmetry_var = tk.StringVar(value="none")
        symmetry_options = [
            "None",
            "Horizontal",
            "Vertical",
            "Both",
            "Rotational",
        ]
        self.symmetry_combo = ttk.Combobox(
            control_frame,
            textvariable=self.symmetry_var,
            values=symmetry_options,
            state="readonly",
            width=12,
        )
        self.symmetry_combo.grid(row=3, column=5, columnspan=2, padx=5, pady=5)
        self.symmetry_combo.bind(
            "<<ComboboxSelected>>",
            self.update_symmetry_mode,
        )

        # Speed control
        ttk.Label(control_frame, text="Speed:").grid(row=0, column=6, padx=5)
        self.speed_var = tk.IntVar(value=50)
        speed_scale = ttk.Scale(
            control_frame,
            from_=10,
            to=500,
            variable=self.speed_var,
            orient=tk.HORIZONTAL,
            length=150,
        )
        speed_scale.grid(row=0, column=7, padx=5)

        # Canvas for grid with scrollbars
        canvas_frame = ttk.Frame(self.root)
        canvas_frame.grid(row=1, column=0, padx=10, pady=10)

        x_scroll = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL)
        y_scroll = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL)

        self.canvas = tk.Canvas(
            canvas_frame,
            width=self.grid_width * self.cell_size,
            height=self.grid_height * self.cell_size,
            bg="white",
            xscrollcommand=x_scroll.set,
            yscrollcommand=y_scroll.set,
            scrollregion=(
                0,
                0,
                self.grid_width * self.cell_size,
                self.grid_height * self.cell_size,
            ),
        )
        x_scroll.config(command=self.canvas.xview)
        y_scroll.config(command=self.canvas.yview)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)

        # Mouse interaction
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        # Zoom (cross-platform: MouseWheel + Button-4/5)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind("<Button-4>", self.on_mousewheel)
        self.canvas.bind("<Button-5>", self.on_mousewheel)
        # Pan with middle mouse
        self.canvas.bind("<ButtonPress-2>", self.on_pan_start)
        self.canvas.bind("<B2-Motion>", self.on_pan_move)
        # Shift + Wheel for horizontal scroll
        self.canvas.bind("<Shift-MouseWheel>", self.on_shift_mousewheel)

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(
            self.root,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W,
        )
        status_bar.grid(row=2, column=0, sticky=(tk.W, tk.E), padx=5, pady=5)

        self.generation = 0

    def switch_mode(self, mode):
        """Switch between different cellular automaton modes"""
        was_running = self.running
        if self.running:
            self.toggle_simulation()

        if mode == "Conway's Game of Life":
            self.current_automaton = ConwayGameOfLife(
                self.grid_width,
                self.grid_height,
            )
            # Enable pattern selector
            self.pattern_combo.config(state="readonly")
            self.load_pattern_button.config(state="normal")
            self._set_rules_controls_enabled(False)
            self._update_pattern_list(
                [
                    "Empty",
                    "Classic Mix",
                    "Glider Gun",
                    "Spaceships",
                    "Oscillators",
                    "Puffers",
                    "R-Pentomino",
                    "Acorn",
                    "Random Soup",
                ]
            )
        elif mode == "High Life":
            self.current_automaton = HighLife(
                self.grid_width,
                self.grid_height,
            )
            # Enable pattern selector
            self.pattern_combo.config(state="readonly")
            self.load_pattern_button.config(state="normal")
            self._set_rules_controls_enabled(False)
            self._update_pattern_list(["Empty", "Replicator", "Random Soup"])
        elif mode == "Immigration Game":
            self.current_automaton = ImmigrationGame(
                self.grid_width,
                self.grid_height,
            )
            # Enable pattern selector
            self.pattern_combo.config(state="readonly")
            self.load_pattern_button.config(state="normal")
            self._set_rules_controls_enabled(False)
            self._update_pattern_list(["Empty", "Color Mix", "Random Soup"])
        elif mode == "Rainbow Game":
            self.current_automaton = RainbowGame(
                self.grid_width,
                self.grid_height,
            )
            # Enable pattern selector
            self.pattern_combo.config(state="readonly")
            self.load_pattern_button.config(state="normal")
            self._set_rules_controls_enabled(False)
            self._update_pattern_list(["Empty", "Rainbow Mix", "Random Soup"])
        elif mode == "Langton's Ant":
            self.current_automaton = LangtonsAnt(
                self.grid_width,
                self.grid_height,
            )
            # Disable pattern selector for Langton's Ant
            self.pattern_combo.config(state="disabled")
            self.load_pattern_button.config(state="disabled")
            self._set_rules_controls_enabled(False)
        elif mode == "Custom Rules":
            # Build LifeLike with current rules
            b, s = parse_bs(self.rules_var.get())
            self.current_automaton = LifeLikeAutomaton(
                self.grid_width,
                self.grid_height,
                birth=b,
                survival=s,
            )
            # Enable patterns limited for life-like
            self.pattern_combo.config(state="readonly")
            self.load_pattern_button.config(state="normal")
            self._set_rules_controls_enabled(True)
            self._update_pattern_list(["Empty", "Random Soup"])

        self.generation = 0
        self.update_display()
        self.update_status()

        if was_running:
            self.toggle_simulation()

    def _set_rules_controls_enabled(self, enabled):
        state = "normal" if enabled else "disabled"
        try:
            self.rules_entry.config(state=state)
        except Exception:
            pass

    def apply_custom_rules(self):
        """Apply rules from entry to current LifeLike automaton."""
        if isinstance(self.current_automaton, LifeLikeAutomaton):
            b, s = parse_bs(self.rules_var.get())
            self.current_automaton.set_rules(b, s)
            # Do not reset grid; just apply new rules going forward

    def _update_pattern_list(self, patterns):
        """Update the pattern dropdown with new values"""
        self.pattern_combo["values"] = patterns
        self.pattern_var.set("Empty")

    def load_pattern_handler(self):
        """Handle pattern selection and loading"""
        pattern_name = self.pattern_var.get()

        if pattern_name == "Empty":
            self.clear_grid()
            return

        # Load patterns for automata that support them
        if hasattr(self.current_automaton, "load_pattern"):
            self.current_automaton.load_pattern(pattern_name)
            self.generation = 0
            self.update_display()
            self.update_status()

    def toggle_simulation(self):
        """Start or stop the simulation"""
        self.running = not self.running
        if self.running:
            self.start_button.config(text="Stop")
            self.run_simulation()
        else:
            self.start_button.config(text="Start")

    def run_simulation(self):
        """Main simulation loop"""
        if self.running:
            self.step_once()
            self.update_delay = 510 - self.speed_var.get()
            self.root.after(self.update_delay, self.run_simulation)

    def step_once(self):
        """Perform one step of the simulation"""
        if self.current_automaton:
            # Record history if recording and not in replay mode
            if self.recording_history and self.history_position < 0:
                self.history.append(np.copy(self.current_automaton.get_grid()))

            self.current_automaton.step()
            self.generation += 1
            self.update_display()
            self.update_status()

    def reset_simulation(self):
        """Reset to initial state"""
        if self.current_automaton:
            self.current_automaton.reset()
            self.generation = 0
            self.update_display()
            self.update_status()

    def clear_grid(self):
        """Clear the entire grid"""
        self.reset_simulation()

    def save_pattern(self):
        """Save current grid state to file"""
        if not self.current_automaton:
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Save Pattern",
        )

        if filename:
            try:
                data = {
                    "mode": self.mode_var.get(),
                    "grid": self.current_automaton.get_grid().tolist(),
                    "width": self.grid_width,
                    "height": self.grid_height,
                    "generation": self.generation,
                }
                # Include rules when in Custom Rules mode
                if self.mode_var.get() == "Custom Rules":
                    data["rules"] = self.rules_var.get()

                with open(filename, "w") as f:
                    json.dump(data, f, indent=2)

                messagebox.showinfo(
                    "Success", f"Pattern saved to {os.path.basename(filename)}"
                )
            except Exception as e:
                messagebox.showerror(
                    "Error",
                    f"Failed to save pattern: {str(e)}",
                )

    def load_saved_pattern(self):
        """Load grid state from file"""
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Load Pattern",
        )

        if filename:
            try:
                with open(filename, "r") as f:
                    data = json.load(f)

                # Switch to the correct mode if needed
                if data["mode"] != self.mode_var.get():
                    self.mode_var.set(data["mode"])
                    self.switch_mode(data["mode"])
                # If custom rules, apply rules string first
                if data.get("mode") == "Custom Rules":
                    rules_str = data.get("rules", "B3/S23")
                    self.rules_var.set(rules_str)
                    if isinstance(self.current_automaton, LifeLikeAutomaton):
                        b, s = parse_bs(rules_str)
                        self.current_automaton.set_rules(b, s)

                # Load the grid
                loaded_grid = np.array(data["grid"])

                # Check dimensions match
                if loaded_grid.shape == (self.grid_height, self.grid_width):
                    self.current_automaton.grid = loaded_grid
                    self.generation = data.get("generation", 0)
                    self.prev_grid = None  # Force full redraw
                    self.update_display()
                    self.update_status()
                    messagebox.showinfo(
                        "Success",
                        f"Pattern loaded from {os.path.basename(filename)}",
                    )
                else:
                    messagebox.showerror(
                        "Error",
                        (
                            "Grid size mismatch. File:"
                            f" {loaded_grid.shape}, Current:"
                            f" {(self.grid_height, self.grid_width)}"
                        ),
                    )
            except Exception as e:
                messagebox.showerror(
                    "Error",
                    f"Failed to load pattern: {str(e)}",
                )

    def update_display(self):
        """Optimized redraw - only update changed cells"""
        if not self.current_automaton:
            return

        grid = self.current_automaton.get_grid()

        # Define color palettes for different modes
        immigration_colors = {
            0: "white",
            1: "#FF6B6B",  # Red
            2: "#4ECDC4",  # Cyan
            3: "#FFE66D",  # Yellow
        }

        rainbow_colors = {
            0: "white",
            1: "#FF0000",  # Red
            2: "#FF7F00",  # Orange
            3: "#FFFF00",  # Yellow
            4: "#00FF00",  # Green
            5: "#0000FF",  # Blue
            6: "#8B00FF",  # Purple
        }

        # Export helpers reuse these color maps
        self._immigration_colors = immigration_colors
        self._rainbow_colors = rainbow_colors

        # First time or after reset - create all cells
        if self.prev_grid is None or self.prev_grid.shape != grid.shape:
            self.canvas.delete("all")
            self.cell_rects = {}

            for i in range(self.grid_height):
                for j in range(self.grid_width):
                    x1 = j * self.cell_size
                    y1 = i * self.cell_size
                    x2 = x1 + self.cell_size
                    y2 = y1 + self.cell_size

                    outline_color = "gray" if self.show_grid_lines else ""
                    rect_id = self.canvas.create_rectangle(
                        x1, y1, x2, y2, fill="white", outline=outline_color
                    )
                    self.cell_rects[(i, j)] = rect_id

            self.prev_grid = np.copy(grid)

        # Update only changed cells
        changed = np.where(self.prev_grid != grid)

        for i, j in zip(changed[0], changed[1]):
            cell_value = grid[i, j]

            # Determine color based on mode and cell value
            if isinstance(self.current_automaton, ImmigrationGame):
                color = immigration_colors.get(cell_value, "white")
            elif isinstance(self.current_automaton, RainbowGame):
                color = rainbow_colors.get(cell_value, "white")
            elif isinstance(self.current_automaton, LangtonsAnt):
                if cell_value == 0:
                    color = "white"
                elif cell_value == 1:
                    color = "black"
                else:  # Ant position
                    color = "red"
            else:
                # Conway's Game of Life and High Life
                color = "white" if cell_value == 0 else "black"

            # Update the cell color
            if (i, j) in self.cell_rects:
                self.canvas.itemconfig(self.cell_rects[(i, j)], fill=color)

        # Update previous grid
        self.prev_grid = np.copy(grid)

    def update_status(self):
        """Update status bar with population count"""
        mode_name = self.mode_var.get()
        population = np.count_nonzero(self.current_automaton.get_grid())
        # Record population history for stats panel
        self.population_history.append(population)

        # Calculate FPS
        current_time = time.time()
        delta = current_time - self.last_update_time
        if delta > 0:
            self.frame_times.append(delta)
            if len(self.frame_times) > 0:
                avg_delta = sum(self.frame_times) / len(self.frame_times)
                self.fps = 1.0 / avg_delta if avg_delta > 0 else 0
        self.last_update_time = current_time

        status_text = f"{mode_name} | Gen: {self.generation} | Pop: {population}"
        if self.running:
            status_text += f" | FPS: {self.fps:.1f}"
        if self.history_position >= 0:
            status_text += f" | Replay: {self.history_position}/{len(self.history)}"

        self.status_var.set(status_text)
        # Update stats panel if open
        if self.stats_window is not None:
            self.draw_stats()

    def open_stats_panel(self):
        """Open or focus the statistics panel"""
        if self.stats_window is not None:
            try:
                self.stats_window.lift()
                return
            except Exception:
                self.stats_window = None

        win = tk.Toplevel(self.root)
        win.title("Population Stats")
        win.protocol("WM_DELETE_WINDOW", self.close_stats_panel)
        self.stats_window = win

        self.stats_canvas = tk.Canvas(
            win,
            width=self.stats_canvas_w,
            height=self.stats_canvas_h,
            bg="white",
        )
        self.stats_canvas.pack(fill=tk.BOTH, expand=True)
        self.draw_stats()

    def close_stats_panel(self):
        if self.stats_window is not None:
            try:
                self.stats_window.destroy()
            except Exception:
                pass
        self.stats_window = None
        self.stats_canvas = None

    def draw_stats(self):
        """Draw population over time on stats canvas"""
        if self.stats_canvas is None:
            return

        c = self.stats_canvas
        w = int(c.winfo_width() or self.stats_canvas_w)
        h = int(c.winfo_height() or self.stats_canvas_h)
        c.delete("all")

        # Axes and labels
        padding = 10
        inner_w = max(1, w - 2 * padding)
        inner_h = max(1, h - 2 * padding)
        c.create_rectangle(
            padding,
            padding,
            padding + inner_w,
            padding + inner_h,
            outline="#ddd",
        )

        # Plot population as percentage of cells
        total_cells = max(1, self.grid_width * self.grid_height)
        values = list(self.population_history)
        if len(values) < 2:
            return
        # Normalize to [0,1]
        ys = [v / total_cells for v in values]
        # Create points across inner_w
        step_x = inner_w / (len(ys) - 1)
        points = []
        for idx, yv in enumerate(ys):
            x = padding + idx * step_x
            y = padding + inner_h * (1.0 - yv)
            points.append((x, y))
        # Draw polyline
        for i in range(1, len(points)):
            x1, y1 = points[i - 1]
            x2, y2 = points[i]
            c.create_line(x1, y1, x2, y2, fill="#1f77b4", width=2)
        # Latest value text
        pct = ys[-1] * 100.0
        c.create_text(
            padding + 5,
            padding + 10,
            anchor=tk.W,
            text=f"Pop: {values[-1]} ({pct:.1f}%)",
            fill="#333",
        )

    # ---------- Export helpers ----------
    def _render_grid_to_image(self, scale=None):
        """Render current grid to a PIL Image. Scale defaults to cell_size."""
        if not PIL_AVAILABLE:
            raise RuntimeError("Pillow not available")
        if not self.current_automaton:
            raise RuntimeError("No automaton to render")

        scale = int(scale or self.cell_size)
        scale = max(1, min(40, scale))
        grid = self.current_automaton.get_grid()
        h, w = grid.shape
        img_w = w * scale
        img_h = h * scale

        # Background
        img = Image.new("RGB", (img_w, img_h), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        mode = self.mode_var.get()
        for y in range(h):
            for x in range(w):
                val = int(grid[y, x])
                if mode in ("Immigration Game",):
                    color = self._immigration_colors.get(val, "white")
                elif mode in ("Rainbow Game",):
                    color = self._rainbow_colors.get(val, "white")
                else:
                    # Binary or ant/custom
                    if val == 0:
                        continue
                    color = "black"

                if color == "white":
                    continue
                x0 = x * scale
                y0 = y * scale
                x1 = x0 + scale
                y1 = y0 + scale
                draw.rectangle([x0, y0, x1, y1], fill=color)

        # Langton's Ant overlay as red square
        if mode == "Langton's Ant" and hasattr(self.current_automaton, "ant_x"):
            ax = self.current_automaton.ant_x * scale
            ay = self.current_automaton.ant_y * scale
            draw.rectangle([ax, ay, ax + scale, ay + scale], fill="#FF0000")

        return img

    def export_png(self):
        """Export current view to a PNG file."""
        if not PIL_AVAILABLE:
            messagebox.showerror(
                "Export Error",
                "Pillow is required. Please install 'Pillow' to export.",
            )
            return
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("All files", "*.*")],
            title="Export PNG",
        )
        if not filename:
            return
        try:
            img = self._render_grid_to_image()
            img.save(filename, format="PNG")
            messagebox.showinfo("Export", "PNG exported successfully")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))

    def export_gif(self):
        """Export an animated GIF by stepping the automaton."""
        if not PIL_AVAILABLE:
            messagebox.showerror(
                "Export Error",
                "Pillow is required. Please install 'Pillow' to export.",
            )
            return
        filename = filedialog.asksaveasfilename(
            defaultextension=".gif",
            filetypes=[("GIF", "*.gif"), ("All files", "*.*")],
            title="Export GIF",
        )
        if not filename:
            return
        steps = simpledialog.askinteger(
            "GIF Frames",
            "How many steps to export?",
            initialvalue=50,
            minvalue=1,
            maxvalue=1000,
        )
        if not steps:
            return
        duration = simpledialog.askinteger(
            "Frame Duration (ms)",
            "Milliseconds per frame:",
            initialvalue=100,
            minvalue=10,
            maxvalue=2000,
        )
        if not duration:
            return
        try:
            # Backup state
            backup = None
            mode = self.mode_var.get()
            if mode == "Langton's Ant":
                a = self.current_automaton
                backup = (
                    np.copy(a.grid),
                    a.ant_x,
                    a.ant_y,
                    a.ant_dir,
                )
            else:
                backup = np.copy(self.current_automaton.get_grid())

            images = []
            for _ in range(steps):
                images.append(self._render_grid_to_image())
                self.current_automaton.step()

            # Restore
            if mode == "Langton's Ant":
                a = self.current_automaton
                a.grid = backup[0]
                a.ant_x, a.ant_y, a.ant_dir = backup[1], backup[2], backup[3]
            else:
                self.current_automaton.grid = backup
            self.update_display()
            self.update_status()

            if images:
                images[0].save(
                    filename,
                    save_all=True,
                    append_images=images[1:],
                    duration=duration,
                    loop=0,
                    optimize=False,
                )
                messagebox.showinfo("Export", "GIF exported successfully")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))

    def on_canvas_click(self, event):
        """Handle mouse click on canvas with drawing modes and symmetry"""
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        x = int(canvas_x // self.cell_size)
        y = int(canvas_y // self.cell_size)

        if not (0 <= x < self.grid_width and 0 <= y < self.grid_height):
            return

        mode = self.draw_mode

        if mode in ("line", "rect"):
            # Start or finish shape
            if self.draw_start is None:
                self.draw_start = (x, y)
            else:
                x0, y0 = self.draw_start
                action = self._current_action()
                if mode == "line":
                    self._draw_line(x0, y0, x, y, action=action)
                else:
                    self._draw_rect(x0, y0, x, y, action=action)
                self.draw_start = None
                self.update_display()
        else:
            # Single cell or continuous draw
            action = self._current_action()
            for px, py in self.apply_symmetry(x, y):
                self._apply_action_to_cell(px, py, action)
            self.update_display()
            self.last_draw_pos = (x, y)

    def on_canvas_drag(self, event):
        """Handle mouse drag on canvas for continuous drawing"""
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        x = int(canvas_x // self.cell_size)
        y = int(canvas_y // self.cell_size)

        if not (0 <= x < self.grid_width and 0 <= y < self.grid_height):
            return

        if self.draw_mode in ("line", "rect"):
            # Wait for release to finalize shape
            return

        action = self._current_action()
        if self.last_draw_pos is not None:
            lx, ly = self.last_draw_pos
            for px, py in self._line_points(lx, ly, x, y):
                for sx, sy in self.apply_symmetry(px, py):
                    self._apply_action_to_cell(sx, sy, action)
        else:
            for px, py in self.apply_symmetry(x, y):
                self._apply_action_to_cell(px, py, action)
        self.last_draw_pos = (x, y)
        self.update_display()

    def on_canvas_release(self, event):
        """Reset drag state after mouse release"""
        self.last_draw_pos = None

    # --- Drawing helpers ---
    def _current_action(self):
        """Map draw_mode to an action string"""
        if self.draw_mode == "pen":
            return "set"
        if self.draw_mode == "eraser":
            return "clear"
        return "toggle"

    def _apply_action_to_cell(self, x, y, action):
        """Apply a drawing action to a cell for the current automaton"""
        automaton = self.current_automaton
        if action == "toggle":
            automaton.handle_click(x, y)
            return

        # 'set' or 'clear'
        val = 1 if action == "set" else 0

        if isinstance(automaton, (ConwayGameOfLife, HighLife)):
            automaton.grid[y, x] = val
        elif isinstance(automaton, ImmigrationGame):
            automaton.grid[y, x] = 1 if action == "set" else 0
        elif isinstance(automaton, RainbowGame):
            automaton.grid[y, x] = 1 if action == "set" else 0
        elif isinstance(automaton, LangtonsAnt):
            # Do not move ant on draw; only set the square color
            if action == "set":
                automaton.grid[y, x] = 1
            else:
                automaton.grid[y, x] = 0

    def _line_points(self, x0, y0, x1, y1):
        """Bresenham's line algorithm yielding integer grid points"""
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        x, y = x0, y0
        points = []
        while True:
            if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
                points.append((x, y))
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x += sx
            if e2 <= dx:
                err += dx
                y += sy
        return points

    def _draw_line(self, x0, y0, x1, y1, action):
        for px, py in self._line_points(x0, y0, x1, y1):
            for sx, sy in self.apply_symmetry(px, py):
                self._apply_action_to_cell(sx, sy, action)

    def _draw_rect(self, x0, y0, x1, y1, action):
        x_min, x_max = sorted((x0, x1))
        y_min, y_max = sorted((y0, y1))
        for yy in range(y_min, y_max + 1):
            for xx in range(x_min, x_max + 1):
                for sx, sy in self.apply_symmetry(xx, yy):
                    self._apply_action_to_cell(sx, sy, action)

    def on_mousewheel(self, event):
        """Zoom in/out around mouse position"""
        # Determine direction
        zoom_in = False
        if hasattr(event, "delta") and event.delta != 0:
            zoom_in = event.delta > 0
        elif hasattr(event, "num"):
            # Linux: Button-4 (up), Button-5 (down)
            zoom_in = event.num == 4

        factor = 1.1 if zoom_in else 0.9
        old_size = self.cell_size
        new_size = max(2, min(64, int(round(old_size * factor))))
        if new_size == old_size:
            return

        # World coords under cursor before zoom
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)

        # Update size and scrollregion
        self.cell_size = new_size
        self.prev_grid = None  # Force full redraw to rebuild rectangles
        self.canvas.config(
            scrollregion=(
                0,
                0,
                self.grid_width * self.cell_size,
                self.grid_height * self.cell_size,
            )
        )
        self.update_display()

        # Keep mouse position anchored
        srw = self.grid_width * self.cell_size
        srh = self.grid_height * self.cell_size
        new_cx = cx * (new_size / old_size)
        new_cy = cy * (new_size / old_size)
        view_w = max(1, self.canvas.winfo_width())
        view_h = max(1, self.canvas.winfo_height())
        left_frac = (new_cx - event.x) / srw if srw > 0 else 0
        top_frac = (new_cy - event.y) / srh if srh > 0 else 0
        max_left = max(0.0, 1.0 - view_w / srw) if srw > 0 else 0.0
        max_top = max(0.0, 1.0 - view_h / srh) if srh > 0 else 0.0
        left_frac = min(max(left_frac, 0.0), max_left)
        top_frac = min(max(top_frac, 0.0), max_top)
        self.canvas.xview_moveto(left_frac)
        self.canvas.yview_moveto(top_frac)

    def on_shift_mousewheel(self, event):
        """Horizontal scroll with Shift + wheel"""
        delta = event.delta if hasattr(event, "delta") else 0
        step = -1 if delta > 0 else 1
        self.canvas.xview_scroll(step, "units")

    def on_pan_start(self, event):
        """Begin panning with middle mouse button"""
        self.canvas.scan_mark(event.x, event.y)

    def on_pan_move(self, event):
        """Pan as the middle mouse is dragged"""
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def rewind_history(self):
        """Go back one step in history"""
        if len(self.history) > 0:
            if self.history_position < 0:
                self.history_position = len(self.history) - 1
            elif self.history_position > 0:
                self.history_position -= 1

            if 0 <= self.history_position < len(self.history):
                self.current_automaton.grid = np.copy(
                    self.history[self.history_position]
                )
                self.prev_grid = None
                self.update_display()
                self.update_status()

    def forward_history(self):
        """Go forward one step in history"""
        if self.history_position >= 0:
            if self.history_position < len(self.history) - 1:
                self.history_position += 1
                self.current_automaton.grid = np.copy(
                    self.history[self.history_position]
                )
            else:
                self.history_position = -1  # Return to live mode

            self.prev_grid = None
            self.update_display()
            self.update_status()

    def clear_history(self):
        """Clear the history buffer"""
        self.history.clear()
        self.history_position = -1
        self.update_status()

    def toggle_grid_lines(self):
        """Toggle visibility of grid lines"""
        self.show_grid_lines = not self.show_grid_lines
        self.prev_grid = None  # Force full redraw
        self.update_display()

    def set_draw_mode(self, mode):
        """Set the drawing mode (toggle, pen, eraser, etc.)"""
        self.draw_mode = mode

    def set_symmetry_mode(self, mode):
        """Set symmetry mode for drawing"""
        self.symmetry_mode = mode

    def update_draw_mode(self):
        """Update draw mode from UI control"""
        self.set_draw_mode(self.draw_mode_var.get())

    def update_symmetry_mode(self, event=None):
        """Update symmetry mode from UI control"""
        mode = self.symmetry_var.get().lower()
        self.set_symmetry_mode(mode)

    def apply_symmetry(self, x, y):
        """Apply symmetry transformations to a cell position"""
        positions = [(x, y)]

        if self.symmetry_mode in ["horizontal", "both"]:
            positions.append((self.grid_width - 1 - x, y))

        if self.symmetry_mode in ["vertical", "both"]:
            positions.append((x, self.grid_height - 1 - y))

        if self.symmetry_mode == "both":
            positions.append((self.grid_width - 1 - x, self.grid_height - 1 - y))

        if self.symmetry_mode == "rotational":
            center_x = self.grid_width // 2
            center_y = self.grid_height // 2
            for angle in [90, 180, 270]:
                # Rotate around center
                dx = x - center_x
                dy = y - center_y
                if angle == 90:
                    positions.append((center_x - dy, center_y + dx))
                elif angle == 180:
                    positions.append((center_x - dx, center_y - dy))
                elif angle == 270:
                    positions.append((center_x + dy, center_y - dx))

        return [
            (px, py)
            for px, py in positions
            if 0 <= px < self.grid_width and 0 <= py < self.grid_height
        ]

    def resize_grid(self, width, height):
        """Resize the grid to new dimensions"""
        old_grid = self.current_automaton.get_grid().copy()

        self.grid_width = width
        self.grid_height = height

        # Reinitialize the automaton
        current_mode = self.mode_var.get()
        self.switch_mode(current_mode)

        # Try to preserve content (copy what fits)
        min_h = min(old_grid.shape[0], height)
        min_w = min(old_grid.shape[1], width)
        self.current_automaton.grid[:min_h, :min_w] = old_grid[:min_h, :min_w]

        # Clear caches
        self.cell_rects = {}
        self.prev_grid = None
        self.history.clear()
        self.history_position = -1

        # Recreate canvas
        self.canvas.config(
            width=self.grid_width * self.cell_size,
            height=self.grid_height * self.cell_size,
            scrollregion=(
                0,
                0,
                self.grid_width * self.cell_size,
                self.grid_height * self.cell_size,
            ),
        )

        self.update_display()
        self.update_status()

    def apply_grid_size_preset(self, event=None):
        """Apply a preset grid size"""
        preset = self.size_preset_var.get()

        if preset.startswith("Small"):
            self.resize_grid(50, 40)
            self.custom_width_var.set(50)
            self.custom_height_var.set(40)
        elif preset.startswith("Medium"):
            self.resize_grid(100, 80)
            self.custom_width_var.set(100)
            self.custom_height_var.set(80)
        elif preset.startswith("Large"):
            self.resize_grid(200, 160)
            self.custom_width_var.set(200)
            self.custom_height_var.set(160)
        # "Custom" option doesn't change anything - user uses spinboxes

    def apply_custom_grid_size(self):
        """Apply custom grid size from spinboxes"""
        new_width = self.custom_width_var.get()
        new_height = self.custom_height_var.get()

        if 10 <= new_width <= 500 and 10 <= new_height <= 500:
            self.resize_grid(new_width, new_height)
            self.size_preset_var.set("Custom")
        else:
            messagebox.showwarning(
                "Invalid Size", "Grid dimensions must be between 10 and 500"
            )


def main():
    root = tk.Tk()
    # Keep reference on root to avoid unused-variable lint and GC
    root.app = CellularAutomatonGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
