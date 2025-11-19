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
        self.root.title("Cellular Automaton GUI")
        
        # All the state variables from before...
        self.grid_width = 100
        self.grid_height = 80  
        self.cell_size = 8
        self.update_delay = 50
        self.running = False
        self.current_automaton = None
        self.cell_rects = {}
        self.prev_grid = None
        self.last_update_time = time.time()
        self.fps = 0
        self.frame_times = deque(maxlen=30)
        self.population_history = deque(maxlen=500)
        self.stats_window = None
        self.stats_canvas = None
        self.stats_canvas_w = 400
        self.stats_canvas_h = 120
        self.history = deque(maxlen=1000)
        self.history_position = -1
        self.recording_history = True
        self.draw_mode = 'toggle'
        self.draw_start = None
        self.last_draw_pos = None
        self.show_grid_lines = True
        self.symmetry_mode = 'none'
        self.zoom_level = 1.0
        self.pan_offset_x = 0
        self.pan_offset_y = 0
        
        self.create_widgets()
        self.setup_keyboard_shortcuts()
        self.switch_mode('Conway\'s Game of Life')

