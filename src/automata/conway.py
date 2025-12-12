# pylint: disable=duplicate-code

"""Conway's Game of Life implementation."""

from __future__ import annotations

import numpy as np
from scipy import signal

from .base import CellularAutomaton


class ConwayGameOfLife(CellularAutomaton):
    """Conway's Game of Life implementation."""

    def __init__(self, width: int, height: int) -> None:
        self.grid = np.zeros((height, width), dtype=int)
        super().__init__(width, height)

    def reset(self) -> None:
        self.grid = np.zeros((self.height, self.width), dtype=int)

    def load_pattern(self, pattern_name: str) -> None:
        """Load a predefined pattern onto the grid."""
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
        elif pattern_name == "Beacon":
            self._add_beacon(center_x, center_y)
        elif pattern_name == "Pulsar":
            self._add_pulsar(center_x, center_y)

    def _add_classic_mix(self, center_x: int, center_y: int) -> None:
        """Add interesting default patterns to the grid."""
        glider = [(1, 0), (2, 1), (0, 2), (1, 2), (2, 2)]
        for dx, dy in glider:
            x, y = center_x - 30 + dx, center_y - 25 + dy
            if 0 <= x < self.width and 0 <= y < self.height:
                self.grid[y, x] = 1

        blinker = [(0, 0), (1, 0), (2, 0)]
        for dx, dy in blinker:
            x, y = center_x + dx - 1, center_y - 30 + dy
            if 0 <= x < self.width and 0 <= y < self.height:
                self.grid[y, x] = 1

        toad = [(1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1)]
        for dx, dy in toad:
            x, y = center_x - 25 + dx, center_y + dy - 1
            if 0 <= x < self.width and 0 <= y < self.height:
                self.grid[y, x] = 1

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

        block = [(0, 0), (1, 0), (0, 1), (1, 1)]
        for dx, dy in block:
            x, y = center_x - 30 + dx, center_y + 20 + dy
            if 0 <= x < self.width and 0 <= y < self.height:
                self.grid[y, x] = 1

        beehive = [(1, 0), (2, 0), (0, 1), (3, 1), (1, 2), (2, 2)]
        for dx, dy in beehive:
            x, y = center_x + 20 + dx, center_y + 20 + dy
            if 0 <= x < self.width and 0 <= y < self.height:
                self.grid[y, x] = 1

    def _add_glider_gun(self, center_x: int, center_y: int) -> None:
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

    def _add_puffers(self, center_x: int, center_y: int) -> None:
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

    def _add_oscillators(self, center_x: int, center_y: int) -> None:
        blinker = [(0, 0), (1, 0), (2, 0)]
        for dx, dy in blinker:
            x, y = center_x - 30 + dx, center_y - 20 + dy
            if 0 <= x < self.width and 0 <= y < self.height:
                self.grid[y, x] = 1

        toad = [(1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1)]
        for dx, dy in toad:
            x, y = center_x - 20 + dx, center_y - 20 + dy
            if 0 <= x < self.width and 0 <= y < self.height:
                self.grid[y, x] = 1

        beacon = [(0, 0), (1, 0), (0, 1), (3, 2), (2, 3), (3, 3)]
        for dx, dy in beacon:
            x, y = center_x - 5 + dx, center_y - 20 + dy
            if 0 <= x < self.width and 0 <= y < self.height:
                self.grid[y, x] = 1

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

    def _add_spaceships(self, center_x: int, center_y: int) -> None:
        glider = [(1, 0), (2, 1), (0, 2), (1, 2), (2, 2)]
        for dx, dy in glider:
            x, y = center_x - 30 + dx, center_y - 20 + dy
            if 0 <= x < self.width and 0 <= y < self.height:
                self.grid[y, x] = 1

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

    def _add_random_soup(self) -> None:
        random_mask = np.random.random(self.grid.shape) < 0.15
        self.grid[random_mask] = 1

    def _add_r_pentomino(self, center_x: int, center_y: int) -> None:
        r_pentomino = [(1, 0), (2, 0), (0, 1), (1, 1), (1, 2)]
        for dx, dy in r_pentomino:
            x, y = center_x + dx - 1, center_y + dy - 1
            if 0 <= x < self.width and 0 <= y < self.height:
                self.grid[y, x] = 1

    def _add_acorn(self, center_x: int, center_y: int) -> None:
        acorn = [(1, 0), (3, 1), (0, 2), (1, 2), (4, 2), (5, 2), (6, 2)]
        for dx, dy in acorn:
            x, y = center_x + dx - 3, center_y + dy - 1
            if 0 <= x < self.width and 0 <= y < self.height:
                self.grid[y, x] = 1

    def _add_beacon(self, center_x: int, center_y: int) -> None:
        beacon = [(0, 0), (1, 0), (0, 1), (3, 2), (2, 3), (3, 3)]
        for dx, dy in beacon:
            x, y = center_x + dx - 1, center_y + dy - 1
            if 0 <= x < self.width and 0 <= y < self.height:
                self.grid[y, x] = 1

    def _add_pulsar(self, center_x: int, center_y: int) -> None:
        pulsar = [
            (2, 0), (3, 0), (4, 0), (8, 0), (9, 0), (10, 0),
            (0, 2), (5, 2), (7, 2), (12, 2),
            (0, 3), (5, 3), (7, 3), (12, 3),
            (0, 4), (5, 4), (7, 4), (12, 4),
            (2, 5), (3, 5), (4, 5), (8, 5), (9, 5), (10, 5),
            (2, 7), (3, 7), (4, 7), (8, 7), (9, 7), (10, 7),
            (0, 8), (5, 8), (7, 8), (12, 8),
            (0, 9), (5, 9), (7, 9), (12, 9),
            (0, 10), (5, 10), (7, 10), (12, 10),
            (2, 12), (3, 12), (4, 12), (8, 12), (9, 12), (10, 12),
        ]
        for dx, dy in pulsar:
            x, y = center_x + dx - 6, center_y + dy - 6
            if 0 <= x < self.width and 0 <= y < self.height:
                self.grid[y, x] = 1

    def step(self) -> None:
        """Advance the automaton by one generation."""
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        neighbors = signal.convolve2d(
            self.grid,
            kernel,
            mode="same",
            boundary="wrap",
        )
        self.grid = (
            ((self.grid == 1) & ((neighbors == 2) | (neighbors == 3)))
            | ((self.grid == 0) & (neighbors == 3))
        ).astype(int)

    def get_grid(self) -> np.ndarray:
        return self.grid  # type: ignore[no-any-return]

    def handle_click(self, x: int, y: int) -> None:
        self.grid[y, x] = 1 - self.grid[y, x]
