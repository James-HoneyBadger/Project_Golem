"""Pattern definitions and loading utilities for cellular automata."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


# Pattern data as (relative_coords, description)
PATTERN_DATA: Dict[str, Dict[str, Tuple[List[Tuple[int, int]], str]]] = {
    "Conway's Game of Life": {
        "Glider Gun": (
            [
                (0, 4), (0, 5), (1, 4), (1, 5), (10, 4), (10, 5), (10, 6),
                (11, 3), (11, 7), (12, 2), (12, 8), (13, 2), (13, 8),
                (14, 5), (15, 3), (15, 7), (16, 4), (16, 5), (16, 6),
                (17, 5), (20, 2), (20, 3), (20, 4), (21, 2), (21, 3),
                (21, 4), (22, 1), (22, 5), (24, 0), (24, 1), (24, 5),
                (24, 6), (34, 2), (34, 3), (35, 2), (35, 3),
            ],
            "Gosper glider gun - produces gliders periodically"
        ),
        "Beacon": (
            [(0, 0), (1, 0), (0, 1), (3, 2), (2, 3), (3, 3)],
            "Beacon - period 2 oscillator"
        ),
        "Pulsar": (
            [
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
            ],
            "Pulsar - period 3 oscillator"
        ),
    },
    "High Life": {
        "Replicator": (
            [(1, 0), (0, 1), (1, 1), (2, 1), (0, 2), (2, 2), (1, 3)],
            "Replicator - grows exponentially"
        ),
    },
}


def get_pattern_coords(mode: str, pattern_name: str) -> List[Tuple[int, int]]:
    """Get the coordinates for a named pattern."""
    if mode in PATTERN_DATA and pattern_name in PATTERN_DATA[mode]:
        return PATTERN_DATA[mode][pattern_name][0]
    return []


def get_pattern_description(mode: str, pattern_name: str) -> str:
    """Get the description for a named pattern."""
    if mode in PATTERN_DATA and pattern_name in PATTERN_DATA[mode]:
        return PATTERN_DATA[mode][pattern_name][1]
    return ""


def apply_pattern_to_grid(
    grid: np.ndarray,
    pattern_coords: List[Tuple[int, int]],
    center_x: int,
    center_y: int
) -> None:
    """Apply a pattern to the grid centered at the given coordinates."""
    height, width = grid.shape
    for dx, dy in pattern_coords:
        x = center_x + dx
        y = center_y + dy
        if 0 <= x < width and 0 <= y < height:
            grid[y, x] = 1
