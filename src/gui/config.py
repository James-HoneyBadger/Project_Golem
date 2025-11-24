"""Static configuration shared by the GUI components."""

from __future__ import annotations

from typing import Callable, Dict, List

# pylint: disable=import-error

from automata import (
    CellularAutomaton,
    ConwayGameOfLife,
    HighLife,
    ImmigrationGame,
    LangtonsAnt,
    RainbowGame,
    parse_bs,
)

# Default custom rule (Conway)
DEFAULT_CUSTOM_RULE = "B3/S23"
DEFAULT_CUSTOM_BIRTH, DEFAULT_CUSTOM_SURVIVAL = parse_bs(DEFAULT_CUSTOM_RULE)

# Factory registry for standard modes
MODE_FACTORIES: Dict[str, Callable[[int, int], CellularAutomaton]] = {
    "Conway's Game of Life": ConwayGameOfLife,
    "High Life": HighLife,
    "Immigration Game": ImmigrationGame,
    "Rainbow Game": RainbowGame,
    "Langton's Ant": LangtonsAnt,
}

# Pattern options per mode
MODE_PATTERNS: Dict[str, List[str]] = {
    "Conway's Game of Life": [
        "Classic Mix",
        "Glider Gun",
        "Spaceships",
        "Oscillators",
        "Puffers",
        "R-Pentomino",
        "Acorn",
        "Random Soup",
    ],
    "High Life": ["Replicator", "Random Soup"],
    "Immigration Game": ["Color Mix", "Random Soup"],
    "Rainbow Game": ["Rainbow Mix", "Random Soup"],
    "Langton's Ant": ["Empty"],
    "Custom Rules": ["Random Soup"],
}

# Color map shared by export and canvas painting
CELL_COLORS = {
    0: "white",
    1: "black",
    2: "red",
    3: "orange",
    4: "yellow",
    5: "green",
    6: "blue",
    7: "purple",
}

EXPORT_COLOR_MAP = {
    0: (255, 255, 255),
    1: (0, 0, 0),
    2: (255, 0, 0),
    3: (255, 128, 0),
    4: (255, 255, 0),
    5: (0, 200, 0),
    6: (0, 0, 255),
    7: (150, 0, 255),
}

MAX_HISTORY_LENGTH = 500
MIN_GRID_SIZE = 10
MAX_GRID_SIZE = 500
DEFAULT_CELL_SIZE = 8
DEFAULT_CANVAS_WIDTH = 800
DEFAULT_CANVAS_HEIGHT = 600
DEFAULT_SPEED = 50
