"""Rendering utilities for drawing grids and applying symmetry."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .config import CELL_COLORS


def draw_grid(
    canvas,
    grid: np.ndarray,
    cell_size: int,
    show_grid: bool,
) -> None:
    """Render the automaton grid to the canvas."""

    height, width = grid.shape
    canvas.delete("all")
    scroll_region = (0, 0, width * cell_size, height * cell_size)
    canvas.configure(scrollregion=scroll_region)

    outline = "gray" if show_grid else ""
    for y in range(height):
        for x in range(width):
            x1 = x * cell_size
            y1 = y * cell_size
            x2 = x1 + cell_size
            y2 = y1 + cell_size
            color = CELL_COLORS.get(int(grid[y, x]), "white")
            canvas.create_rectangle(
                x1,
                y1,
                x2,
                y2,
                fill=color,
                outline=outline,
                width=1,
            )


def symmetry_positions(
    x: int,
    y: int,
    grid_width: int,
    grid_height: int,
    symmetry: str,
) -> List[Tuple[int, int]]:
    """Return the list of coordinates affected by the symmetry mode."""

    positions = {(x, y)}
    if symmetry in ("Horizontal", "Both"):
        positions.add((grid_width - 1 - x, y))
    if symmetry in ("Vertical", "Both"):
        positions.add((x, grid_height - 1 - y))
    if symmetry == "Both":
        positions.add((grid_width - 1 - x, grid_height - 1 - y))
    if symmetry == "Radial":
        cx, cy = grid_width // 2, grid_height // 2
        dx, dy = x - cx, y - cy
        radial = {
            (cx + dx, cy + dy),
            (cx - dx, cy - dy),
            (cx - dy, cy + dx),
            (cx + dy, cy - dx),
        }
        positions.update(radial)
    return list(positions)
