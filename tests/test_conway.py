"""Tests for Conway's Game of Life automaton"""

import sys
import os
import numpy as np

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Import from the monolithic main.py
from main import ConwayGameOfLife


def test_conway_initialization():
    """Test that Conway's Game of Life initializes with empty grid"""
    life = ConwayGameOfLife(10, 10)
    assert life.width == 10
    assert life.height == 10
    assert life.grid.shape == (10, 10)
    assert np.all(life.grid == 0)


def test_blinker_oscillates():
    """Test that a blinker oscillates with period 2"""
    life = ConwayGameOfLife(5, 5)

    # Set up horizontal blinker at center
    life.grid[2, 1] = 1
    life.grid[2, 2] = 1
    life.grid[2, 3] = 1

    initial_state = life.grid.copy()

    # After one step, should be vertical
    life.step()
    assert life.grid[1, 2] == 1
    assert life.grid[2, 2] == 1
    assert life.grid[3, 2] == 1
    assert life.grid[2, 1] == 0
    assert life.grid[2, 3] == 0

    # After another step, should be back to horizontal
    life.step()
    assert np.array_equal(life.grid, initial_state)


def test_block_is_stable():
    """Test that a block (2x2 square) is stable"""
    life = ConwayGameOfLife(5, 5)

    # Set up block in center
    life.grid[2, 2] = 1
    life.grid[2, 3] = 1
    life.grid[3, 2] = 1
    life.grid[3, 3] = 1

    initial_state = life.grid.copy()

    # Should remain unchanged after steps
    life.step()
    assert np.array_equal(life.grid, initial_state)

    life.step()
    assert np.array_equal(life.grid, initial_state)


def test_toad_oscillates():
    """Test that a toad oscillates with period 2"""
    life = ConwayGameOfLife(6, 6)

    # Set up toad (horizontal orientation)
    life.grid[2, 2] = 1
    life.grid[2, 3] = 1
    life.grid[2, 4] = 1
    life.grid[3, 1] = 1
    life.grid[3, 2] = 1
    life.grid[3, 3] = 1

    initial_state = life.grid.copy()

    # After one step
    life.step()
    # Should have changed
    assert not np.array_equal(life.grid, initial_state)

    # After two steps, should be back
    life.step()
    assert np.array_equal(life.grid, initial_state)


def test_glider_moves():
    """Test that a glider pattern moves across the grid"""
    life = ConwayGameOfLife(10, 10)

    # Set up glider at top-left
    life.grid[1, 2] = 1
    life.grid[2, 3] = 1
    life.grid[3, 1] = 1
    life.grid[3, 2] = 1
    life.grid[3, 3] = 1

    # Count live cells
    initial_count = np.sum(life.grid)

    # After 4 steps, glider should have moved (still 5 cells)
    for _ in range(4):
        life.step()

    # Should still have 5 live cells
    assert np.sum(life.grid) == initial_count


def test_handle_click_toggles():
    """Test that clicking toggles cell state"""
    life = ConwayGameOfLife(5, 5)

    # Initially empty
    assert life.grid[2, 2] == 0

    # Click to activate
    life.handle_click(2, 2)
    assert life.grid[2, 2] == 1

    # Click again to deactivate
    life.handle_click(2, 2)
    assert life.grid[2, 2] == 0


def test_reset_clears_grid():
    """Test that reset clears the grid"""
    life = ConwayGameOfLife(5, 5)

    # Add some live cells
    life.grid[1, 1] = 1
    life.grid[2, 2] = 1
    life.grid[3, 3] = 1

    # Reset should clear everything
    life.reset()
    assert np.all(life.grid == 0)


def test_empty_grid_stays_empty():
    """Test that an empty grid stays empty"""
    life = ConwayGameOfLife(5, 5)

    initial_state = life.grid.copy()
    life.step()

    assert np.array_equal(life.grid, initial_state)
