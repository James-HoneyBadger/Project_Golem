# Development Guide

This document helps you navigate the codebase and contribute changes.

## Project structure
```
HB_Game_Of_Life/
├── src/
│   └── main.py           # Main application (GUI + automata implementations)
├── examples/             # Example pattern files (JSON)
├── backups/              # Development backups and temp files
├── docs/                 # Documentation
│   ├── USER_GUIDE.md
│   └── DEVELOPMENT.md
├── requirements.txt      # Dependencies
├── run.sh                # Helper launcher
├── .gitignore            # Ignore rules
├── LICENSE               # MIT License
└── README.md             # Overview and quick start
```

## Code overview
Currently, the application logic and GUI live in `src/main.py`. Key areas:

- Automata base class (`CellularAutomaton`) and implementations:
  - `ConwayGameOfLife`
  - `HighLife`
  - `ImmigrationGame`
  - `RainbowGame`
  - `LangtonsAnt`
  - `LifeLikeAutomaton` (Custom Rules)
- GUI (`CellularAutomatonGUI`):
  - Control panel creation (mode selection, patterns, save/load/export, rules, grid size, draw modes, symmetry, speed, grid toggle)
  - Canvas rendering (`update_display`) with optional grid lines
  - Drawing interactions (`on_canvas_click`, `on_canvas_drag`, `apply_draw_action`)
  - Simulation loop (`run_simulation`, `step_once`)

Performance:
- Neighbor counting for life-like automata uses `scipy.signal.convolve2d` where applicable
- Rendering is straightforward; future optimization could draw only changed cells

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/main.py
```

## Development tasks (suggested roadmap)
- Split `src/main.py` into modules:
  - `src/automata/` for each automaton
  - `src/gui/` for GUI and drawing utilities
  - `src/patterns/` for built-in pattern definitions
- Add unit tests (pytest) for automata step functions and pattern loaders
- Add keyboard shortcuts (space for start/stop, S step, C clear, G toggle grid)
- Add export GIF/MP4 recording option
- Optimize rendering (only dirty cells update)

## Testing
### Running Tests
The project uses pytest for testing. Tests are located in the `tests/` directory.

**Install pytest** (if not already installed):
```bash
pip install pytest
```

**Run all tests:**
```bash
pytest tests/ -v
```

**Run specific test file:**
```bash
pytest tests/test_conway.py -v
```

**Run specific test function:**
```bash
pytest tests/test_conway.py::test_blinker_oscillates -v
```

### Test Coverage
Current test coverage includes:
- **Conway's Game of Life** (`tests/test_conway.py`):
  - Initialization and grid setup
  - Known oscillators (blinker, toad)
  - Still lifes (block)
  - Spaceships (glider movement)
  - User interactions (cell toggling via clicks)
  - Edge cases (empty grid, reset functionality)

### Writing New Tests
When adding tests for other automata or features:

1. Create test files in `tests/` directory (e.g., `test_highlife.py`)
2. Import the automaton class from `src.main`
3. Test known patterns and edge cases
4. Verify step logic matches expected behavior

Example test structure:
```python
# tests/test_example.py
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from main import ConwayGameOfLife

def test_blinker_oscillates():
    """Test that a blinker pattern oscillates with period 2."""
    life = ConwayGameOfLife(10, 10)
    life.grid[:] = 0
    
    # Set up horizontal blinker at row 5
    for x in [4,5,6]:
        life.grid[5, x] = 1
    
    # After one step, should be vertical
    life.step()
    assert life.grid[4,5] == life.grid[5,5] == life.grid[6,5] == 1
    
    # After second step, back to horizontal
    life.step()
    assert life.grid[5,4] == life.grid[5,5] == life.grid[5,6] == 1
```

### Future Testing Goals
- Add tests for remaining automata:
  - HighLife (B36/S23 rules)
  - ImmigrationGame (4-state interactions)
  - RainbowGame (6-color evolution)
  - LangtonsAnt (directional rules)
  - LifeLikeAutomaton (custom rule parsing)
- Test pattern loaders (ensure correct placement and offsets)
- Test custom rule parsing (`parse_bs()` function)
- Add GUI interaction tests (if feasible with tkinter)
- Implement coverage reporting with pytest-cov
```

## Releasing
- Update docs (README + USER_GUIDE) if features change
- Tag releases in Git for milestone features

## Contributing
Issues and pull requests are welcome. Please:
- Keep PRs focused and small
- Include tests when changing logic
- Update documentation for user-facing changes
