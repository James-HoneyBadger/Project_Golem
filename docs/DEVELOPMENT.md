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
Add `pytest` and create tests for:
- Conway/HighLife rules behavior for known patterns (blinker, toad, block)
- Pattern loaders place cells correctly (centered with offsets)
- Custom rules parsing and application

Example test scaffold:
```python
# tests/test_conway.py
import numpy as np
from src.main import ConwayGameOfLife

def test_blinker_oscillates():
    life = ConwayGameOfLife(10, 10)
    life.grid[:] = 0
    # horizontal blinker at (4,5)
    for x in [4,5,6]:
        life.grid[5, x] = 1
    life.step()
    # should be vertical at x=5, y in [4,5,6]
    assert life.grid[4,5] == life.grid[5,5] == life.grid[6,5] == 1
```

## Releasing
- Update docs (README + USER_GUIDE) if features change
- Tag releases in Git for milestone features

## Contributing
Issues and pull requests are welcome. Please:
- Keep PRs focused and small
- Include tests when changing logic
- Update documentation for user-facing changes
