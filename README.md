git clone https://github.com/James-HoneyBadger/Project_Golem.git
## Project Golem

An interactive Tkinter-based workbench for experimenting with cellular
automata. The simulator ships with several classic rules, a custom B/S rule
editor, drawing tools, and quick exporting to PNG.

---

## Highlights

- **Multiple automata**: Conway's Life, HighLife, Immigration, Rainbow,
  Langton's Ant, and fully custom life-like rules.
- **Pattern presets** per mode for quick experimentation.
- **Drawing tools** with toggle/pen/eraser modes plus symmetry helpers.
- **Live statistics** for population deltas, peaks, and density.
- **Save/Load** patterns as JSON and **export PNG** snapshots (when Pillow is
  installed).
- **Keyboard shortcuts**: `Space` (start/stop), `S` (step), `C` (clear), `G`
  (toggle grid).

---

## Requirements

- Python 3.8+
- Tkinter (bundled with most Python installations)
- NumPy 1.24+
- SciPy 1.11+ (used for fast convolutions)
- Pillow 10+ (optional, enables PNG export)

Install dependencies from the repository root:

```bash
pip install -r requirements.txt
```

---

## Getting Started

Run the simulator from the project root:

```bash
python src/main.py
```

Or use the helper script on Unix-like systems:

```bash
./run.sh
```

**Quick workflow**
1. Pick a mode from the **Mode** dropdown.
2. Choose a **Pattern** or draw on the canvas.
3. Press **Start** (or hit `Space`) to run the simulation.
4. Adjust **Speed**, drawing tools, symmetry, and grid size as needed.
5. Save patterns (`Save`) or export snapshots (`Export PNG`).

---

## Controls at a Glance

| Action | UI Control | Shortcut |
| --- | --- | --- |
| Start/Stop simulation | Start button | `Space` |
| Step one generation | Step button | `S` |
| Clear grid | Clear button | `C` |
| Toggle grid lines | Toggle Grid button | `G` |
| Resize grid | Presets or custom width/height | – |
| Apply custom B/S rule | Apply Rules | – |

Mouse interactions:

- Click to toggle/draw/erase (depends on draw mode).
- Drag while in Pen or Eraser to paint continuously.
- Symmetry options mirror strokes across selected axes.

---

## Available Modes & Patterns

- **Conway's Game of Life**: Classic Mix, Glider Gun, Spaceships, Oscillators,
  Puffers, R-Pentomino, Acorn, Random Soup.
- **HighLife (B36/S23)**: Replicator, Random Soup.
- **Immigration Game**: Color Mix, Random Soup.
- **Rainbow Game**: Rainbow Mix, Random Soup.
- **Langton's Ant**: Empty.
- **Custom Rules**: Random Soup starter pattern plus editable B/S fields.

---

## Project Structure

```
Project_Golem/
├── src/
│   ├── automata/        # Automaton implementations
│   ├── gui/             # GUI modules (app, config, state, ui, rendering)
│   └── main.py          # Thin entry point (delegates to gui.app)
├── docs/                # README-style documentation
├── examples/            # Sample patterns
├── tests/               # Unit tests
├── requirements.txt
├── run.sh
├── LICENSE
└── README.md
```

Key GUI modules:

- `gui/app.py`: High-level application orchestration.
- `gui/ui.py`: Widget construction and event wiring.
- `gui/state.py`: Mutable simulation state container.
- `gui/config.py`: Shared constants and mode registries.
- `gui/rendering.py`: Canvas drawing helpers.

---

## Development Notes

- Launch tests with `pytest`. Current coverage targets the Conway automaton;
  extending coverage for other modes is encouraged.
- `flake8` enforces an 80-character line limit; run `flake8 src tests` before
  committing.
- To add a new automaton, implement it under `src/automata/`, expose it from
  `automata/__init__.py`, and register it in `gui/config.py`.
- The GUI is intentionally modular: prefer adding features in dedicated helper
  modules rather than growing `gui/app.py`.

Further details can be found in:

- `docs/USER_GUIDE.md` – end-user walkthrough.
- `docs/DEVELOPMENT.md` – contributor guidelines and code map.

---

## License

This project is distributed under the MIT License. See the `LICENSE` file for
full terms.
