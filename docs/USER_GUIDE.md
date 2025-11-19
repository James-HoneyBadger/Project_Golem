# Cellular Automaton Simulator — User Guide

This guide explains how to install, run, and use the Cellular Automaton Simulator, including all controls, modes, patterns, saving/loading, and exporting.

## What it is
A fast, interactive simulator for cellular automata with multiple modes:
- Conway's Game of Life
- High Life (B36/S23)
- Immigration Game (multi-state colors)
- Rainbow Game (6 colors)
- Langton's Ant
- Custom Rules (life-like B/S rules)

The app is written in Python with Tkinter and NumPy (SciPy optional for speed), and can export snapshots as PNG images.

---

## Installation

Requirements:
- Python 3.8+
- NumPy
- SciPy (for fast convolution-based neighbor counting)
- Pillow (optional, for PNG export)

Install from the repo root:
```bash
pip install -r requirements.txt
```

If you use a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate  # Windows (PowerShell)
pip install -r requirements.txt
```

---

## Running the app
From the repo root:
```bash
python src/main.py
```

Or use the helper script:
```bash
./run.sh
```

---

## UI overview
The top control panel is organized in rows.

1) Mode and simulation controls
- Mode: Choose one of the automaton modes
- Start: Toggle run/pause
- Step: Advance one generation
- Clear: Clear grid to empty state
- Reset: Reset to the mode's initial state

2) Patterns and file actions
- Pattern: Select a pattern (auto-loads when selected)
- Save: Save current grid to a JSON file
- Load File: Load a saved JSON pattern
- Export PNG: Save a PNG snapshot (requires Pillow)

3) Grid size
- Presets: 50x50, 100x100, 150x150, 200x200, Custom
- W/H + Apply: Set custom dimensions (10–500)

4) Drawing tools and symmetry
- Draw mode: Toggle, Pen, Eraser
- Symmetry: None, Horizontal, Vertical, Both, Radial

5) Speed and display
- Speed: Simulation speed (higher = faster)
- Toggle Grid: Show/hide gridlines
- Generation: Current generation count

Canvas: The large white area where the automaton is displayed.

---

## Modes and patterns
- Conway's Game of Life
  - Patterns: Classic Mix, Glider Gun, Spaceships, Oscillators, Puffers, R-Pentomino, Acorn, Random Soup
- High Life (B36/S23)
  - Patterns: Replicator, Random Soup
- Immigration Game
  - Patterns: Color Mix, Random Soup
- Rainbow Game
  - Patterns: Rainbow Mix, Random Soup
- Langton's Ant
  - Patterns: Empty
- Custom Rules
  - Patterns: Random Soup (start and then set rules)

Note: When switching modes, the first pattern in the list is selected by default and may automatically initialize the grid for you.

---

## Drawing on the grid
- Toggle mode: Clicking a cell flips it between active/inactive
- Pen mode: Clicking/dragging paints active cells
- Eraser mode: Clicking/dragging clears cells

Symmetry options affect every click/drag:
- Horizontal: Mirrors across the vertical axis
- Vertical: Mirrors across the horizontal axis
- Both: Four-way mirroring
- Radial: 4-way rotation around the center

Tip: For precise editing, temporarily disable simulation (Stop) and enable grid lines.

---

## Resizing the grid
Use presets (50x50, 100x100, 150x150, 200x200) or set custom W/H and Apply. Resizing recreates the automaton for the new size.

---

## Saving, loading, and exporting
- Save: Writes a JSON file with mode, width, height, and the grid state
- Load File: Reads a JSON file created by the app and restores the state (mode and size will be applied)
- Export PNG: Saves a PNG image of the current grid (requires Pillow)

Tip: Keep your pattern files in the `examples/` folder for easy sharing.

---

## Performance tips
- Lower grid dimensions or increase cell size to improve rendering speed
- Use SciPy (installed via requirements.txt) for fast neighbor counting
- Hide grid lines if you want a cleaner look and slightly less draw overhead

---

## Troubleshooting
- Tkinter not found
  - Ensure you have Python with Tk support installed
- ImportError: No module named 'scipy' or 'numpy'
  - Install dependencies: `pip install -r requirements.txt`
- Export PNG button missing
  - Pillow is optional; install it with `pip install Pillow`
- Slow performance on very large grids
  - Reduce grid size or cell size; ensure SciPy is installed

---

## FAQ
- Can I add my own rules?
  - Yes. Switch to Custom Rules and set B (birth) and S (survival) fields, then Apply Rules.
- Are there keyboard shortcuts?
  - Currently, controls are provided via buttons and mouse interactions on canvas.
- Can I export animations/GIFs?
  - Not yet. PNG snapshots are supported; animations can be added in the future.

Enjoy exploring patterns! If you discover interesting ones, save them and share via JSON.
