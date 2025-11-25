# Project Golem — User Guide

This guide explains how to install, run, and use Project Golem. It summarizes controls, available modes, drawing tools, and
workflow tips for saving and exporting.

## What it is
A fast, interactive simulator for cellular automata with multiple modes:
- Conway's Game of Life
- High Life (B36/S23)
- Immigration Game (multi-state colors)
- Rainbow Game (6 colors)
- Langton's Ant
- Custom Rules (life-like B/S rules)

The app is written in Python with Tkinter and NumPy (SciPy optional for
speed) and exports PNG snapshots when Pillow is installed.

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

1) **Mode & simulation controls**
  - Mode: Choose an automaton.
  - Start: Toggle run/pause (`Space`).
  - Step: Advance one generation (`S`).
  - Clear: Clear the grid (`C`).
  - Reset: Return to the mode's initial state.

2) **Patterns & persistence**
  - Pattern: Select a preset; it loads immediately.
  - Save: Write the current grid to JSON.
  - Load File: Load a saved pattern.
  - Export PNG: Save a snapshot (button appears when Pillow is installed).

3) **Custom rules** *(visible in Custom Rules mode)*
  - B field: Digits for birth neighbor counts.
  - S field: Digits for survival neighbor counts.
  - Apply Rules: Apply the values and restart the automaton.

4) **Grid size**
  - Presets: 50×50, 100×100, 150×150, 200×200, Custom.
  - Custom width/height (10–500) plus Apply.

5) **Drawing tools & symmetry**
  - Draw mode: Toggle, Pen, Eraser.
  - Symmetry: None, Horizontal, Vertical, Both, Radial.

6) **Speed & stats**
  - Speed slider: Adjust simulation speed.
  - Toggle Grid: Show/hide gridlines (`G`).
  - Statistics label: Live cells, delta, peak, density.

Canvas: The large white area renders the automaton.

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
- Toggle mode: Clicking a cell flips it between active/inactive.
- Pen mode: Clicking/dragging paints active cells.
- Eraser mode: Clicking/dragging clears cells.

Symmetry options mirror each action:
- Horizontal: Mirror across the vertical axis.
- Vertical: Mirror across the horizontal axis.
- Both: Four-way mirroring.
- Radial: Four-way rotation around the center.

Tip: For precise editing, pause the simulation and enable grid lines.

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
