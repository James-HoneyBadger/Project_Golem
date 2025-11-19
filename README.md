# Cellular Automaton GUI

A high-performance graphical user interface for simulating various cellular automata, including Conway's Game of Life, High Life, Immigration Game, Rainbow Game, and Langton's Ant.

## Features

### Multiple Modes:
- **Conway's Game of Life** - Classic cellular automaton with birth/death rules (B3/S23)
- **High Life (B36/S23)** - Variant that can create replicator patterns
- **Immigration Game** - Multi-state automaton with 4 colors (red, cyan, yellow)
- **Rainbow Game** - 6-color variant (red, orange, yellow, green, blue, purple)
- **Langton's Ant** - Simple ant that creates complex patterns
- **Custom Rules** - Define your own B/S (Birth/Survival) rules for life-like automata

### Performance Optimizations ⚡
- **Vectorized neighbor counting** using NumPy convolution (10-100x faster than loops)
- **Optimized rendering** - Only redraws changed cells instead of entire grid
- **Population statistics** - Live cell count displayed in status bar

### Interactive Controls:
- **Pattern Library** - Pre-loaded patterns for each mode
  - Conway's: Classic Mix, Glider Gun, Spaceships, Oscillators, Puffers, R-Pentomino, Acorn, Random Soup
  - High Life: Replicator, Random Soup
  - Immigration/Rainbow: Color Mix, Random Soup
- **Start/Stop** - Run or pause the simulation
- **Step** - Advance one generation at a time
- **Reset** - Return to initial state
- **Clear** - Clear the entire grid
- **Speed Control** - Adjust simulation speed with slider

### Save/Load Functionality 💾
- **Save patterns** - Export current grid state to JSON file
- **Load patterns** - Import previously saved patterns
- Preserves mode, grid state, generation count, and custom rules (if applicable)

### Export Features 📸
- **Export PNG** - Save current grid state as an image
- **Export GIF** - Create animated sequences with customizable:
  - Number of frames (1-1000 steps)
  - Frame duration (10-2000ms per frame)
  - Non-destructive (restores grid state after export)

### Keyboard Shortcuts ⌨️
- **Space** - Play/Pause simulation
- **R** - Reset to initial state
- **C** - Clear grid
- **S** - Step one generation
- **1-5** - Quick mode selection
  - 1: Conway's Game of Life
  - 2: High Life
  - 3: Immigration Game
  - 4: Rainbow Game
  - 5: Langton's Ant
- **Ctrl+S** - Save pattern
- **Ctrl+O** - Load pattern
- **Left Arrow** - Rewind history (go back one generation)
- **Right Arrow** - Forward in history (replay next generation)
- **G** - Toggle grid lines on/off

### Custom Rules (B/S Notation) 🧬
- **Birth/Survival** - Define rules using standard notation (e.g., B3/S23)
- **Live Editing** - Apply new rules without resetting the grid
- **Examples**:
  - `B3/S23` - Conway's Game of Life (standard)
  - `B36/S23` - High Life (replicators)
  - `B368/S245` - Morley
  - `B3/S012345678` - Life Without Death
  - `B2/S` - Seeds (all cells die immediately)
- **Pattern Support** - Custom mode includes Random Soup for testing

### Statistics Panel 📊
- **Population Graph** - Live visualization of population over time
- **500-point Buffer** - Tracks recent history for trend analysis
- **Percentage Display** - Shows population as % of total cells
- **Toplevel Window** - Non-blocking, resizable panel

### History & Replay 🎬
- **History Recording** - Automatically records up to 1000 generations
- **Rewind/Forward Buttons** - Navigate through simulation history
- **Replay Position** - Shows current position in history buffer
- **Non-destructive** - Replay doesn't modify the recorded history

### Advanced Controls 🎨
- **Grid Size Presets** - Quick resize to Small (50×40), Medium (100×80), Large (200×160)
- **Custom Grid Size** - Set any dimensions from 10×10 to 500×500
- **Content Preservation** - Resizing preserves existing patterns
- **Grid Lines Toggle** - Show/hide grid lines for cleaner visualization
- **FPS Display** - Real-time frames per second indicator when running

### Drawing Tools 🖌️
- **Toggle Mode** - Click to toggle cell state (default)
- **Pen Mode** - Draw cells in active state
- **Eraser Mode** - Clear cells to inactive state
- **Symmetry Options**:
  - None - Normal single-cell drawing
  - Horizontal - Mirror across vertical axis
  - Vertical - Mirror across horizontal axis
  - Both - 4-way symmetry (horizontal + vertical)
  - Rotational - 4-way rotational symmetry around center

### Mouse Interaction:
- **Click** cells to toggle them (Conway's/High Life) or cycle colors (Immigration/Rainbow)
- **Drag** to paint multiple cells
- **Click** to position Langton's Ant

## Requirements

- Python 3.7+
- tkinter (usually included with Python)
- numpy >= 1.24.0
- scipy >= 1.11.0
- Pillow >= 10.0.0 (for PNG/GIF export)

## Installation

## Project Structure

```
HB_Game_Of_Life/
├── src/
│   └── main.py           # Main application file
├── examples/             # Example pattern files
│   └── README.md
├── backups/              # Development backups (not in git)
│   └── README.md
├── docs/                 # Documentation and screenshots
├── requirements.txt      # Python dependencies
├── run.sh                # Convenient launch script
├── .gitignore            # Git ignore rules
├── LICENSE               # MIT License
└── README.md             # This file
```

## Installation

Install dependencies:
```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install numpy scipy Pillow
```

Or if using the virtual environment:
```bash
cd /home/james/Cell_Auto
source .venv/bin/activate  # On Linux/Mac
pip install -r requirements.txt
```

## Usage

Run the application:

**Quick Start:**
```bash
# Clone the repository
git clone https://github.com/James-HoneyBadger/HB_Game_Of_Life.git
cd HB_Game_Of_Life

# Install dependencies
pip install -r requirements.txt

# Run the application
python src/main.py
```

Or use the convenient launch script:
```bash
./run.sh
```

### Getting Started:

1. **Select a mode** from the dropdown menu
2. **Load a pattern** or click cells to create your own
3. **Press Start** or **Space** to begin the simulation
4. **Adjust speed** with the slider for faster/slower evolution
5. **Save interesting patterns** for later (Ctrl+S)

## Documentation

- User Guide: docs/USER_GUIDE.md
- Development Guide: docs/DEVELOPMENT.md
- License: LICENSE (MIT)

### Pattern Highlights:

#### Conway's Game of Life:
- **Glider Gun** - Creates an endless stream of gliders (discovered by Bill Gosper)
- **R-Pentomino** - Tiny pattern that evolves for 1,103 generations
- **Acorn** - Takes 5,206 generations to stabilize from just 7 cells
- **Random Soup** - Creates unpredictable, chaotic patterns

#### High Life:
- **Replicator** - Unique pattern that copies itself (only exists in High Life rules)

#### Immigration/Rainbow Games:
- **Color Mix** - Pre-set colored patterns that interact
- **Random Soup** - Watch colors compete and blend

## How It Works

### Conway's Game of Life Rules:
- Any live cell with 2 or 3 live neighbors survives
- Any dead cell with exactly 3 live neighbors becomes alive
- All other cells die or stay dead

### High Life Rules (B36/S23):
- Birth on 3 or 6 neighbors
- Survival on 2 or 3 neighbors

### Immigration Game:
- Follows Conway's rules but maintains cell colors
- New cells inherit average color from parents
- Creates beautiful color competition dynamics

### Rainbow Game:
- 6 distinct colors create flowing patterns
- Color averaging produces smooth transitions

### Langton's Ant Rules:
- At a white square: turn right, flip color, move forward
- At a black square: turn left, flip color, move forward

## Performance

The application uses highly optimized algorithms:

- **Convolution-based neighbor counting** replaces slow nested loops
- **Differential rendering** updates only changed cells
- Can handle 100×80 grids at 60+ FPS on modern hardware
- Larger grids possible with adjusted cell size

## Customization

Edit these variables in the code to customize grid size:
- `self.grid_width` - Number of cells horizontally (default: 100)
- `self.grid_height` - Number of cells vertically (default: 80)
- `self.cell_size` - Size of each cell in pixels (default: 8)

## File Format

Saved patterns use JSON format:
```json
{
  "mode": "Conway's Game of Life",
  "grid": [[0, 1, 0], [0, 1, 0], ...],
  "width": 100,
  "height": 80,
  "generation": 42,
  "rules": "B3/S23"
}
```

The `rules` field is included when saving Custom Rules mode patterns.

## Credits

- Conway's Game of Life: John Conway (1970)
- Langton's Ant: Chris Langton (1986)
- High Life rules: Discovered by various researchers
- Implementation: Enhanced with modern optimization techniques

## License

Free to use and modify.
