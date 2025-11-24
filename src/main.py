#!/usr/bin/env python3
"""Tkinter GUI entry point for the cellular automaton simulator."""

# pylint: disable=too-many-instance-attributes,too-many-statements
# pylint: disable=missing-function-docstring

from __future__ import annotations

import json
from collections import deque
from typing import Callable, Dict, Iterable, List

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np

try:
    from PIL import Image

    PIL_AVAILABLE = True
    PIL_NEAREST = getattr(getattr(Image, "Resampling", Image), "NEAREST")
except ImportError:
    PIL_AVAILABLE = False
    PIL_NEAREST = None

from automata import (
    CellularAutomaton,
    ConwayGameOfLife,
    HighLife,
    ImmigrationGame,
    LangtonsAnt,
    LifeLikeAutomaton,
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


class CellularAutomatonGUI:
    """Main GUI orchestrating simulation state and rendering."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Cellular Automaton Simulator")

        # Grid configuration
        self.grid_width = 100
        self.grid_height = 100
        self.cell_size = 8

        # Runtime state
        self.running = False
        self.current_automaton: CellularAutomaton | None = None
        self.generation = 0

        # Custom rules data
        self.custom_birth = set(DEFAULT_CUSTOM_BIRTH)
        self.custom_survival = set(DEFAULT_CUSTOM_SURVIVAL)

        # Population tracking for quick stats
        self.population_history: deque[int] = deque(maxlen=500)
        self.population_peak = 0

        # Tk variables
        self.mode_var = tk.StringVar(value="Conway's Game of Life")
        self.pattern_var = tk.StringVar(value="Classic Mix")
        self.speed_var = tk.IntVar(value=50)
        self.grid_size_var = tk.StringVar(value="100x100")
        self.custom_width = tk.IntVar(value=100)
        self.custom_height = tk.IntVar(value=100)
        self.draw_mode_var = tk.StringVar(value="toggle")
        self.symmetry_var = tk.StringVar(value="None")
        self.show_grid = True

        # Widgets populated in helpers
        self.birth_entry: tk.Entry | None = None
        self.survival_entry: tk.Entry | None = None
        self.apply_rules_button: tk.Button | None = None
        self.pattern_combo: ttk.Combobox | None = None
        self.start_button: tk.Button | None = None
        self.gen_label: tk.Label | None = None
        self.population_label: tk.Label | None = None
        self.canvas: tk.Canvas | None = None

        self._create_widgets()
        self._configure_bindings()
        self.switch_mode(self.mode_var.get())
        self._update_widgets_enabled_state()
        self._update_display()

    # ------------------------------------------------------------------
    # Widget / bindings setup
    # ------------------------------------------------------------------
    def _create_widgets(self) -> None:
        control_frame = tk.Frame(self.root, pady=10)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10)

        # Row 0: Mode selection + primary controls
        tk.Label(
            control_frame,
            text="Mode:",
        ).grid(row=0, column=0, padx=5, sticky=tk.E)
        mode_combo = ttk.Combobox(
            control_frame,
            textvariable=self.mode_var,
            state="readonly",
            width=20,
            values=list(MODE_PATTERNS.keys()),
        )
        mode_combo.grid(row=0, column=1, padx=5)
        mode_combo.bind(
            "<<ComboboxSelected>>",
            lambda _event: self.switch_mode(self.mode_var.get()),
        )

        self.start_button = tk.Button(
            control_frame,
            text="Start",
            command=self.toggle_simulation,
            width=10,
            bg="#4caf50",
            fg="white",
        )
        self.start_button.grid(row=0, column=2, padx=5)
        tk.Button(
            control_frame,
            text="Step",
            command=self.step_once,
            width=8,
        ).grid(row=0, column=3, padx=5)
        tk.Button(
            control_frame,
            text="Clear",
            command=self.clear_grid,
            width=8,
            bg="#f44336",
            fg="white",
        ).grid(row=0, column=4, padx=5)
        tk.Button(
            control_frame,
            text="Reset",
            command=self.reset_simulation,
            width=8,
        ).grid(row=0, column=5, padx=5)

        # Row 1: Pattern and IO controls
        tk.Label(
            control_frame,
            text="Pattern:",
        ).grid(row=1, column=0, padx=5, sticky=tk.E)
        self.pattern_combo = ttk.Combobox(
            control_frame,
            textvariable=self.pattern_var,
            state="readonly",
            width=20,
        )
        self.pattern_combo.grid(row=1, column=1, padx=5)
        self.pattern_combo.bind(
            "<<ComboboxSelected>>", lambda _event: self.load_pattern_handler()
        )
        tk.Button(
            control_frame,
            text="Save",
            command=self.save_pattern,
            width=8,
        ).grid(row=1, column=2, padx=5)
        tk.Button(
            control_frame,
            text="Load File",
            command=self.load_saved_pattern,
            width=10,
        ).grid(row=1, column=3, padx=5)
        if PIL_AVAILABLE:
            tk.Button(
                control_frame,
                text="Export PNG",
                command=self.export_png,
                width=12,
            ).grid(row=1, column=4, padx=5, columnspan=2)

        # Row 2: Custom rules (enabled only for Custom mode)
        tk.Label(
            control_frame,
            text="Custom B/S:",
        ).grid(row=2, column=0, padx=5, sticky=tk.E)
        tk.Label(control_frame, text="B:").grid(row=2, column=1, sticky=tk.W)
        self.birth_entry = tk.Entry(control_frame, width=8)
        self.birth_entry.grid(row=2, column=1, padx=(20, 5), sticky=tk.W)
        self.birth_entry.insert(
            0,
            "".join(str(n) for n in sorted(self.custom_birth)),
        )

        tk.Label(control_frame, text="S:").grid(row=2, column=2, sticky=tk.W)
        self.survival_entry = tk.Entry(control_frame, width=8)
        self.survival_entry.grid(row=2, column=2, padx=(20, 5), sticky=tk.W)
        self.survival_entry.insert(
            0,
            "".join(str(n) for n in sorted(self.custom_survival)),
        )

        self.apply_rules_button = tk.Button(
            control_frame,
            text="Apply Rules",
            command=self.apply_custom_rules,
            width=12,
        )
        self.apply_rules_button.grid(row=2, column=3, padx=5)

        # Row 3: Grid size controls
        tk.Label(
            control_frame,
            text="Grid Size:",
        ).grid(row=3, column=0, padx=5, sticky=tk.E)
        size_combo = ttk.Combobox(
            control_frame,
            textvariable=self.grid_size_var,
            state="readonly",
            width=12,
            values=["50x50", "100x100", "150x150", "200x200", "Custom"],
        )
        size_combo.grid(row=3, column=1, padx=5, sticky=tk.W)
        size_combo.bind("<<ComboboxSelected>>", self.on_size_preset_change)

        tk.Label(control_frame, text="W:").grid(row=3, column=2, sticky=tk.W)
        width_spinbox = tk.Spinbox(
            control_frame,
            from_=10,
            to=500,
            textvariable=self.custom_width,
            width=6,
        )
        width_spinbox.grid(row=3, column=2, padx=(20, 2), sticky=tk.W)
        tk.Label(control_frame, text="H:").grid(row=3, column=3, sticky=tk.W)
        height_spinbox = tk.Spinbox(
            control_frame,
            from_=10,
            to=500,
            textvariable=self.custom_height,
            width=6,
        )
        height_spinbox.grid(row=3, column=3, padx=(20, 2), sticky=tk.W)
        tk.Button(
            control_frame,
            text="Apply",
            command=self.apply_custom_grid_size,
            width=8,
        ).grid(row=3, column=4, padx=5)

        # Row 4: Drawing tools
        tk.Label(
            control_frame,
            text="Draw:",
        ).grid(row=4, column=0, padx=5, sticky=tk.E)
        tk.Radiobutton(
            control_frame,
            text="Toggle",
            variable=self.draw_mode_var,
            value="toggle",
        ).grid(row=4, column=1, sticky=tk.W)
        tk.Radiobutton(
            control_frame,
            text="Pen",
            variable=self.draw_mode_var,
            value="pen",
        ).grid(row=4, column=2, sticky=tk.W)
        tk.Radiobutton(
            control_frame,
            text="Eraser",
            variable=self.draw_mode_var,
            value="eraser",
        ).grid(row=4, column=3, sticky=tk.W)

        tk.Label(
            control_frame,
            text="Symmetry:",
        ).grid(row=4, column=4, sticky=tk.E)
        symmetry_combo = ttk.Combobox(
            control_frame,
            textvariable=self.symmetry_var,
            state="readonly",
            width=12,
            values=["None", "Horizontal", "Vertical", "Both", "Radial"],
        )
        symmetry_combo.grid(row=4, column=5, padx=5)

        # Row 5: Speed and generation
        tk.Label(
            control_frame,
            text="Speed:",
        ).grid(row=5, column=0, padx=5, sticky=tk.E)
        speed_slider = tk.Scale(
            control_frame,
            from_=1,
            to=100,
            orient=tk.HORIZONTAL,
            variable=self.speed_var,
            length=150,
        )
        speed_slider.grid(row=5, column=1, columnspan=2, sticky=tk.W, padx=5)

        tk.Button(
            control_frame,
            text="Toggle Grid",
            command=self.toggle_grid,
            width=12,
        ).grid(row=5, column=3, padx=5)

        self.gen_label = tk.Label(
            control_frame,
            text="Generation: 0",
            font=("Arial", 10, "bold"),
        )
        self.gen_label.grid(row=5, column=4, columnspan=2, padx=5)

        # Row 6: Population stats
        stats_frame = tk.Frame(control_frame)
        stats_frame.grid(
            row=6,
            column=0,
            columnspan=6,
            sticky=tk.W,
            pady=(8, 0),
        )
        self.population_label = tk.Label(
            stats_frame,
            text="Live: 0 | Δ: +0 | Peak: 0 | Density: 0.0%",
        )
        self.population_label.pack(side=tk.LEFT)

        # Canvas + scrollbars
        canvas_frame = tk.Frame(self.root)
        canvas_frame.pack(
            side=tk.TOP,
            fill=tk.BOTH,
            expand=True,
            padx=10,
            pady=10,
        )
        self.canvas = tk.Canvas(
            canvas_frame,
            bg="white",
            width=800,
            height=600,
        )
        h_scroll = tk.Scrollbar(
            canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview
        )
        v_scroll = tk.Scrollbar(
            canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview
        )
        self.canvas.configure(
            xscrollcommand=h_scroll.set,
            yscrollcommand=v_scroll.set,
        )

        self.canvas.grid(row=0, column=0, sticky=tk.NSEW)
        h_scroll.grid(row=1, column=0, sticky=tk.EW)
        v_scroll.grid(row=0, column=1, sticky=tk.NS)
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)

        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)

    def _configure_bindings(self) -> None:
        self.root.bind("<space>", lambda event: self.toggle_simulation())
        self.root.bind("<Key-s>", lambda event: self.step_once())
        self.root.bind("<Key-S>", lambda event: self.step_once())
        self.root.bind("<Key-c>", lambda event: self.clear_grid())
        self.root.bind("<Key-C>", lambda event: self.clear_grid())
        self.root.bind("<Key-g>", lambda event: self.toggle_grid())
        self.root.bind("<Key-G>", lambda event: self.toggle_grid())

    def _update_widgets_enabled_state(self) -> None:
        is_custom = self.mode_var.get() == "Custom Rules"
        state = tk.NORMAL if is_custom else tk.DISABLED
        for widget in (
            self.birth_entry,
            self.survival_entry,
            self.apply_rules_button,
        ):
            if widget is not None:
                widget.configure(state=state)

    # ------------------------------------------------------------------
    # Automaton control
    # ------------------------------------------------------------------
    def switch_mode(self, mode_name: str) -> None:
        self.stop_simulation()
        if mode_name == "Custom Rules":
            self.current_automaton = LifeLikeAutomaton(
                self.grid_width,
                self.grid_height,
                self.custom_birth,
                self.custom_survival,
            )
        else:
            factory = MODE_FACTORIES.get(mode_name)
            if factory is None:
                raise ValueError(f"Unsupported mode: {mode_name}")
            self.current_automaton = factory(self.grid_width, self.grid_height)

        patterns = MODE_PATTERNS.get(mode_name, ["Empty"])
        if self.pattern_combo is not None:
            self.pattern_combo["values"] = patterns
            self.pattern_var.set(patterns[0])

        if (
            patterns
            and patterns[0] != "Empty"
            and hasattr(self.current_automaton, "load_pattern")
        ):
            self.current_automaton.load_pattern(  # type: ignore[attr-defined]
                patterns[0]
            )

        if mode_name == "Custom Rules":
            if self.birth_entry and self.survival_entry:
                self.birth_entry.delete(0, tk.END)
            self.birth_entry.insert(
                0,
                "".join(str(n) for n in sorted(self.custom_birth)),
            )
            self.survival_entry.delete(0, tk.END)
            self.survival_entry.insert(
                0,
                "".join(str(n) for n in sorted(self.custom_survival)),
            )

        self._reset_generation()
        self._update_widgets_enabled_state()
        self._update_display()

    def load_pattern_handler(self) -> None:
        if not self.current_automaton:
            return
        pattern_name = self.pattern_var.get()
        if pattern_name == "Empty":
            self.current_automaton.reset()
        elif hasattr(self.current_automaton, "load_pattern"):
            self.current_automaton.load_pattern(  # type: ignore[attr-defined]
                pattern_name
            )
        self._reset_generation()
        self._update_display()

    def toggle_simulation(self) -> None:
        self.running = not self.running
        if not self.start_button:
            return
        if self.running:
            self.start_button.config(text="Stop", bg="#ff9800")
            self.root.after(0, self._run_simulation_loop)
        else:
            self.start_button.config(text="Start", bg="#4caf50")

    def stop_simulation(self) -> None:
        self.running = False
        if self.start_button:
            self.start_button.config(text="Start", bg="#4caf50")

    def _run_simulation_loop(self) -> None:
        if not self.running:
            return
        self.step_once()
        delay = max(10, 1010 - self.speed_var.get() * 10)
        self.root.after(delay, self._run_simulation_loop)

    def step_once(self) -> None:
        if not self.current_automaton:
            return
        self.current_automaton.step()
        self.generation += 1
        if self.gen_label:
            self.gen_label.config(text=f"Generation: {self.generation}")
        self._update_display()

    def reset_simulation(self) -> None:
        if not self.current_automaton:
            return
        self.stop_simulation()
        self.current_automaton.reset()
        self._reset_generation()
        self._update_display()

    def clear_grid(self) -> None:
        if not self.current_automaton:
            return
        self.stop_simulation()
        self.current_automaton.reset()
        self._reset_generation()
        self._update_display()

    def apply_custom_rules(self) -> None:
        if not isinstance(self.current_automaton, LifeLikeAutomaton):
            messagebox.showinfo(
                "Not Custom Mode",
                "Switch to Custom Rules to apply B/S settings.",
            )
            return
        if not self.birth_entry or not self.survival_entry:
            return
        try:
            birth = {int(ch) for ch in self.birth_entry.get().strip() if ch.isdigit()}
            survival = {
                int(ch) for ch in self.survival_entry.get().strip() if ch.isdigit()
            }
        except ValueError as exc:
            messagebox.showerror(
                "Invalid Input",
                f"Failed to parse rules: {exc}",
            )
            return
        self.custom_birth = set(birth)
        self.custom_survival = set(survival)
        self.current_automaton.set_rules(
            self.custom_birth,
            self.custom_survival,
        )
        self.current_automaton.reset()
        self._reset_generation()
        self._update_display()

    # ------------------------------------------------------------------
    # Grid size helpers
    # ------------------------------------------------------------------
    def on_size_preset_change(self, _event: object) -> None:
        preset = self.grid_size_var.get()
        if preset == "Custom":
            return
        try:
            width_str, height_str = preset.split("x", 1)
            width = int(width_str)
            height = int(height_str)
        except ValueError:
            messagebox.showerror(
                "Invalid size",
                f"Could not parse preset '{preset}'.",
            )
            return
        self.resize_grid(width, height)

    def apply_custom_grid_size(self) -> None:
        self.resize_grid(self.custom_width.get(), self.custom_height.get())

    def resize_grid(self, width: int, height: int) -> None:
        width = max(10, min(width, 500))
        height = max(10, min(height, 500))
        self.grid_width = width
        self.grid_height = height
        current_mode = self.mode_var.get()
        self.switch_mode(current_mode)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_pattern(self) -> None:
        if not self.current_automaton:
            return
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not filename:
            return
        grid = self.current_automaton.get_grid()
        payload: Dict[str, object] = {
            "mode": self.mode_var.get(),
            "width": self.grid_width,
            "height": self.grid_height,
            "grid": grid.tolist(),
        }
        if isinstance(self.current_automaton, LifeLikeAutomaton):
            payload["birth"] = sorted(self.current_automaton.birth)
            payload["survival"] = sorted(self.current_automaton.survival)
        try:
            with open(filename, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            messagebox.showinfo("Saved", "Pattern saved successfully.")
        except OSError as exc:
            messagebox.showerror(
                "Save Failed",
                f"Could not save pattern: {exc}",
            )

    def load_saved_pattern(self) -> None:
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not filename:
            return
        try:
            with open(filename, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            messagebox.showerror(
                "Load Failed",
                f"Could not read pattern: {exc}",
            )
            return

        mode = data.get("mode", "Conway's Game of Life")
        width = int(data.get("width", self.grid_width))
        height = int(data.get("height", self.grid_height))
        grid_data = np.array(data.get("grid", []), dtype=int)

        # Resize and switch mode first
        self.grid_width = width
        self.grid_height = height
        self.mode_var.set(mode)
        self.switch_mode(mode)

        if isinstance(self.current_automaton, LifeLikeAutomaton):
            birth = data.get("birth")
            survival = data.get("survival")
            if isinstance(birth, Iterable) and isinstance(survival, Iterable):
                birth_set = {int(value) for value in birth}
                survival_set = {int(value) for value in survival}
                self.custom_birth = birth_set
                self.custom_survival = survival_set
                self.current_automaton.set_rules(birth_set, survival_set)
                if self.birth_entry and self.survival_entry:
                    self.birth_entry.delete(0, tk.END)
                    self.birth_entry.insert(
                        0, "".join(str(n) for n in sorted(birth_set))
                    )
                    self.survival_entry.delete(0, tk.END)
                    self.survival_entry.insert(
                        0, "".join(str(n) for n in sorted(survival_set))
                    )

        if grid_data.size:
            expected_shape = (self.grid_height, self.grid_width)
            try:
                self.current_automaton.grid = grid_data.reshape(expected_shape)
            except ValueError:
                messagebox.showwarning(
                    "Shape Mismatch",
                    (
                        "Saved grid size did not match current settings. "
                        "Resetting grid."
                    ),
                )
                self.current_automaton.reset()
        self._reset_generation()
        self._update_display()
        messagebox.showinfo("Loaded", "Pattern loaded successfully.")

    def export_png(self) -> None:
        if not (PIL_AVAILABLE and self.current_automaton):
            messagebox.showerror(
                "Unavailable",
                "Pillow is required for PNG export.",
            )
            return
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
        )
        if not filename:
            return
        grid = self.current_automaton.get_grid()
        image = Image.new("RGB", (self.grid_width, self.grid_height), "white")
        pixels = image.load()
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                value = int(grid[y, x])
                pixels[x, y] = EXPORT_COLOR_MAP.get(value, (0, 0, 0))
        scale = max(1, 800 // max(self.grid_width, self.grid_height))
        image = image.resize(
            (self.grid_width * scale, self.grid_height * scale),
            PIL_NEAREST,
        )
        try:
            image.save(filename)
            messagebox.showinfo("Exported", f"PNG saved to {filename}")
        except OSError as exc:
            messagebox.showerror("Export Failed", f"Could not save PNG: {exc}")

    # ------------------------------------------------------------------
    # Rendering and drawing interactions
    # ------------------------------------------------------------------
    def _update_display(self) -> None:
        if not (self.current_automaton and self.canvas):
            return
        grid = self.current_automaton.get_grid()
        width = self.grid_width * self.cell_size
        height = self.grid_height * self.cell_size
        self.canvas.delete("all")
        self.canvas.configure(scrollregion=(0, 0, width, height))
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                x1 = x * self.cell_size
                y1 = y * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                color = CELL_COLORS.get(int(grid[y, x]), "white")
                outline = "gray" if getattr(self, "show_grid", True) else ""
                self.canvas.create_rectangle(
                    x1, y1, x2, y2, fill=color, outline=outline, width=1
                )
        self._update_statistics(grid)

    def toggle_grid(self) -> None:
        self.show_grid = not getattr(self, "show_grid", True)
        self._update_display()

    def on_canvas_click(self, event: tk.Event[tk.Misc]) -> None:
        self._handle_canvas_interaction(event)

    def on_canvas_drag(self, event: tk.Event[tk.Misc]) -> None:
        self._handle_canvas_interaction(event)

    def _handle_canvas_interaction(self, event: tk.Event[tk.Misc]) -> None:
        if not (self.canvas and self.current_automaton):
            return
        x = int(self.canvas.canvasx(event.x) // self.cell_size)
        y = int(self.canvas.canvasy(event.y) // self.cell_size)
        if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
            self._apply_draw_action(x, y)
            self._update_display()

    def _apply_draw_action(self, x: int, y: int) -> None:
        if not self.current_automaton:
            return
        mode = self.draw_mode_var.get()
        symmetry = self.symmetry_var.get()
        positions = self._symmetry_positions(x, y, symmetry)
        for px, py in positions:
            if not (0 <= px < self.grid_width and 0 <= py < self.grid_height):
                continue
            if mode == "toggle":
                self.current_automaton.handle_click(px, py)
            elif mode == "pen":
                self.current_automaton.grid[py, px] = 1
            elif mode == "eraser":
                self.current_automaton.grid[py, px] = 0

    def _symmetry_positions(
        self, x: int, y: int, symmetry: str
    ) -> List[tuple[int, int]]:
        positions = {(x, y)}
        if symmetry in ("Horizontal", "Both"):
            positions.add((self.grid_width - 1 - x, y))
        if symmetry in ("Vertical", "Both"):
            positions.add((x, self.grid_height - 1 - y))
        if symmetry == "Both":
            positions.add((self.grid_width - 1 - x, self.grid_height - 1 - y))
        if symmetry == "Radial":
            cx, cy = self.grid_width // 2, self.grid_height // 2
            dx, dy = x - cx, y - cy
            radial = {
                (cx + dx, cy + dy),
                (cx - dx, cy - dy),
                (cx - dy, cy + dx),
                (cx + dy, cy - dx),
            }
            positions.update(radial)
        return list(positions)

    # ------------------------------------------------------------------
    # Statistics helpers
    # ------------------------------------------------------------------
    def _reset_generation(self) -> None:
        self.generation = 0
        if self.gen_label:
            self.gen_label.config(text="Generation: 0")
        self.population_history.clear()
        self.population_peak = 0

    def _update_statistics(self, grid: np.ndarray) -> None:
        live_cells = int(np.count_nonzero(grid))
        if self.population_history and self.population_history[-1] == live_cells:
            delta = 0
        else:
            previous = self.population_history[-1] if self.population_history else 0
            delta = live_cells - previous
            self.population_history.append(live_cells)
        self.population_peak = max(self.population_peak, live_cells)
        total = grid.size if grid.size else 1
        density = (live_cells / total) * 100
        if self.population_label:
            stats_text = (
                f"Live: {live_cells} | Δ: {delta:+d} | Peak: "
                f"{self.population_peak} | Density: {density:.1f}%"
            )
            self.population_label.config(text=stats_text)


def main() -> None:
    root = tk.Tk()
    CellularAutomatonGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
