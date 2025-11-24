"""Refactored GUI application composed of focused helper modules."""

from __future__ import annotations

import json
from typing import Iterable

import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np

from automata import LifeLikeAutomaton

from .config import (
    DEFAULT_CUSTOM_BIRTH,
    DEFAULT_CUSTOM_SURVIVAL,
    EXPORT_COLOR_MAP,
    MAX_GRID_SIZE,
    MIN_GRID_SIZE,
    MODE_FACTORIES,
    MODE_PATTERNS,
)
from .rendering import draw_grid, symmetry_positions
from .state import SimulationState
from .ui import Callbacks, TkVars, Widgets, build_ui

try:
    from PIL import Image

    PIL_AVAILABLE = True
    PIL_NEAREST = getattr(getattr(Image, "Resampling", Image), "NEAREST")
except ImportError:
    Image = None
    PIL_AVAILABLE = False
    PIL_NEAREST = None


class AutomatonApp:
    """High-level GUI orchestrator for the cellular automaton simulator."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Cellular Automaton Simulator")

        self.state = SimulationState()
        self.custom_birth = set(DEFAULT_CUSTOM_BIRTH)
        self.custom_survival = set(DEFAULT_CUSTOM_SURVIVAL)

        self.tk_vars: TkVars = self._create_variables()
        callbacks = Callbacks(
            switch_mode=self.switch_mode,
            step_once=self.step_once,
            clear_grid=self.clear_grid,
            reset_simulation=self.reset_simulation,
            load_pattern=self.load_pattern_handler,
            save_pattern=self.save_pattern,
            load_saved_pattern=self.load_saved_pattern,
            export_png=self.export_png,
            apply_custom_rules=self.apply_custom_rules,
            size_preset_changed=self.on_size_preset_change,
            apply_custom_size=self.apply_custom_grid_size,
            toggle_grid=self.toggle_grid,
            on_canvas_click=self.on_canvas_click,
            on_canvas_drag=self.on_canvas_drag,
        )
        self.widgets: Widgets = build_ui(
            root=self.root,
            variables=self.tk_vars,
            callbacks=callbacks,
            show_export=PIL_AVAILABLE,
        )
        self.widgets.start_button.configure(command=self.toggle_simulation)
        self._widgets_init_defaults()

        self._configure_bindings()
        self.switch_mode(self.tk_vars.mode.get())
        self._update_widgets_enabled_state()
        self._update_display()

    # ------------------------------------------------------------------
    # Variable and widget helpers
    # ------------------------------------------------------------------
    def _create_variables(self) -> TkVars:
        return TkVars(
            mode=tk.StringVar(value="Conway's Game of Life"),
            pattern=tk.StringVar(value="Classic Mix"),
            speed=tk.IntVar(value=50),
            grid_size=tk.StringVar(value="100x100"),
            custom_width=tk.IntVar(value=100),
            custom_height=tk.IntVar(value=100),
            draw_mode=tk.StringVar(value="toggle"),
            symmetry=tk.StringVar(value="None"),
        )

    def _widgets_init_defaults(self) -> None:
        birth_values = "".join(str(n) for n in sorted(self.custom_birth))
        survival_values = "".join(str(n) for n in sorted(self.custom_survival))
        self.widgets.birth_entry.insert(0, birth_values)
        self.widgets.survival_entry.insert(0, survival_values)

    def _configure_bindings(self) -> None:
        self.root.bind("<space>", lambda _event: self.toggle_simulation())
        self.root.bind("<Key-s>", lambda _event: self.step_once())
        self.root.bind("<Key-S>", lambda _event: self.step_once())
        self.root.bind("<Key-c>", lambda _event: self.clear_grid())
        self.root.bind("<Key-C>", lambda _event: self.clear_grid())
        self.root.bind("<Key-g>", lambda _event: self.toggle_grid())
        self.root.bind("<Key-G>", lambda _event: self.toggle_grid())

    def _update_widgets_enabled_state(self) -> None:
        is_custom = self.tk_vars.mode.get() == "Custom Rules"
        state = tk.NORMAL if is_custom else tk.DISABLED
        for widget in (
            self.widgets.birth_entry,
            self.widgets.survival_entry,
            self.widgets.apply_rules_button,
        ):
            widget.configure(state=state)

    # ------------------------------------------------------------------
    # Automaton control
    # ------------------------------------------------------------------
    def switch_mode(self, mode_name: str) -> None:
        self.stop_simulation()
        if mode_name == "Custom Rules":
            self.state.current_automaton = LifeLikeAutomaton(
                self.state.grid_width,
                self.state.grid_height,
                self.custom_birth,
                self.custom_survival,
            )
        else:
            factory = MODE_FACTORIES.get(mode_name)
            if factory is None:
                raise ValueError(f"Unsupported mode: {mode_name}")
            self.state.current_automaton = factory(
                self.state.grid_width,
                self.state.grid_height,
            )

        patterns = MODE_PATTERNS.get(mode_name, ["Empty"])
        self.widgets.pattern_combo["values"] = patterns
        self.tk_vars.pattern.set(patterns[0])

        automaton = self.state.current_automaton
        if patterns:
            first_pattern = patterns[0]
        else:
            first_pattern = "Empty"
        if first_pattern != "Empty" and hasattr(automaton, "load_pattern"):
            automaton.load_pattern(first_pattern)  # type: ignore[attr-defined]

        if mode_name == "Custom Rules":
            self._sync_custom_entries()

        self.state.reset_generation()
        self._update_generation_label()
        self._update_widgets_enabled_state()
        self._update_display()

    def _sync_custom_entries(self) -> None:
        birth_values = "".join(str(n) for n in sorted(self.custom_birth))
        survival_values = "".join(str(n) for n in sorted(self.custom_survival))
        self.widgets.birth_entry.delete(0, tk.END)
        self.widgets.birth_entry.insert(0, birth_values)
        self.widgets.survival_entry.delete(0, tk.END)
        self.widgets.survival_entry.insert(0, survival_values)

    def load_pattern_handler(self) -> None:
        automaton = self.state.current_automaton
        if not automaton:
            return
        pattern_name = self.tk_vars.pattern.get()
        if pattern_name == "Empty":
            automaton.reset()
        elif hasattr(automaton, "load_pattern"):
            automaton.load_pattern(pattern_name)  # type: ignore[attr-defined]
        self.state.reset_generation()
        self._update_generation_label()
        self._update_display()

    def toggle_simulation(self) -> None:
        self.state.running = not self.state.running
        if self.state.running:
            self.widgets.start_button.config(text="Stop", bg="#ff9800")
            self.root.after(0, self._run_simulation_loop)
        else:
            self.widgets.start_button.config(text="Start", bg="#4caf50")

    def stop_simulation(self) -> None:
        self.state.running = False
        self.widgets.start_button.config(text="Start", bg="#4caf50")

    def _run_simulation_loop(self) -> None:
        if not self.state.running:
            return
        self.step_once()
        delay = max(10, 1010 - self.tk_vars.speed.get() * 10)
        self.root.after(delay, self._run_simulation_loop)

    def step_once(self) -> None:
        automaton = self.state.current_automaton
        if not automaton:
            return
        automaton.step()
        self.state.generation += 1
        self._update_generation_label()
        self._update_display()

    def _update_generation_label(self) -> None:
        self.widgets.gen_label.config(text=f"Generation: {self.state.generation}")

    def reset_simulation(self) -> None:
        automaton = self.state.current_automaton
        if not automaton:
            return
        self.stop_simulation()
        automaton.reset()
        self.state.reset_generation()
        self._update_generation_label()
        self._update_display()

    def clear_grid(self) -> None:
        automaton = self.state.current_automaton
        if not automaton:
            return
        self.stop_simulation()
        automaton.reset()
        self.state.reset_generation()
        self._update_generation_label()
        self._update_display()

    def apply_custom_rules(self) -> None:
        automaton = self.state.current_automaton
        if not isinstance(automaton, LifeLikeAutomaton):
            messagebox.showinfo(
                "Not Custom Mode",
                "Switch to Custom Rules to apply B/S settings.",
            )
            return
        try:
            birth = self.widgets.birth_entry.get().strip()
            survival = self.widgets.survival_entry.get().strip()
            birth_set = {int(ch) for ch in birth if ch.isdigit()}
            survival_set = {int(ch) for ch in survival if ch.isdigit()}
        except ValueError as exc:
            messagebox.showerror(
                "Invalid Input",
                f"Failed to parse rules: {exc}",
            )
            return
        self.custom_birth = set(birth_set)
        self.custom_survival = set(survival_set)
        automaton.set_rules(self.custom_birth, self.custom_survival)
        automaton.reset()
        self.state.reset_generation()
        self._update_generation_label()
        self._update_display()

    # ------------------------------------------------------------------
    # Grid size helpers
    # ------------------------------------------------------------------
    def on_size_preset_change(self, _event: tk.Event[tk.Misc]) -> None:
        preset = self.tk_vars.grid_size.get()
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
        self.resize_grid(
            self.tk_vars.custom_width.get(),
            self.tk_vars.custom_height.get(),
        )

    def resize_grid(self, width: int, height: int) -> None:
        width = max(MIN_GRID_SIZE, min(width, MAX_GRID_SIZE))
        height = max(MIN_GRID_SIZE, min(height, MAX_GRID_SIZE))
        self.state.grid_width = width
        self.state.grid_height = height
        self.state.current_automaton = None
        self.switch_mode(self.tk_vars.mode.get())

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_pattern(self) -> None:
        automaton = self.state.current_automaton
        if not automaton:
            return
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not filename:
            return
        grid = automaton.get_grid()
        payload = {
            "mode": self.tk_vars.mode.get(),
            "width": self.state.grid_width,
            "height": self.state.grid_height,
            "grid": grid.tolist(),
        }
        if isinstance(automaton, LifeLikeAutomaton):
            payload["birth"] = sorted(automaton.birth)
            payload["survival"] = sorted(automaton.survival)
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
        width = int(data.get("width", self.state.grid_width))
        height = int(data.get("height", self.state.grid_height))
        grid_data = np.array(data.get("grid", []), dtype=int)

        self.state.grid_width = width
        self.state.grid_height = height
        self.tk_vars.mode.set(mode)
        self.switch_mode(mode)

        automaton = self.state.current_automaton
        if isinstance(automaton, LifeLikeAutomaton):
            birth = data.get("birth")
            survival = data.get("survival")
            if isinstance(birth, Iterable) and isinstance(survival, Iterable):
                birth_set = {int(value) for value in birth}
                survival_set = {int(value) for value in survival}
                self.custom_birth = birth_set
                self.custom_survival = survival_set
                automaton.set_rules(birth_set, survival_set)
                self._sync_custom_entries()

        if grid_data.size:
            expected_shape = (self.state.grid_height, self.state.grid_width)
            try:
                automaton.grid = grid_data.reshape(
                    expected_shape
                )  # type: ignore[attr-defined]
            except ValueError:
                messagebox.showwarning(
                    "Shape Mismatch",
                    (
                        "Saved grid size did not match current settings. "
                        "Resetting grid."
                    ),
                )
                automaton.reset()
        self.state.reset_generation()
        self._update_generation_label()
        self._update_display()
        messagebox.showinfo("Loaded", "Pattern loaded successfully.")

    def export_png(self) -> None:
        if not (PIL_AVAILABLE and self.state.current_automaton and Image):
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
        grid = self.state.current_automaton.get_grid()
        image = Image.new(
            "RGB",
            (self.state.grid_width, self.state.grid_height),
            "white",
        )
        pixels = image.load()
        for y in range(self.state.grid_height):
            for x in range(self.state.grid_width):
                value = int(grid[y, x])
                pixels[x, y] = EXPORT_COLOR_MAP.get(value, (0, 0, 0))
        max_dimension = max(
            self.state.grid_width,
            self.state.grid_height,
        )
        scale = max(1, 800 // max_dimension)
        image = image.resize(
            (self.state.grid_width * scale, self.state.grid_height * scale),
            PIL_NEAREST,
        )
        try:
            image.save(filename)
            messagebox.showinfo("Exported", f"PNG saved to {filename}")
        except OSError as exc:
            messagebox.showerror("Export Failed", f"Could not save PNG: {exc}")

    # ------------------------------------------------------------------
    # Rendering and interactions
    # ------------------------------------------------------------------
    def _update_display(self) -> None:
        automaton = self.state.current_automaton
        if not (automaton and self.widgets.canvas):
            return
        grid = automaton.get_grid()
        draw_grid(
            self.widgets.canvas,
            grid,
            self.state.cell_size,
            self.state.show_grid,
        )
        stats = self.state.update_population_stats(grid)
        self.widgets.population_label.config(text=stats)

    def toggle_grid(self) -> None:
        self.state.show_grid = not self.state.show_grid
        self._update_display()

    def on_canvas_click(self, event: tk.Event[tk.Misc]) -> None:
        self._handle_canvas_interaction(event)

    def on_canvas_drag(self, event: tk.Event[tk.Misc]) -> None:
        self._handle_canvas_interaction(event)

    def _handle_canvas_interaction(self, event: tk.Event[tk.Misc]) -> None:
        automaton = self.state.current_automaton
        if not (automaton and self.widgets.canvas):
            return
        canvas_x = self.widgets.canvas.canvasx(event.x)
        canvas_y = self.widgets.canvas.canvasy(event.y)
        x = int(canvas_x // self.state.cell_size)
        y = int(canvas_y // self.state.cell_size)
        if 0 <= x < self.state.grid_width and 0 <= y < self.state.grid_height:
            self._apply_draw_action(x, y)
            self._update_display()

    def _apply_draw_action(self, x: int, y: int) -> None:
        automaton = self.state.current_automaton
        if not automaton:
            return
        positions = symmetry_positions(
            x,
            y,
            self.state.grid_width,
            self.state.grid_height,
            self.tk_vars.symmetry.get(),
        )
        for px, py in positions:
            within_width = 0 <= px < self.state.grid_width
            within_height = 0 <= py < self.state.grid_height
            if not (within_width and within_height):
                continue
            if self.tk_vars.draw_mode.get() == "toggle":
                automaton.handle_click(px, py)
            elif self.tk_vars.draw_mode.get() == "pen":
                automaton.grid[py, px] = 1
            elif self.tk_vars.draw_mode.get() == "eraser":
                automaton.grid[py, px] = 0


def launch() -> None:
    root = tk.Tk()
    AutomatonApp(root)
    root.mainloop()
