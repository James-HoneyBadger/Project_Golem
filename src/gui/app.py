"""Refactored GUI application composed of focused helper modules."""

from __future__ import annotations

import json
from typing import Iterable

import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np

from automata import LifeLikeAutomaton

from .config import (
    DEFAULT_CANVAS_HEIGHT,
    DEFAULT_CANVAS_WIDTH,
    DEFAULT_CELL_SIZE,
    DEFAULT_CUSTOM_BIRTH,
    DEFAULT_CUSTOM_SURVIVAL,
    DEFAULT_SPEED,
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
except ImportError:
    Image = None
    PIL_AVAILABLE = False


def _nearest_resample_filter() -> object | None:
    """Return the Pillow nearest-neighbour filter if available."""

    if not (PIL_AVAILABLE and Image):
        return None
    resampling = getattr(Image, "Resampling", Image)
    return getattr(resampling, "NEAREST", None)


class AutomatonApp:
    """High-level GUI orchestrator for the cellular automaton simulator."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Project Golem")

        self.settings_file = "settings.json"
        self.settings = self._load_settings()

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

        self.state.show_grid = self.settings.get("show_grid", True)

        self._configure_bindings()
        self.switch_mode(self.tk_vars.mode.get())
        self._update_widgets_enabled_state()
        self._update_display()

        # Save settings on exit
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self) -> None:
        """Save settings and close the application."""
        self._save_settings()
        self.root.destroy()

    # ------------------------------------------------------------------
    # Variable and widget helpers
    # ------------------------------------------------------------------
    def _load_settings(self) -> dict:
        """Load user settings from file."""
        try:
            with open(self.settings_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save_settings(self) -> None:
        """Save current settings to file."""
        settings = {
            "mode": self.tk_vars.mode.get(),
            "pattern": self.tk_vars.pattern.get(),
            "speed": self.tk_vars.speed.get(),
            "grid_size": self.tk_vars.grid_size.get(),
            "custom_width": self.tk_vars.custom_width.get(),
            "custom_height": self.tk_vars.custom_height.get(),
            "draw_mode": self.tk_vars.draw_mode.get(),
            "symmetry": self.tk_vars.symmetry.get(),
            "show_grid": self.state.show_grid,
        }
        try:
            with open(self.settings_file, "w", encoding="utf-8") as f:
                json.dump(settings, f, indent=2)
        except OSError:
            pass  # Silently fail if can't save

    def _create_variables(self) -> TkVars:
        settings = self.settings
        return TkVars(
            mode=tk.StringVar(value=settings.get("mode", "Conway's Game of Life")),
            pattern=tk.StringVar(value=settings.get("pattern", "Classic Mix")),
            speed=tk.IntVar(value=settings.get("speed", DEFAULT_SPEED)),
            grid_size=tk.StringVar(value=settings.get("grid_size", "100x100")),
            custom_width=tk.IntVar(value=settings.get("custom_width", 100)),
            custom_height=tk.IntVar(value=settings.get("custom_height", 100)),
            draw_mode=tk.StringVar(value=settings.get("draw_mode", "toggle")),
            symmetry=tk.StringVar(value=settings.get("symmetry", "None")),
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
        """Switch to the requested automaton mode and refresh the grid."""

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
        """Mirror the active custom rule sets into the entry widgets."""

        birth_values = "".join(str(n) for n in sorted(self.custom_birth))
        survival_values = "".join(str(n) for n in sorted(self.custom_survival))
        self.widgets.birth_entry.delete(0, tk.END)
        self.widgets.birth_entry.insert(0, birth_values)
        self.widgets.survival_entry.delete(0, tk.END)
        self.widgets.survival_entry.insert(0, survival_values)

    def load_pattern_handler(self) -> None:
        """Load the currently selected pattern into the simulation grid."""

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
        """Start or pause the simulation loop."""

        self.state.running = not self.state.running
        if self.state.running:
            self.widgets.start_button.config(text="Stop", bg="#ff9800")
            self.root.after(0, self._run_simulation_loop)
        else:
            self.widgets.start_button.config(text="Start", bg="#4caf50")

    def stop_simulation(self) -> None:
        """Force the simulation into a stopped state."""

        self.state.running = False
        self.widgets.start_button.config(text="Start", bg="#4caf50")

    def _run_simulation_loop(self) -> None:
        """Advance the automaton while the simulation is marked running."""

        if not self.state.running:
            return
        self.step_once()
        delay = max(10, 1010 - self.tk_vars.speed.get() * 10)
        self.root.after(delay, self._run_simulation_loop)

    def step_once(self) -> None:
        """Advance the automaton by a single generation."""

        automaton = self.state.current_automaton
        if not automaton:
            return
        automaton.step()
        self.state.generation += 1
        self._update_generation_label()
        self._update_display()

    def _update_generation_label(self) -> None:
        generation_text = f"Generation: {self.state.generation}"
        self.widgets.gen_label.config(text=generation_text)

    def reset_simulation(self) -> None:
        """Reset the automaton grid to its starting state."""

        automaton = self.state.current_automaton
        if not automaton:
            return
        self.stop_simulation()
        automaton.reset()
        self.state.reset_generation()
        self._update_generation_label()
        self._update_display()

    def clear_grid(self) -> None:
        """Clear the grid and pause the simulation."""

        automaton = self.state.current_automaton
        if not automaton:
            return
        self.stop_simulation()
        automaton.reset()
        self.state.reset_generation()
        self._update_generation_label()
        self._update_display()

    def apply_custom_rules(self) -> None:
        """Apply custom birth/survival rule strings to the automaton."""

        automaton = self.state.current_automaton
        if not isinstance(automaton, LifeLikeAutomaton):
            messagebox.showinfo(
                "Not Custom Mode",
                "Switch to Custom Rules to apply B/S settings.",
            )
            return
        
        birth_text = self.widgets.birth_entry.get().strip()
        survival_text = self.widgets.survival_entry.get().strip()
        
        # Validate input
        if not birth_text and not survival_text:
            messagebox.showerror(
                "Invalid Input",
                "At least one of birth or survival rules must be specified.",
            )
            return
        
        try:
            birth_set = {int(ch) for ch in birth_text if ch.isdigit()}
            survival_set = {int(ch) for ch in survival_text if ch.isdigit()}
            
            # Check for valid neighbor counts (0-8)
            invalid_birth = birth_set - set(range(9))
            invalid_survival = survival_set - set(range(9))
            if invalid_birth or invalid_survival:
                invalid = sorted(invalid_birth | invalid_survival)
                messagebox.showerror(
                    "Invalid Input",
                    f"Neighbor counts must be between 0-8. Invalid: {invalid}",
                )
                return
                
        except ValueError as exc:
            messagebox.showerror(
                "Invalid Input",
                f"Failed to parse rules: {exc}",
            )
            return
        
        self.custom_birth = birth_set
        self.custom_survival = survival_set
        automaton.set_rules(self.custom_birth, self.custom_survival)
        automaton.reset()
        self.state.reset_generation()
        self._update_generation_label()
        self._update_display()
        
        messagebox.showinfo(
            "Rules Applied",
            f"Birth: {sorted(birth_set)}\nSurvival: {sorted(survival_set)}",
        )

    # ------------------------------------------------------------------
    # Grid size helpers
    # ------------------------------------------------------------------
    def on_size_preset_change(self, _event: tk.Event[tk.Misc]) -> None:
        """Resize the grid when a preset dimension is selected."""

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
        """Resize the grid based on custom width and height spinboxes."""

        self.resize_grid(
            self.tk_vars.custom_width.get(),
            self.tk_vars.custom_height.get(),
        )

    def resize_grid(self, width: int, height: int) -> None:
        """Clamp and apply a new grid size, rebuilding the automaton."""

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
        """Persist the current grid and rules to a JSON file."""

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
        """Load a pattern JSON file into the active automaton."""

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
                f"Could not read file '{filename}': {exc}",
            )
            return

        # Validate required fields
        required_fields = ["mode", "width", "height", "grid"]
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            messagebox.showerror(
                "Invalid File",
                f"Missing required fields: {missing_fields}",
            )
            return

        try:
            mode = data["mode"]
            width = int(data["width"])
            height = int(data["height"])
            grid_data = np.array(data["grid"], dtype=int)
        except (ValueError, KeyError) as exc:
            messagebox.showerror(
                "Invalid Data",
                f"Invalid data format: {exc}",
            )
            return

        # Validate dimensions
        if width < 10 or width > MAX_GRID_SIZE or height < 10 or height > MAX_GRID_SIZE:
            messagebox.showerror(
                "Invalid Size",
                f"Grid size must be between 10x10 and {MAX_GRID_SIZE}x{MAX_GRID_SIZE}",
            )
            return

        # Validate grid data
        expected_size = width * height
        if grid_data.size != expected_size:
            messagebox.showerror(
                "Invalid Grid",
                f"Grid data size ({grid_data.size}) doesn't match dimensions ({width}x{height} = {expected_size})",
            )
            return

        self.state.grid_width = width
        self.state.grid_height = height
        self.tk_vars.mode.set(mode)
        self.switch_mode(mode)

        automaton = self.state.current_automaton
        if isinstance(automaton, LifeLikeAutomaton):
            birth = data.get("birth", [])
            survival = data.get("survival", [])
            try:
                birth_set = {int(value) for value in birth}
                survival_set = {int(value) for value in survival}
                self.custom_birth = birth_set
                self.custom_survival = survival_set
                automaton.set_rules(birth_set, survival_set)
                self._sync_custom_entries()
            except (ValueError, TypeError):
                messagebox.showwarning(
                    "Invalid Rules",
                    "Could not load custom rules, using defaults.",
                )

        try:
            expected_shape = (self.state.grid_height, self.state.grid_width)
            automaton.grid = grid_data.reshape(expected_shape)
        except ValueError:
            messagebox.showwarning(
                "Shape Mismatch",
                "Saved grid size did not match current settings. Resetting grid.",
            )
            automaton.reset()
        
        self.state.reset_generation()
        self._update_generation_label()
        self._update_display()
        messagebox.showinfo("Loaded", f"Pattern loaded from {filename}")

    def export_png(self) -> None:
        """Export the current grid as a Pillow PNG image."""

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
            _nearest_resample_filter(),
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
        """Redraw the canvas and population statistics."""

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
        """Toggle grid line visibility and refresh the canvas."""

        self.state.show_grid = not self.state.show_grid
        self._update_display()

    def on_canvas_click(self, event: tk.Event[tk.Misc]) -> None:
        """Handle a canvas click based on the active draw mode."""

        self._handle_canvas_interaction(event)

    def on_canvas_drag(self, event: tk.Event[tk.Misc]) -> None:
        """Handle a canvas drag while the pointer button is held."""

        self._handle_canvas_interaction(event)

    def _handle_canvas_interaction(self, event: tk.Event[tk.Misc]) -> None:
        """Translate canvas coordinates into grid mutations."""

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
        """Apply the selected drawing action at the given grid coordinate."""

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
    """Create the Tk root window and start the simulator event loop."""

    root = tk.Tk()
    AutomatonApp(root)
    root.mainloop()
