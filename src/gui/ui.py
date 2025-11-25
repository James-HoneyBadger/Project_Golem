"""Widget construction and Tk variable helpers for the GUI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import tkinter as tk
from tkinter import ttk

from .config import DEFAULT_CANVAS_HEIGHT, DEFAULT_CANVAS_WIDTH, MODE_PATTERNS


class Tooltip:
    """Simple tooltip implementation for Tkinter widgets."""

    def __init__(self, widget: tk.Widget, text: str) -> None:
        self.widget = widget
        self.text = text
        self.tooltip_window = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event: tk.Event[tk.Misc]) -> None:
        """Display the tooltip near the widget."""
        if self.tooltip_window:
            return
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25
        self.tooltip_window = tk.Toplevel(self.widget)
        self.tooltip_window.wm_overrideredirect(True)
        self.tooltip_window.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            self.tooltip_window,
            text=self.text,
            background="#ffffe0",
            relief="solid",
            borderwidth=1,
            font=("Arial", 10),
        )
        label.pack()

    def hide_tooltip(self, event: tk.Event[tk.Misc]) -> None:
        """Hide the tooltip."""
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None


# pylint: disable=too-many-instance-attributes
@dataclass
class TkVars:


# pylint: disable=too-many-instance-attributes
@dataclass
class TkVars:
    """Container for the Tkinter variables shared across widgets."""

    mode: tk.StringVar
    pattern: tk.StringVar
    speed: tk.IntVar
    grid_size: tk.StringVar
    custom_width: tk.IntVar
    custom_height: tk.IntVar
    draw_mode: tk.StringVar
    symmetry: tk.StringVar


# pylint: disable=too-many-instance-attributes
@dataclass
class Widgets:
    """References to widgets that the application interacts with later."""

    start_button: tk.Button
    pattern_combo: ttk.Combobox
    birth_entry: tk.Entry
    survival_entry: tk.Entry
    apply_rules_button: tk.Button
    gen_label: tk.Label
    population_label: tk.Label
    canvas: tk.Canvas


# pylint: disable=too-many-instance-attributes
@dataclass
class Callbacks:
    """Callback definitions for UI events."""

    switch_mode: Callable[[str], None]
    step_once: Callable[[], None]
    clear_grid: Callable[[], None]
    reset_simulation: Callable[[], None]
    load_pattern: Callable[[], None]
    save_pattern: Callable[[], None]
    load_saved_pattern: Callable[[], None]
    export_png: Callable[[], None]
    apply_custom_rules: Callable[[], None]
    size_preset_changed: Callable[[tk.Event[tk.Misc]], None]
    apply_custom_size: Callable[[], None]
    toggle_grid: Callable[[], None]
    on_canvas_click: Callable[[tk.Event[tk.Misc]], None]
    on_canvas_drag: Callable[[tk.Event[tk.Misc]], None]


def build_ui(
    root: tk.Tk,
    variables: TkVars,
    callbacks: Callbacks,
    show_export: bool,
) -> Widgets:
    """Create the Tkinter widget layout and wire up callbacks."""

    _configure_style(root)
    sidebar, content = _create_layout(root)

    pattern_combo = _add_automaton_section(
        sidebar,
        variables,
        callbacks,
        show_export,
    )
    start_button, gen_label = _add_simulation_section(
        sidebar,
        variables,
        callbacks,
    )
    population_label = _add_population_section(sidebar)
    (
        birth_entry,
        survival_entry,
        apply_rules_button,
    ) = _add_custom_rules_section(sidebar, callbacks)
    _add_grid_section(sidebar, variables, callbacks)
    _add_drawing_section(sidebar, variables)
    canvas = _add_canvas_area(content, callbacks)

    return Widgets(
        start_button=start_button,
        pattern_combo=pattern_combo,
        birth_entry=birth_entry,
        survival_entry=survival_entry,
        apply_rules_button=apply_rules_button,
        gen_label=gen_label,
        population_label=population_label,
        canvas=canvas,
    )


def _configure_style(root: tk.Tk) -> None:
    """Apply a neutral ttk theme with subtle card styling."""

    style = ttk.Style(root)
    if "clam" in style.theme_names():
        style.theme_use("clam")
    style.configure("Card.TLabelframe", padding=8)
    style.configure("Card.TFrame", padding=8)


def _create_layout(root: tk.Tk) -> tuple[ttk.Frame, ttk.Frame]:
    """Create the shell layout and return the sidebar and content frames."""

    shell = ttk.Frame(root, padding=10)
    shell.pack(fill=tk.BOTH, expand=True)
    shell.columnconfigure(1, weight=1)
    shell.rowconfigure(0, weight=1)

    sidebar = ttk.Frame(shell, style="Card.TFrame")
    sidebar.grid(row=0, column=0, sticky="nsw", padx=(0, 10))

    content = ttk.Frame(shell)
    content.grid(row=0, column=1, sticky="nsew")
    content.rowconfigure(0, weight=1)
    content.columnconfigure(0, weight=1)

    return sidebar, content


def _add_automaton_section(
    parent: ttk.Frame,
    variables: TkVars,
    callbacks: Callbacks,
    show_export: bool,
) -> ttk.Combobox:
    """Build the automaton selection area and return the pattern combobox."""

    mode_frame = ttk.Labelframe(
        parent,
        text="Automaton",
        style="Card.TLabelframe",
    )
    mode_frame.pack(fill=tk.X, pady=(0, 10))

    ttk.Label(mode_frame, text="Mode").pack(anchor=tk.W)
    mode_combo = ttk.Combobox(
        mode_frame,
        textvariable=variables.mode,
        state="readonly",
        values=list(MODE_PATTERNS.keys()),
    )
    mode_combo.pack(fill=tk.X, pady=(2, 6))
    Tooltip(mode_combo, "Select the type of cellular automaton to simulate")
    mode_combo.bind(
        "<<ComboboxSelected>>",
        lambda _event: callbacks.switch_mode(variables.mode.get()),
    )

    ttk.Label(mode_frame, text="Pattern").pack(anchor=tk.W)
    pattern_combo = ttk.Combobox(
        mode_frame,
        textvariable=variables.pattern,
        state="readonly",
    )
    pattern_combo.pack(fill=tk.X, pady=(2, 6))
    Tooltip(pattern_combo, "Choose a preset pattern to load")
    pattern_combo.bind(
        "<<ComboboxSelected>>",
        lambda _event: callbacks.load_pattern(),
    )

    row = ttk.Frame(mode_frame)
    row.pack(fill=tk.X, pady=(4, 0))
    save_button = ttk.Button(row, text="Save", command=callbacks.save_pattern)
    save_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4))
    Tooltip(save_button, "Save the current grid as a JSON file")
    load_button = ttk.Button(row, text="Load", command=callbacks.load_saved_pattern)
    load_button.pack(side=tk.LEFT, expand=True, fill=tk.X)
    Tooltip(load_button, "Load a pattern from a JSON file")
    if show_export:
        export_button = ttk.Button(
            mode_frame,
            text="Export PNG",
            command=callbacks.export_png,
        )
        export_button.pack(fill=tk.X, pady=(6, 0))
        Tooltip(export_button, "Export the current grid as a PNG image")

    return pattern_combo


def _add_simulation_section(
    parent: ttk.Frame,
    variables: TkVars,
    callbacks: Callbacks,
) -> tuple[tk.Button, ttk.Label]:
    """Add simulation controls and return start button and generation label."""

    frame = ttk.Labelframe(
        parent,
        text="Simulation",
        style="Card.TLabelframe",
    )
    frame.pack(fill=tk.X, pady=(0, 10))

    toolbar = ttk.Frame(frame)
    toolbar.pack(fill=tk.X)

    start_button = tk.Button(
        toolbar,
        text="Start",
        command=lambda: None,
        width=9,
        bg="#4caf50",
        fg="white",
        relief=tk.FLAT,
    )
    start_button.pack(side=tk.LEFT, padx=(0, 6))
    Tooltip(start_button, "Start or stop the simulation (Space)")

    step_button = ttk.Button(
        toolbar,
        text="Step",
        command=callbacks.step_once,
        width=7,
    )
    step_button.pack(side=tk.LEFT)
    Tooltip(step_button, "Advance one generation (S)")

    clear_button = ttk.Button(
        toolbar,
        text="Clear",
        command=callbacks.clear_grid,
        width=7,
    )
    clear_button.pack(side=tk.LEFT, padx=(6, 0))
    Tooltip(clear_button, "Clear the grid (C)")

    reset_button = ttk.Button(
        toolbar,
        text="Reset",
        command=callbacks.reset_simulation,
        width=7,
    )
    reset_button.pack(side=tk.LEFT, padx=(6, 0))
    Tooltip(reset_button, "Reset to initial pattern")

    ttk.Label(frame, text="Speed").pack(anchor=tk.W, pady=(8, 2))
    speed_scale = tk.Scale(
        frame,
        from_=1,
        to=100,
        orient=tk.HORIZONTAL,
        variable=variables.speed,
        length=200,
        showvalue=False,
    )
    speed_scale.pack(fill=tk.X)
    Tooltip(speed_scale, "Adjust simulation speed (higher = faster)")

    grid_button = ttk.Button(frame, text="Toggle Grid", command=callbacks.toggle_grid)
    grid_button.pack(fill=tk.X, pady=(8, 0))
    Tooltip(grid_button, "Show/hide grid lines (G)")

    gen_label = ttk.Label(
        frame,
        text="Generation: 0",
        font=("Arial", 10, "bold"),
    )
    gen_label.pack(anchor=tk.W, pady=(8, 0))

    return start_button, gen_label


def _add_population_section(parent: ttk.Frame) -> ttk.Label:
    """Add the population stats card and return the label widget."""

    frame = ttk.Labelframe(
        parent,
        text="Population",
        style="Card.TLabelframe",
    )
    frame.pack(fill=tk.X, pady=(0, 10))
    label = ttk.Label(
        frame,
        text="Live: 0 | Δ: +0 | Peak: 0 | Density: 0.0%",
        wraplength=220,
        justify=tk.LEFT,
    )
    label.pack(anchor=tk.W)
    return label


def _add_custom_rules_section(
    parent: ttk.Frame,
    callbacks: Callbacks,
) -> tuple[ttk.Entry, ttk.Entry, ttk.Button]:
    """Add the custom rule inputs and return the relevant widgets."""

    frame = ttk.Labelframe(
        parent,
        text="Custom Rules",
        style="Card.TLabelframe",
    )
    frame.pack(fill=tk.X, pady=(0, 10))

    row = ttk.Frame(frame)
    row.pack(fill=tk.X)
    ttk.Label(row, text="B").pack(side=tk.LEFT)
    birth_entry = ttk.Entry(row, width=8)
    birth_entry.pack(side=tk.LEFT, padx=(4, 12))
    Tooltip(birth_entry, "Birth rule: digits for neighbor counts that create life")
    ttk.Label(row, text="S").pack(side=tk.LEFT)
    survival_entry = ttk.Entry(row, width=8)
    survival_entry.pack(side=tk.LEFT, padx=(4, 0))
    Tooltip(survival_entry, "Survival rule: digits for neighbor counts that sustain life")

    apply_button = ttk.Button(
        frame,
        text="Apply",
        command=callbacks.apply_custom_rules,
    )
    apply_button.pack(fill=tk.X, pady=(6, 0))
    Tooltip(apply_button, "Apply the custom birth/survival rules")

    return birth_entry, survival_entry, apply_button


def _add_grid_section(
    parent: ttk.Frame,
    variables: TkVars,
    callbacks: Callbacks,
) -> None:
    """Add grid configuration controls."""

    frame = ttk.Labelframe(
        parent,
        text="Grid",
        style="Card.TLabelframe",
    )
    frame.pack(fill=tk.X, pady=(0, 10))

    ttk.Label(frame, text="Preset").pack(anchor=tk.W)
    size_combo = ttk.Combobox(
        frame,
        textvariable=variables.grid_size,
        state="readonly",
        values=["50x50", "100x100", "150x150", "200x200", "Custom"],
    )
    size_combo.pack(fill=tk.X, pady=(2, 6))
    Tooltip(size_combo, "Choose a preset grid size")
    size_combo.bind("<<ComboboxSelected>>", callbacks.size_preset_changed)

    row = ttk.Frame(frame)
    row.pack(fill=tk.X)
    ttk.Label(row, text="W").pack(side=tk.LEFT)
    width_spin = tk.Spinbox(
        row,
        from_=10,
        to=500,
        textvariable=variables.custom_width,
        width=5,
    )
    width_spin.pack(side=tk.LEFT, padx=(4, 12))
    Tooltip(width_spin, "Custom grid width (10-500)")
    ttk.Label(row, text="H").pack(side=tk.LEFT)
    height_spin = tk.Spinbox(
        row,
        from_=10,
        to=500,
        textvariable=variables.custom_height,
        width=5,
    )
    height_spin.pack(side=tk.LEFT, padx=(4, 0))
    Tooltip(height_spin, "Custom grid height (10-500)")

    apply_size_button = ttk.Button(frame, text="Apply", command=callbacks.apply_custom_size)
    apply_size_button.pack(fill=tk.X, pady=(6, 0))
    Tooltip(apply_size_button, "Apply custom grid dimensions")


def _add_drawing_section(parent: ttk.Frame, variables: TkVars) -> None:
    """Add drawing tool radio buttons and symmetry selector."""

    frame = ttk.Labelframe(
        parent,
        text="Drawing",
        style="Card.TLabelframe",
    )
    frame.pack(fill=tk.X)

    ttk.Label(frame, text="Tool").pack(anchor=tk.W)
    row = ttk.Frame(frame)
    row.pack(anchor=tk.W, pady=(2, 6))
    toggle_radio = ttk.Radiobutton(
        row,
        text="Toggle",
        variable=variables.draw_mode,
        value="toggle",
    )
    toggle_radio.pack(side=tk.LEFT)
    Tooltip(toggle_radio, "Click to toggle cells on/off")
    pen_radio = ttk.Radiobutton(
        row,
        text="Pen",
        variable=variables.draw_mode,
        value="pen",
    )
    pen_radio.pack(side=tk.LEFT, padx=(8, 0))
    Tooltip(pen_radio, "Click and drag to draw live cells")
    eraser_radio = ttk.Radiobutton(
        row,
        text="Eraser",
        variable=variables.draw_mode,
        value="eraser",
    )
    eraser_radio.pack(side=tk.LEFT, padx=(8, 0))
    Tooltip(eraser_radio, "Click and drag to erase cells")

    ttk.Label(frame, text="Symmetry").pack(anchor=tk.W)
    symmetry_combo = ttk.Combobox(
        frame,
        textvariable=variables.symmetry,
        state="readonly",
        values=["None", "Horizontal", "Vertical", "Both", "Radial"],
    )
    symmetry_combo.pack(fill=tk.X, pady=(2, 0))
    Tooltip(symmetry_combo, "Mirror drawing actions across axes")


def _add_canvas_area(parent: ttk.Frame, callbacks: Callbacks) -> tk.Canvas:
    """Create the scrollable canvas area and return the canvas widget."""

    frame = ttk.Frame(parent)
    frame.grid(row=0, column=0, sticky="nsew")

    canvas = tk.Canvas(
        frame,
        bg="white",
        width=DEFAULT_CANVAS_WIDTH,
        height=DEFAULT_CANVAS_HEIGHT,
        highlightthickness=1,
        highlightbackground="#cccccc",
    )
    h_scroll = ttk.Scrollbar(frame, orient=tk.HORIZONTAL, command=canvas.xview)
    v_scroll = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=canvas.yview)
    canvas.configure(xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)
    canvas.grid(row=0, column=0, sticky=tk.NSEW)
    h_scroll.grid(row=1, column=0, sticky=tk.EW, pady=(4, 0))
    v_scroll.grid(row=0, column=1, sticky=tk.NS, padx=(4, 0))

    frame.rowconfigure(0, weight=1)
    frame.columnconfigure(0, weight=1)

    canvas.bind("<Button-1>", callbacks.on_canvas_click)
    canvas.bind("<B1-Motion>", callbacks.on_canvas_drag)

    return canvas
