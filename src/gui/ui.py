"""Widget construction and Tk variable helpers for the GUI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import tkinter as tk
from tkinter import ttk

from .config import DEFAULT_CANVAS_HEIGHT, DEFAULT_CANVAS_WIDTH, MODE_PATTERNS


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

    control_frame = tk.Frame(root, pady=10)
    control_frame.pack(side=tk.TOP, fill=tk.X, padx=10)

    # Row 0: Mode selection + primary controls
    tk.Label(control_frame, text="Mode:").grid(
        row=0,
        column=0,
        padx=5,
        sticky=tk.E,
    )
    mode_combo = ttk.Combobox(
        control_frame,
        textvariable=variables.mode,
        state="readonly",
        width=20,
        values=list(MODE_PATTERNS.keys()),
    )
    mode_combo.grid(row=0, column=1, padx=5)
    mode_combo.bind(
        "<<ComboboxSelected>>",
        lambda _event: callbacks.switch_mode(variables.mode.get()),
    )

    start_button = tk.Button(
        control_frame,
        text="Start",
        command=lambda: None,  # replaced by caller
        width=10,
        bg="#4caf50",
        fg="white",
    )
    start_button.grid(row=0, column=2, padx=5)

    tk.Button(
        control_frame,
        text="Step",
        command=callbacks.step_once,
        width=8,
    ).grid(row=0, column=3, padx=5)
    tk.Button(
        control_frame,
        text="Clear",
        command=callbacks.clear_grid,
        width=8,
        bg="#f44336",
        fg="white",
    ).grid(row=0, column=4, padx=5)
    tk.Button(
        control_frame,
        text="Reset",
        command=callbacks.reset_simulation,
        width=8,
    ).grid(row=0, column=5, padx=5)

    # Row 1: Pattern and IO controls
    tk.Label(control_frame, text="Pattern:").grid(
        row=1,
        column=0,
        padx=5,
        sticky=tk.E,
    )
    pattern_combo = ttk.Combobox(
        control_frame,
        textvariable=variables.pattern,
        state="readonly",
        width=20,
    )
    pattern_combo.grid(row=1, column=1, padx=5)
    pattern_combo.bind(
        "<<ComboboxSelected>>",
        lambda _event: callbacks.load_pattern(),
    )

    tk.Button(
        control_frame,
        text="Save",
        command=callbacks.save_pattern,
        width=8,
    ).grid(row=1, column=2, padx=5)
    tk.Button(
        control_frame,
        text="Load File",
        command=callbacks.load_saved_pattern,
        width=10,
    ).grid(row=1, column=3, padx=5)

    if show_export:
        tk.Button(
            control_frame,
            text="Export PNG",
            command=callbacks.export_png,
            width=12,
        ).grid(row=1, column=4, padx=5, columnspan=2)

    # Row 2: Custom rules (enabled for Custom mode)
    tk.Label(control_frame, text="Custom B/S:").grid(
        row=2, column=0, padx=5, sticky=tk.E
    )
    tk.Label(control_frame, text="B:").grid(row=2, column=1, sticky=tk.W)
    birth_entry = tk.Entry(control_frame, width=8)
    birth_entry.grid(row=2, column=1, padx=(20, 5), sticky=tk.W)
    tk.Label(control_frame, text="S:").grid(row=2, column=2, sticky=tk.W)
    survival_entry = tk.Entry(control_frame, width=8)
    survival_entry.grid(row=2, column=2, padx=(20, 5), sticky=tk.W)
    apply_rules_button = tk.Button(
        control_frame,
        text="Apply Rules",
        command=callbacks.apply_custom_rules,
        width=12,
    )
    apply_rules_button.grid(row=2, column=3, padx=5)

    # Row 3: Grid size controls
    tk.Label(control_frame, text="Grid Size:").grid(
        row=3, column=0, padx=5, sticky=tk.E
    )
    size_combo = ttk.Combobox(
        control_frame,
        textvariable=variables.grid_size,
        state="readonly",
        width=12,
        values=["50x50", "100x100", "150x150", "200x200", "Custom"],
    )
    size_combo.grid(row=3, column=1, padx=5, sticky=tk.W)
    size_combo.bind("<<ComboboxSelected>>", callbacks.size_preset_changed)

    tk.Label(control_frame, text="W:").grid(row=3, column=2, sticky=tk.W)
    tk.Spinbox(
        control_frame,
        from_=10,
        to=500,
        textvariable=variables.custom_width,
        width=6,
    ).grid(row=3, column=2, padx=(20, 2), sticky=tk.W)
    tk.Label(control_frame, text="H:").grid(row=3, column=3, sticky=tk.W)
    tk.Spinbox(
        control_frame,
        from_=10,
        to=500,
        textvariable=variables.custom_height,
        width=6,
    ).grid(row=3, column=3, padx=(20, 2), sticky=tk.W)
    tk.Button(
        control_frame,
        text="Apply",
        command=callbacks.apply_custom_size,
        width=8,
    ).grid(row=3, column=4, padx=5)

    # Row 4: Drawing tools
    tk.Label(control_frame, text="Draw:").grid(
        row=4,
        column=0,
        padx=5,
        sticky=tk.E,
    )
    tk.Radiobutton(
        control_frame,
        text="Toggle",
        variable=variables.draw_mode,
        value="toggle",
    ).grid(row=4, column=1, sticky=tk.W)
    tk.Radiobutton(
        control_frame,
        text="Pen",
        variable=variables.draw_mode,
        value="pen",
    ).grid(row=4, column=2, sticky=tk.W)
    tk.Radiobutton(
        control_frame,
        text="Eraser",
        variable=variables.draw_mode,
        value="eraser",
    ).grid(row=4, column=3, sticky=tk.W)

    tk.Label(control_frame, text="Symmetry:").grid(
        row=4,
        column=4,
        sticky=tk.E,
    )
    ttk.Combobox(
        control_frame,
        textvariable=variables.symmetry,
        state="readonly",
        width=12,
        values=["None", "Horizontal", "Vertical", "Both", "Radial"],
    ).grid(row=4, column=5, padx=5)

    # Row 5: Speed and generation
    tk.Label(control_frame, text="Speed:").grid(
        row=5,
        column=0,
        padx=5,
        sticky=tk.E,
    )
    tk.Scale(
        control_frame,
        from_=1,
        to=100,
        orient=tk.HORIZONTAL,
        variable=variables.speed,
        length=150,
    ).grid(row=5, column=1, columnspan=2, sticky=tk.W, padx=5)

    tk.Button(
        control_frame,
        text="Toggle Grid",
        command=callbacks.toggle_grid,
        width=12,
    ).grid(row=5, column=3, padx=5)

    gen_label = tk.Label(
        control_frame,
        text="Generation: 0",
        font=("Arial", 10, "bold"),
    )
    gen_label.grid(row=5, column=4, columnspan=2, padx=5)

    stats_frame = tk.Frame(control_frame)
    stats_frame.grid(row=6, column=0, columnspan=6, sticky=tk.W, pady=(8, 0))
    population_label = tk.Label(
        stats_frame,
        text="Live: 0 | Δ: +0 | Peak: 0 | Density: 0.0%",
    )
    population_label.pack(side=tk.LEFT)

    # Canvas + scrollbars
    canvas_frame = tk.Frame(root)
    canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
    canvas = tk.Canvas(
        canvas_frame,
        bg="white",
        width=DEFAULT_CANVAS_WIDTH,
        height=DEFAULT_CANVAS_HEIGHT,
    )
    h_scroll = tk.Scrollbar(
        canvas_frame,
        orient=tk.HORIZONTAL,
        command=canvas.xview,
    )
    v_scroll = tk.Scrollbar(
        canvas_frame,
        orient=tk.VERTICAL,
        command=canvas.yview,
    )
    canvas.configure(xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)
    canvas.grid(row=0, column=0, sticky=tk.NSEW)
    h_scroll.grid(row=1, column=0, sticky=tk.EW)
    v_scroll.grid(row=0, column=1, sticky=tk.NS)
    canvas_frame.rowconfigure(0, weight=1)
    canvas_frame.columnconfigure(0, weight=1)

    canvas.bind("<Button-1>", callbacks.on_canvas_click)
    canvas.bind("<B1-Motion>", callbacks.on_canvas_drag)

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
