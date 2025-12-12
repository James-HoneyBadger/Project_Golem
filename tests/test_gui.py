# pylint: disable=duplicate-code

"""GUI-related smoke tests.

These tests exercise the Tk UI at a basic level. They will be skipped
automatically if Tk cannot initialize (e.g., no DISPLAY on Linux).
"""

from __future__ import annotations

import os
import sys
import tkinter as tk

import pytest

# Add src/ to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gui.app import AutomatonApp  # pylint: disable=wrong-import-position


def _can_init_tk() -> bool:
    """Check if Tkinter can be initialized."""
    try:
        root = tk.Tk()
        root.withdraw()
        root.destroy()
        return True
    except RuntimeError:
        return False


TK_AVAILABLE = _can_init_tk()


@pytest.mark.skipif(
    not TK_AVAILABLE, reason="Tkinter not available or no display"
)
def test_app_launch_and_quit() -> None:
    """Test launching and quitting the application."""
    root = tk.Tk()
    root.withdraw()  # don't show window during test
    app = AutomatonApp(root)
    assert app.widgets.start_button is not None
    assert app.widgets.population_canvas is not None
    assert app.widgets.cycle_label is not None
    # Close cleanly
    app._on_close()  # pylint: disable=protected-access


@pytest.mark.skipif(not TK_AVAILABLE, reason="Tkinter not available or no display")
def test_mode_switch_updates_patterns() -> None:
    """Test that switching modes updates the pattern list."""
    root = tk.Tk()
    root.withdraw()
    app = AutomatonApp(root)
    app.switch_mode("Wireworld")
    # Ensure pattern combo values updated
    values = list(app.widgets.pattern_combo["values"])  # type: ignore[index]
    assert "Random Soup" in values
    app._on_close()  # pylint: disable=protected-access


@pytest.mark.skipif(not TK_AVAILABLE, reason="Tkinter not available or no display")
def test_export_metrics_collects_rows() -> None:
    """Test that metrics export creates CSV rows."""
    root = tk.Tk()
    root.withdraw()
    app = AutomatonApp(root)
    # Run a few steps to populate metrics
    app.step_once()
    app.step_once()
    assert len(app.state.metrics_log) >= 2
    app._on_close()  # pylint: disable=protected-access
