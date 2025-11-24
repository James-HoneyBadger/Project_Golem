#!/usr/bin/env python3
"""Application entry point for the cellular automaton simulator."""

from __future__ import annotations

from gui.app import launch


def main() -> None:
    """Start the Tkinter event loop."""

    launch()


if __name__ == "__main__":
    main()
