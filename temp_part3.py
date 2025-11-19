
    def switch_mode(self, mode):
        """Switch between different cellular automaton modes"""
        was_running = self.running
        if self.running:
            self.toggle_simulation()

        if mode == "Conway's Game of Life":
            self.current_automaton = ConwayGameOfLife(
                self.grid_width,
                self.grid_height,
            )
            # Enable pattern selector
            self.pattern_combo.config(state="readonly")
            self.load_pattern_button.config(state="normal")
            self._set_rules_controls_enabled(False)
            self._update_pattern_list(
                [
                    "Empty",
                    "Classic Mix",
                    "Glider Gun",
                    "Spaceships",
                    "Oscillators",
                    "Puffers",
                    "R-Pentomino",
                    "Acorn",
                    "Random Soup",
                ]
            )
        elif mode == "High Life":
            self.current_automaton = HighLife(
                self.grid_width,
                self.grid_height,
            )
            # Enable pattern selector
            self.pattern_combo.config(state="readonly")
            self.load_pattern_button.config(state="normal")
            self._set_rules_controls_enabled(False)
            self._update_pattern_list(["Empty", "Replicator", "Random Soup"])
        elif mode == "Immigration Game":
            self.current_automaton = ImmigrationGame(
                self.grid_width,
                self.grid_height,
            )
            # Enable pattern selector
            self.pattern_combo.config(state="readonly")
            self.load_pattern_button.config(state="normal")
            self._set_rules_controls_enabled(False)
            self._update_pattern_list(["Empty", "Color Mix", "Random Soup"])
        elif mode == "Rainbow Game":
            self.current_automaton = RainbowGame(
                self.grid_width,
                self.grid_height,
            )
            # Enable pattern selector
            self.pattern_combo.config(state="readonly")
            self.load_pattern_button.config(state="normal")
            self._set_rules_controls_enabled(False)
            self._update_pattern_list(["Empty", "Rainbow Mix", "Random Soup"])
        elif mode == "Langton's Ant":
            self.current_automaton = LangtonsAnt(
                self.grid_width,
                self.grid_height,
            )
            # Disable pattern selector for Langton's Ant
            self.pattern_combo.config(state="disabled")
            self.load_pattern_button.config(state="disabled")
            self._set_rules_controls_enabled(False)
        elif mode == "Custom Rules":
            # Build LifeLike with current rules
            b, s = parse_bs(self.rules_var.get())
            self.current_automaton = LifeLikeAutomaton(
                self.grid_width,
                self.grid_height,
                birth=b,
                survival=s,
            )
            # Enable patterns limited for life-like
            self.pattern_combo.config(state="readonly")
            self.load_pattern_button.config(state="normal")
            self._set_rules_controls_enabled(True)
            self._update_pattern_list(["Empty", "Random Soup"])

        self.generation = 0
        self.update_display()
        self.update_status()

        if was_running:
            self.toggle_simulation()

    def _set_rules_controls_enabled(self, enabled):
        state = "normal" if enabled else "disabled"
        try:
            self.rules_entry.config(state=state)
        except Exception:
            pass

    def apply_custom_rules(self):
        """Apply rules from entry to current LifeLike automaton."""
        if isinstance(self.current_automaton, LifeLikeAutomaton):
            b, s = parse_bs(self.rules_var.get())
            self.current_automaton.set_rules(b, s)
            # Do not reset grid; just apply new rules going forward

    def _update_pattern_list(self, patterns):
        """Update the pattern dropdown with new values"""
        self.pattern_combo["values"] = patterns
        self.pattern_var.set("Empty")

    def load_pattern_handler(self):
        """Handle pattern selection and loading"""
        pattern_name = self.pattern_var.get()

        if pattern_name == "Empty":
            self.clear_grid()
            return

        # Load patterns for automata that support them
        if hasattr(self.current_automaton, "load_pattern"):
            self.current_automaton.load_pattern(pattern_name)
            self.generation = 0
            self.update_display()
            self.update_status()

    def toggle_simulation(self):
        """Start or stop the simulation"""
        self.running = not self.running
        if self.running:
            self.start_button.config(text="Stop")
            self.run_simulation()
        else:
            self.start_button.config(text="Start")

    def run_simulation(self):
        """Main simulation loop"""
        if self.running:
            self.step_once()
            self.update_delay = 510 - self.speed_var.get()
            self.root.after(self.update_delay, self.run_simulation)

    def step_once(self):
        """Perform one step of the simulation"""
        if self.current_automaton:
            # Record history if recording and not in replay mode
            if self.recording_history and self.history_position < 0:
                self.history.append(np.copy(self.current_automaton.get_grid()))

            self.current_automaton.step()
            self.generation += 1
            self.update_display()
            self.update_status()

    def reset_simulation(self):
        """Reset to initial state"""
        if self.current_automaton:
            self.current_automaton.reset()
            self.generation = 0
            self.update_display()
            self.update_status()

    def clear_grid(self):
        """Clear the entire grid"""
        self.reset_simulation()

    def save_pattern(self):
        """Save current grid state to file"""
        if not self.current_automaton:
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Save Pattern",
        )

        if filename:
            try:
                data = {
                    "mode": self.mode_var.get(),
                    "grid": self.current_automaton.get_grid().tolist(),
                    "width": self.grid_width,
                    "height": self.grid_height,
                    "generation": self.generation,
                }
                # Include rules when in Custom Rules mode
                if self.mode_var.get() == "Custom Rules":
                    data["rules"] = self.rules_var.get()

                with open(filename, "w") as f:
                    json.dump(data, f, indent=2)

                messagebox.showinfo(
                    "Success", f"Pattern saved to {os.path.basename(filename)}"
                )
            except Exception as e:
                messagebox.showerror(
                    "Error",
                    f"Failed to save pattern: {str(e)}",
                )

    def load_saved_pattern(self):
        """Load grid state from file"""
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Load Pattern",
        )

        if filename:
            try:
                with open(filename, "r") as f:
                    data = json.load(f)

                # Switch to the correct mode if needed
                if data["mode"] != self.mode_var.get():
                    self.mode_var.set(data["mode"])
                    self.switch_mode(data["mode"])
                # If custom rules, apply rules string first
                if data.get("mode") == "Custom Rules":
                    rules_str = data.get("rules", "B3/S23")
                    self.rules_var.set(rules_str)
                    if isinstance(self.current_automaton, LifeLikeAutomaton):
                        b, s = parse_bs(rules_str)
                        self.current_automaton.set_rules(b, s)

                # Load the grid
                loaded_grid = np.array(data["grid"])

                # Check dimensions match
                if loaded_grid.shape == (self.grid_height, self.grid_width):
                    self.current_automaton.grid = loaded_grid
                    self.generation = data.get("generation", 0)
                    self.prev_grid = None  # Force full redraw
                    self.update_display()
                    self.update_status()
                    messagebox.showinfo(
                        "Success",
                        f"Pattern loaded from {os.path.basename(filename)}",
                    )
                else:
                    messagebox.showerror(
                        "Error",
                        (
                            "Grid size mismatch. File:"
                            f" {loaded_grid.shape}, Current:"
                            f" {(self.grid_height, self.grid_width)}"
                        ),
                    )
            except Exception as e:
                messagebox.showerror(
                    "Error",
                    f"Failed to load pattern: {str(e)}",
                )

    def update_display(self):
        """Optimized redraw - only update changed cells"""
        if not self.current_automaton:
            return

        grid = self.current_automaton.get_grid()

        # Define color palettes for different modes
        immigration_colors = {
            0: "white",
            1: "#FF6B6B",  # Red
            2: "#4ECDC4",  # Cyan
            3: "#FFE66D",  # Yellow
        }

        rainbow_colors = {
            0: "white",
            1: "#FF0000",  # Red
            2: "#FF7F00",  # Orange
            3: "#FFFF00",  # Yellow
            4: "#00FF00",  # Green
            5: "#0000FF",  # Blue
            6: "#8B00FF",  # Purple
        }

        # Export helpers reuse these color maps
        self._immigration_colors = immigration_colors
        self._rainbow_colors = rainbow_colors

        # First time or after reset - create all cells
        if self.prev_grid is None or self.prev_grid.shape != grid.shape:
            self.canvas.delete("all")
            self.cell_rects = {}

            for i in range(self.grid_height):
                for j in range(self.grid_width):
                    x1 = j * self.cell_size
                    y1 = i * self.cell_size
                    x2 = x1 + self.cell_size
                    y2 = y1 + self.cell_size

                    outline_color = "gray" if self.show_grid_lines else ""
                    rect_id = self.canvas.create_rectangle(
                        x1, y1, x2, y2, fill="white", outline=outline_color
                    )
                    self.cell_rects[(i, j)] = rect_id

            self.prev_grid = np.copy(grid)

        # Update only changed cells
        changed = np.where(self.prev_grid != grid)

        for i, j in zip(changed[0], changed[1]):
            cell_value = grid[i, j]

            # Determine color based on mode and cell value
            if isinstance(self.current_automaton, ImmigrationGame):
                color = immigration_colors.get(cell_value, "white")
            elif isinstance(self.current_automaton, RainbowGame):
                color = rainbow_colors.get(cell_value, "white")
            elif isinstance(self.current_automaton, LangtonsAnt):
                if cell_value == 0:
                    color = "white"
                elif cell_value == 1:
                    color = "black"
                else:  # Ant position
                    color = "red"
            else:
                # Conway's Game of Life and High Life
                color = "white" if cell_value == 0 else "black"

            # Update the cell color
            if (i, j) in self.cell_rects:
                self.canvas.itemconfig(self.cell_rects[(i, j)], fill=color)

        # Update previous grid
        self.prev_grid = np.copy(grid)

    def update_status(self):
        """Update status bar with population count"""
        mode_name = self.mode_var.get()
        population = np.count_nonzero(self.current_automaton.get_grid())
        # Record population history for stats panel
        self.population_history.append(population)

        # Calculate FPS
        current_time = time.time()
        delta = current_time - self.last_update_time
        if delta > 0:
            self.frame_times.append(delta)
            if len(self.frame_times) > 0:
                avg_delta = sum(self.frame_times) / len(self.frame_times)
                self.fps = 1.0 / avg_delta if avg_delta > 0 else 0
        self.last_update_time = current_time

        status_text = f"{mode_name} | Gen: {self.generation} | Pop: {population}"
        if self.running:
            status_text += f" | FPS: {self.fps:.1f}"
        if self.history_position >= 0:
            status_text += f" | Replay: {self.history_position}/{len(self.history)}"

        self.status_var.set(status_text)
        # Update stats panel if open
        if self.stats_window is not None:
            self.draw_stats()

    def open_stats_panel(self):
        """Open or focus the statistics panel"""
        if self.stats_window is not None:
            try:
                self.stats_window.lift()
                return
            except Exception:
                self.stats_window = None

        win = tk.Toplevel(self.root)
        win.title("Population Stats")
        win.protocol("WM_DELETE_WINDOW", self.close_stats_panel)
        self.stats_window = win

        self.stats_canvas = tk.Canvas(
            win,
            width=self.stats_canvas_w,
            height=self.stats_canvas_h,
            bg="white",
        )
        self.stats_canvas.pack(fill=tk.BOTH, expand=True)
        self.draw_stats()

    def close_stats_panel(self):
        if self.stats_window is not None:
            try:
                self.stats_window.destroy()
            except Exception:
                pass
        self.stats_window = None
        self.stats_canvas = None

    def draw_stats(self):
        """Draw population over time on stats canvas"""
        if self.stats_canvas is None:
            return

        c = self.stats_canvas
        w = int(c.winfo_width() or self.stats_canvas_w)
        h = int(c.winfo_height() or self.stats_canvas_h)
        c.delete("all")

        # Axes and labels
        padding = 10
        inner_w = max(1, w - 2 * padding)
        inner_h = max(1, h - 2 * padding)
        c.create_rectangle(
            padding,
            padding,
            padding + inner_w,
            padding + inner_h,
            outline="#ddd",
        )

        # Plot population as percentage of cells
        total_cells = max(1, self.grid_width * self.grid_height)
        values = list(self.population_history)
        if len(values) < 2:
            return
        # Normalize to [0,1]
        ys = [v / total_cells for v in values]
        # Create points across inner_w
        step_x = inner_w / (len(ys) - 1)
        points = []
        for idx, yv in enumerate(ys):
            x = padding + idx * step_x
            y = padding + inner_h * (1.0 - yv)
            points.append((x, y))
        # Draw polyline
        for i in range(1, len(points)):
            x1, y1 = points[i - 1]
            x2, y2 = points[i]
            c.create_line(x1, y1, x2, y2, fill="#1f77b4", width=2)
        # Latest value text
        pct = ys[-1] * 100.0
        c.create_text(
            padding + 5,
            padding + 10,
            anchor=tk.W,
            text=f"Pop: {values[-1]} ({pct:.1f}%)",
            fill="#333",
        )

    # ---------- Export helpers ----------
    def _render_grid_to_image(self, scale=None):
        """Render current grid to a PIL Image. Scale defaults to cell_size."""
        if not PIL_AVAILABLE:
            raise RuntimeError("Pillow not available")
        if not self.current_automaton:
            raise RuntimeError("No automaton to render")

        scale = int(scale or self.cell_size)
        scale = max(1, min(40, scale))
        grid = self.current_automaton.get_grid()
        h, w = grid.shape
        img_w = w * scale
        img_h = h * scale

        # Background
        img = Image.new("RGB", (img_w, img_h), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        mode = self.mode_var.get()
        for y in range(h):
            for x in range(w):
                val = int(grid[y, x])
                if mode in ("Immigration Game",):
                    color = self._immigration_colors.get(val, "white")
                elif mode in ("Rainbow Game",):
                    color = self._rainbow_colors.get(val, "white")
                else:
                    # Binary or ant/custom
                    if val == 0:
                        continue
                    color = "black"

                if color == "white":
                    continue
                x0 = x * scale
                y0 = y * scale
                x1 = x0 + scale
                y1 = y0 + scale
                draw.rectangle([x0, y0, x1, y1], fill=color)

        # Langton's Ant overlay as red square
        if mode == "Langton's Ant" and hasattr(self.current_automaton, "ant_x"):
            ax = self.current_automaton.ant_x * scale
            ay = self.current_automaton.ant_y * scale
            draw.rectangle([ax, ay, ax + scale, ay + scale], fill="#FF0000")

        return img

    def export_png(self):
        """Export current view to a PNG file."""
        if not PIL_AVAILABLE:
            messagebox.showerror(
                "Export Error",
                "Pillow is required. Please install 'Pillow' to export.",
            )
            return
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("All files", "*.*")],
            title="Export PNG",
        )
        if not filename:
            return
        try:
            img = self._render_grid_to_image()
            img.save(filename, format="PNG")
            messagebox.showinfo("Export", "PNG exported successfully")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))

    def export_gif(self):
        """Export an animated GIF by stepping the automaton."""
        if not PIL_AVAILABLE:
            messagebox.showerror(
                "Export Error",
                "Pillow is required. Please install 'Pillow' to export.",
            )
            return
        filename = filedialog.asksaveasfilename(
            defaultextension=".gif",
            filetypes=[("GIF", "*.gif"), ("All files", "*.*")],
            title="Export GIF",
        )
        if not filename:
            return
        steps = simpledialog.askinteger(
            "GIF Frames",
            "How many steps to export?",
            initialvalue=50,
            minvalue=1,
            maxvalue=1000,
        )
        if not steps:
            return
        duration = simpledialog.askinteger(
            "Frame Duration (ms)",
            "Milliseconds per frame:",
            initialvalue=100,
            minvalue=10,
            maxvalue=2000,
        )
        if not duration:
            return
        try:
            # Backup state
            backup = None
            mode = self.mode_var.get()
            if mode == "Langton's Ant":
                a = self.current_automaton
                backup = (
                    np.copy(a.grid),
                    a.ant_x,
                    a.ant_y,
                    a.ant_dir,
                )
            else:
                backup = np.copy(self.current_automaton.get_grid())

            images = []
            for _ in range(steps):
                images.append(self._render_grid_to_image())
                self.current_automaton.step()

            # Restore
            if mode == "Langton's Ant":
                a = self.current_automaton
                a.grid = backup[0]
                a.ant_x, a.ant_y, a.ant_dir = backup[1], backup[2], backup[3]
            else:
                self.current_automaton.grid = backup
            self.update_display()
            self.update_status()

            if images:
                images[0].save(
                    filename,
                    save_all=True,
                    append_images=images[1:],
                    duration=duration,
                    loop=0,
                    optimize=False,
                )
                messagebox.showinfo("Export", "GIF exported successfully")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))

    def on_canvas_click(self, event):
        """Handle mouse click on canvas with drawing modes and symmetry"""
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        x = int(canvas_x // self.cell_size)
        y = int(canvas_y // self.cell_size)

        if not (0 <= x < self.grid_width and 0 <= y < self.grid_height):
            return

        mode = self.draw_mode

        if mode in ("line", "rect"):
            # Start or finish shape
            if self.draw_start is None:
                self.draw_start = (x, y)
            else:
                x0, y0 = self.draw_start
                action = self._current_action()
                if mode == "line":
                    self._draw_line(x0, y0, x, y, action=action)
                else:
                    self._draw_rect(x0, y0, x, y, action=action)
                self.draw_start = None
                self.update_display()
        else:
            # Single cell or continuous draw
            action = self._current_action()
            for px, py in self.apply_symmetry(x, y):
                self._apply_action_to_cell(px, py, action)
            self.update_display()
            self.last_draw_pos = (x, y)

    def on_canvas_drag(self, event):
        """Handle mouse drag on canvas for continuous drawing"""
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        x = int(canvas_x // self.cell_size)
        y = int(canvas_y // self.cell_size)

        if not (0 <= x < self.grid_width and 0 <= y < self.grid_height):
            return

        if self.draw_mode in ("line", "rect"):
            # Wait for release to finalize shape
            return

        action = self._current_action()
        if self.last_draw_pos is not None:
            lx, ly = self.last_draw_pos
            for px, py in self._line_points(lx, ly, x, y):
                for sx, sy in self.apply_symmetry(px, py):
                    self._apply_action_to_cell(sx, sy, action)
        else:
            for px, py in self.apply_symmetry(x, y):
                self._apply_action_to_cell(px, py, action)
        self.last_draw_pos = (x, y)
        self.update_display()

    def on_canvas_release(self, event):
        """Reset drag state after mouse release"""
        self.last_draw_pos = None

    # --- Drawing helpers ---
    def _current_action(self):
        """Map draw_mode to an action string"""
        if self.draw_mode == "pen":
            return "set"
        if self.draw_mode == "eraser":
            return "clear"
        return "toggle"

    def _apply_action_to_cell(self, x, y, action):
        """Apply a drawing action to a cell for the current automaton"""
        automaton = self.current_automaton
        if action == "toggle":
            automaton.handle_click(x, y)
            return

        # 'set' or 'clear'
        val = 1 if action == "set" else 0

        if isinstance(automaton, (ConwayGameOfLife, HighLife)):
            automaton.grid[y, x] = val
        elif isinstance(automaton, ImmigrationGame):
            automaton.grid[y, x] = 1 if action == "set" else 0
        elif isinstance(automaton, RainbowGame):
            automaton.grid[y, x] = 1 if action == "set" else 0
        elif isinstance(automaton, LangtonsAnt):
            # Do not move ant on draw; only set the square color
            if action == "set":
                automaton.grid[y, x] = 1
            else:
                automaton.grid[y, x] = 0

    def _line_points(self, x0, y0, x1, y1):
        """Bresenham's line algorithm yielding integer grid points"""
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        x, y = x0, y0
        points = []
        while True:
            if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
                points.append((x, y))
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x += sx
            if e2 <= dx:
                err += dx
                y += sy
        return points

    def _draw_line(self, x0, y0, x1, y1, action):
        for px, py in self._line_points(x0, y0, x1, y1):
            for sx, sy in self.apply_symmetry(px, py):
                self._apply_action_to_cell(sx, sy, action)

    def _draw_rect(self, x0, y0, x1, y1, action):
        x_min, x_max = sorted((x0, x1))
        y_min, y_max = sorted((y0, y1))
        for yy in range(y_min, y_max + 1):
            for xx in range(x_min, x_max + 1):
                for sx, sy in self.apply_symmetry(xx, yy):
                    self._apply_action_to_cell(sx, sy, action)

    def on_mousewheel(self, event):
        """Zoom in/out around mouse position"""
        # Determine direction
        zoom_in = False
        if hasattr(event, "delta") and event.delta != 0:
            zoom_in = event.delta > 0
        elif hasattr(event, "num"):
            # Linux: Button-4 (up), Button-5 (down)
            zoom_in = event.num == 4

        factor = 1.1 if zoom_in else 0.9
        old_size = self.cell_size
        new_size = max(2, min(64, int(round(old_size * factor))))
        if new_size == old_size:
            return

        # World coords under cursor before zoom
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)

        # Update size and scrollregion
        self.cell_size = new_size
        self.prev_grid = None  # Force full redraw to rebuild rectangles
        self.canvas.config(
            scrollregion=(
                0,
                0,
                self.grid_width * self.cell_size,
                self.grid_height * self.cell_size,
            )
        )
        self.update_display()

        # Keep mouse position anchored
        srw = self.grid_width * self.cell_size
        srh = self.grid_height * self.cell_size
        new_cx = cx * (new_size / old_size)
        new_cy = cy * (new_size / old_size)
        view_w = max(1, self.canvas.winfo_width())
        view_h = max(1, self.canvas.winfo_height())
        left_frac = (new_cx - event.x) / srw if srw > 0 else 0
        top_frac = (new_cy - event.y) / srh if srh > 0 else 0
        max_left = max(0.0, 1.0 - view_w / srw) if srw > 0 else 0.0
        max_top = max(0.0, 1.0 - view_h / srh) if srh > 0 else 0.0
        left_frac = min(max(left_frac, 0.0), max_left)
        top_frac = min(max(top_frac, 0.0), max_top)
        self.canvas.xview_moveto(left_frac)
        self.canvas.yview_moveto(top_frac)

    def on_shift_mousewheel(self, event):
        """Horizontal scroll with Shift + wheel"""
        delta = event.delta if hasattr(event, "delta") else 0
        step = -1 if delta > 0 else 1
        self.canvas.xview_scroll(step, "units")

    def on_pan_start(self, event):
        """Begin panning with middle mouse button"""
        self.canvas.scan_mark(event.x, event.y)

    def on_pan_move(self, event):
        """Pan as the middle mouse is dragged"""
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def rewind_history(self):
        """Go back one step in history"""
        if len(self.history) > 0:
            if self.history_position < 0:
                self.history_position = len(self.history) - 1
            elif self.history_position > 0:
                self.history_position -= 1

            if 0 <= self.history_position < len(self.history):
                self.current_automaton.grid = np.copy(
                    self.history[self.history_position]
                )
                self.prev_grid = None
                self.update_display()
                self.update_status()

    def forward_history(self):
        """Go forward one step in history"""
        if self.history_position >= 0:
            if self.history_position < len(self.history) - 1:
                self.history_position += 1
                self.current_automaton.grid = np.copy(
                    self.history[self.history_position]
                )
            else:
                self.history_position = -1  # Return to live mode

            self.prev_grid = None
            self.update_display()
            self.update_status()

    def clear_history(self):
        """Clear the history buffer"""
        self.history.clear()
        self.history_position = -1
        self.update_status()

    def toggle_grid_lines(self):
        """Toggle visibility of grid lines"""
        self.show_grid_lines = not self.show_grid_lines
        self.prev_grid = None  # Force full redraw
        self.update_display()

    def set_draw_mode(self, mode):
        """Set the drawing mode (toggle, pen, eraser, etc.)"""
        self.draw_mode = mode

    def set_symmetry_mode(self, mode):
        """Set symmetry mode for drawing"""
        self.symmetry_mode = mode

    def update_draw_mode(self):
        """Update draw mode from UI control"""
        self.set_draw_mode(self.draw_mode_var.get())

    def update_symmetry_mode(self, event=None):
        """Update symmetry mode from UI control"""
        mode = self.symmetry_var.get().lower()
        self.set_symmetry_mode(mode)

    def apply_symmetry(self, x, y):
        """Apply symmetry transformations to a cell position"""
        positions = [(x, y)]

        if self.symmetry_mode in ["horizontal", "both"]:
            positions.append((self.grid_width - 1 - x, y))

        if self.symmetry_mode in ["vertical", "both"]:
            positions.append((x, self.grid_height - 1 - y))

        if self.symmetry_mode == "both":
            positions.append((self.grid_width - 1 - x, self.grid_height - 1 - y))

        if self.symmetry_mode == "rotational":
            center_x = self.grid_width // 2
            center_y = self.grid_height // 2
            for angle in [90, 180, 270]:
                # Rotate around center
                dx = x - center_x
                dy = y - center_y
                if angle == 90:
                    positions.append((center_x - dy, center_y + dx))
                elif angle == 180:
                    positions.append((center_x - dx, center_y - dy))
                elif angle == 270:
                    positions.append((center_x + dy, center_y - dx))

        return [
            (px, py)
            for px, py in positions
            if 0 <= px < self.grid_width and 0 <= py < self.grid_height
        ]

    def resize_grid(self, width, height):
        """Resize the grid to new dimensions"""
        old_grid = self.current_automaton.get_grid().copy()

        self.grid_width = width
        self.grid_height = height

        # Reinitialize the automaton
        current_mode = self.mode_var.get()
        self.switch_mode(current_mode)

        # Try to preserve content (copy what fits)
        min_h = min(old_grid.shape[0], height)
        min_w = min(old_grid.shape[1], width)
        self.current_automaton.grid[:min_h, :min_w] = old_grid[:min_h, :min_w]

        # Clear caches
        self.cell_rects = {}
        self.prev_grid = None
        self.history.clear()
        self.history_position = -1

        # Recreate canvas
        self.canvas.config(
            width=self.grid_width * self.cell_size,
            height=self.grid_height * self.cell_size,
            scrollregion=(
                0,
                0,
                self.grid_width * self.cell_size,
                self.grid_height * self.cell_size,
            ),
        )

        self.update_display()
        self.update_status()

    def apply_grid_size_preset(self, event=None):
        """Apply a preset grid size"""
        preset = self.size_preset_var.get()

        if preset.startswith("Small"):
            self.resize_grid(50, 40)
            self.custom_width_var.set(50)
            self.custom_height_var.set(40)
        elif preset.startswith("Medium"):
            self.resize_grid(100, 80)
            self.custom_width_var.set(100)
            self.custom_height_var.set(80)
        elif preset.startswith("Large"):
            self.resize_grid(200, 160)
            self.custom_width_var.set(200)
            self.custom_height_var.set(160)
        # "Custom" option doesn't change anything - user uses spinboxes

    def apply_custom_grid_size(self):
        """Apply custom grid size from spinboxes"""
        new_width = self.custom_width_var.get()
        new_height = self.custom_height_var.get()

        if 10 <= new_width <= 500 and 10 <= new_height <= 500:
            self.resize_grid(new_width, new_height)
            self.size_preset_var.set("Custom")
        else:
            messagebox.showwarning(
                "Invalid Size", "Grid dimensions must be between 10 and 500"
            )


def main():
    root = tk.Tk()
    # Keep reference on root to avoid unused-variable lint and GC
    root.app = CellularAutomatonGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
