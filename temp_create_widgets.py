    def create_widgets(self):
        # Control panel
        control_frame = tk.Frame(self.root, bg='white')
        control_frame.grid(row=0, column=0, padx=10, pady=10, sticky=tk.W+tk.E)
        
        # Row 0: Mode and basic controls
        tk.Label(control_frame, text="Mode:", bg='white').grid(row=0, column=0, padx=5, sticky=tk.W)
        self.mode_combobox = ttk.Combobox(control_frame, textvariable=self.mode_var, state='readonly', width=20)
        self.mode_combobox['values'] = [mode.replace('_', ' ').title() for mode in self.modes]
        self.mode_combobox.grid(row=0, column=1, padx=5)
        
        tk.Button(control_frame, text="Start", command=self.start_simulation, bg='white').grid(row=0, column=2, padx=5)
        tk.Button(control_frame, text="Step", command=self.step_simulation, bg='white').grid(row=0, column=3, padx=5)
        tk.Button(control_frame, text="Clear", command=self.clear_grid, bg='white').grid(row=0, column=4, padx=5)
        tk.Button(control_frame, text="Reset", command=self.reset_simulation, bg='white').grid(row=0, column=5, padx=5)
        
        # Row 1: Pattern and file operations
        tk.Label(control_frame, text="Pattern:", bg='white').grid(row=1, column=0, padx=5, sticky=tk.W)
        self.pattern_combobox = ttk.Combobox(control_frame, textvariable=self.pattern_var, state='readonly', width=20)
        self.pattern_combobox['values'] = list(self.patterns.keys())
        self.pattern_combobox.grid(row=1, column=1, padx=5)
        
        tk.Button(control_frame, text="Load Pattern", command=self.load_pattern, bg='white').grid(row=1, column=2, padx=5)
        tk.Button(control_frame, text="Save", command=self.save_state, bg='white').grid(row=1, column=3, padx=5)
        tk.Button(control_frame, text="Load File", command=self.load_state, bg='white').grid(row=1, column=4, padx=5)
        tk.Button(control_frame, text="Export PNG", command=self.export_as_png, bg='white').grid(row=1, column=5, padx=5)
        tk.Button(control_frame, text="Export GIF", command=self.export_as_gif, bg='white').grid(row=1, column=6, padx=5)
        
        # Row 2: Custom rules
        tk.Label(control_frame, text="Custom Rules:", bg='white').grid(row=2, column=0, padx=5, sticky=tk.W)
        tk.Label(control_frame, text="B:", bg='white').grid(row=2, column=1, sticky=tk.E)
        self.birth_entry = tk.Entry(control_frame, width=10)
        self.birth_entry.grid(row=2, column=2, padx=5)
        tk.Label(control_frame, text="S:", bg='white').grid(row=2, column=3, sticky=tk.E)
        self.survival_entry = tk.Entry(control_frame, width=10)
        self.survival_entry.grid(row=2, column=4, padx=5)
        tk.Button(control_frame, text="Apply Rules", command=self.apply_custom_rules, bg='white').grid(row=2, column=5, padx=5)
        
        # Row 3: Grid size
        tk.Label(control_frame, text="Grid Size:", bg='white').grid(row=3, column=0, padx=5, sticky=tk.W)
        self.size_combobox = ttk.Combobox(control_frame, textvariable=self.grid_size_var, state='readonly', width=15)
        self.size_combobox['values'] = ['50x50', '100x100', '150x150', '200x200', 'Custom']
        self.size_combobox.grid(row=3, column=1, padx=5)
        
        tk.Label(control_frame, text="W:", bg='white').grid(row=3, column=2, sticky=tk.E)
        self.width_spinbox = tk.Spinbox(control_frame, from_=10, to=500, textvariable=self.custom_width, width=8)
        self.width_spinbox.grid(row=3, column=3, padx=5)
        tk.Label(control_frame, text="H:", bg='white').grid(row=3, column=4, sticky=tk.E)
        self.height_spinbox = tk.Spinbox(control_frame, from_=10, to=500, textvariable=self.custom_height, width=8)
        self.height_spinbox.grid(row=3, column=5, padx=5)
        tk.Button(control_frame, text="Apply Size", command=self.apply_grid_size, bg='white').grid(row=3, column=6, padx=5)
        
        # Row 4: Draw mode and symmetry
        tk.Label(control_frame, text="Draw Mode:", bg='white').grid(row=4, column=0, padx=5, sticky=tk.W)
        tk.Radiobutton(control_frame, text="Toggle", variable=self.draw_mode_var, value='toggle', bg='white').grid(row=4, column=1, sticky=tk.W)
        tk.Radiobutton(control_frame, text="Pen", variable=self.draw_mode_var, value='pen', bg='white').grid(row=4, column=2, sticky=tk.W)
        tk.Radiobutton(control_frame, text="Eraser", variable=self.draw_mode_var, value='eraser', bg='white').grid(row=4, column=3, sticky=tk.W)
        
        tk.Label(control_frame, text="Symmetry:", bg='white').grid(row=4, column=4, padx=5, sticky=tk.E)
        self.symmetry_combobox = ttk.Combobox(control_frame, textvariable=self.symmetry_var, state='readonly', width=15)
        self.symmetry_combobox['values'] = ['None', 'Horizontal', 'Vertical', 'Both', 'Radial']
        self.symmetry_combobox.grid(row=4, column=5, padx=5)
        
        # Row 5: Speed and history
        tk.Label(control_frame, text="Speed:", bg='white').grid(row=5, column=0, padx=5, sticky=tk.W)
        self.speed_slider = tk.Scale(control_frame, from_=1, to=100, orient=tk.HORIZONTAL, variable=self.speed_var, bg='white')
        self.speed_slider.grid(row=5, column=1, columnspan=2, padx=5, sticky=tk.W+tk.E)
        
        tk.Button(control_frame, text="<<", command=self.history_backward, bg='white', width=3).grid(row=5, column=3, padx=2)
        tk.Button(control_frame, text=">>", command=self.history_forward, bg='white', width=3).grid(row=5, column=4, padx=2)
        tk.Button(control_frame, text="Toggle Grid", command=self.toggle_grid, bg='white').grid(row=5, column=5, padx=5)
        tk.Button(control_frame, text="Stats", command=self.show_statistics, bg='white').grid(row=5, column=6, padx=5)
        
        # Canvas for grid with scrollbars
        canvas_frame = tk.Frame(self.root, bg='white')
        canvas_frame.grid(row=1, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        
        self.canvas = tk.Canvas(canvas_frame, bg='white', width=800, height=600)
        self.h_scrollbar = tk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.v_scrollbar = tk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=self.h_scrollbar.set, yscrollcommand=self.v_scrollbar.set)
        
        self.canvas.grid(row=0, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.h_scrollbar.grid(row=1, column=0, sticky=tk.E+tk.W)
        self.v_scrollbar.grid(row=0, column=1, sticky=tk.N+tk.S)
        
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)
        
        self.root.rowconfigure(1, weight=1)
        self.root.columnconfigure(0, weight=1)
        
        # Bindings
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.canvas.bind("<Button-3>", self.on_right_click)
        self.mode_combobox.bind('<<ComboboxSelected>>', self.on_mode_change)
        self.size_combobox.bind('<<ComboboxSelected>>', self.on_size_change)
        
        self.draw_grid()

