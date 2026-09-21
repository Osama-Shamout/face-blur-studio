import os
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox

import cv2
import numpy as np

# Optional: better Windows camera-name detection
try:
    from pygrabber.dshow_graph import FilterGraph
    HAS_PYGRABBER = True
except ImportError:
    HAS_PYGRABBER = False

# Optional: virtual camera output
try:
    import pyvirtualcam
    HAS_VIRTUALCAM = True
except ImportError:
    HAS_VIRTUALCAM = False


APP_TITLE = "Face Blur Studio"
DANGER = "#ef4444"


def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _lerp_color(c1, c2, t):
    r1, g1, b1 = _hex_to_rgb(c1)
    r2, g2, b2 = _hex_to_rgb(c2)
    r = int(r1 + (r2 - r1) * t)
    g = int(g1 + (g2 - g1) * t)
    b = int(b1 + (b2 - b1) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


class FaceBlurStudio:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1180x820")
        self.root.minsize(900, 650)
        self.theme_name = "dark"
        self.themes = {
            "dark": {
                "BG": "#050505", "CARD": "#0d0d0d", "CARD_2": "#171717",
                "CARD_3": "#242424", "TEXT": "#f5f5f5", "MUTED": "#9a9a9a",
                "CYAN": "#ffffff", "ACCENT": "#8b5cf6", "ACCENT_2": "#22c55e",
                "DANGER": "#ef4444", "BORDER": "#2a2a2a"
            },
            "light": {
                "BG": "#f4f4f5", "CARD": "#ffffff", "CARD_2": "#f0f0f1",
                "CARD_3": "#e4e4e7", "TEXT": "#111111", "MUTED": "#666666",
                "CYAN": "#111111", "ACCENT": "#7c3aed", "ACCENT_2": "#16a34a",
                "DANGER": "#dc2626", "BORDER": "#d4d4d8"
            }
        }
        self.colors = self.themes[self.theme_name]
        self.root.configure(bg=self.colors["BG"])

        # ---- preview pipeline state ----
        self.preview_frame = None
        self.preview_lock = threading.Lock()
        self.latest_preview = None          # raw PPM bytes, built off the UI thread
        self.latest_preview_dims = (0, 0)
        self.preview_photo = None
        self.preview_target_size = (480, 270)  # updated live from the widget's Configure event

        self.running = False
        self.stop_event = threading.Event()
        self.worker = None
        self.cap = None
        self.vcam = None
        self.last_error = ""
        self.fps_value = 0.0
        self.face_count = 0
        # Performance: detect faces only every N frames and reuse the last result.
        # This cuts detector workload substantially while keeping the processed
        # video responsive.
        self.detect_interval = 3
        self._cached_faces = []
        self._processed_frames = 0
        self._theme_transitioning = False
        self._hover_jobs = {}

        self.flow_arrows = []
        self._anim_phase = 0.0

        self.camera_devices = []
        self.model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "face_detection_yunet_2023mar.onnx",
        )
        self.use_yunet = os.path.isfile(self.model_path)
        self.detector = None

        self.camera_var = tk.StringVar()
        # Output is automatic. Processed video is exposed through Unity Capture.
        self.output_var = tk.StringVar(value="Unity Capture")
        self.resolution_var = tk.StringVar(value="640x480")
        self.mode_var = tk.StringVar(value="Blur")
        self.shape_var = tk.StringVar(value="Ellipse")
        self.mirror_var = tk.BooleanVar(value=True)
        self.preview_var = tk.BooleanVar(value=True)
        self.auto_reconnect_var = tk.BooleanVar(value=True)

        self.blur_var = tk.DoubleVar(value=51)
        self.padding_var = tk.DoubleVar(value=25)
        self.threshold_var = tk.DoubleVar(value=0.60)
        self.pixel_var = tk.DoubleVar(value=12)
        self.brightness_var = tk.DoubleVar(value=0)
        self.contrast_var = tk.DoubleVar(value=1.0)

        self._setup_styles()
        self._build_ui()
        self.refresh_cameras()

        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._tick_status()
        self._preview_tick()
        self._animate_ui()

    # ---------- UI ----------
    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        c = self.colors
        style.configure("TCombobox", fieldbackground=c["CARD_2"], background=c["CARD_2"],
                        foreground=c["TEXT"], arrowcolor=c["TEXT"], bordercolor=c["BORDER"],
                        lightcolor=c["BORDER"], darkcolor=c["BORDER"], padding=8)
        style.map("TCombobox", fieldbackground=[("readonly", c["CARD_2"])],
                  foreground=[("readonly", c["TEXT"])])
        style.configure("TCheckbutton", background=c["CARD"], foreground=c["TEXT"],
                        font=("Segoe UI", 10))
        style.map("TCheckbutton", background=[("active", c["CARD"])],
                  foreground=[("active", c["TEXT"])])
        style.configure("TScale", background=c["CARD"], troughcolor=c["CARD_3"])
        style.map("TScale", troughcolor=[("active", c["ACCENT"])])
        style.configure("TCombobox", borderwidth=0)

    def _build_ui(self):
        c = self.colors

        # Header
        header = tk.Frame(self.root, bg=c["BG"], height=92)
        header.pack(fill="x", padx=24, pady=(16, 6))
        header.pack_propagate(False)

        title_box = tk.Frame(header, bg=c["BG"])
        title_box.pack(side="left", anchor="w")
        self.hero_title = tk.Label(title_box, text="FACE BLUR", bg=c["BG"], fg=c["TEXT"],
                                   font=("Segoe UI", 30, "bold"))
        self.hero_title.pack(anchor="w")
        self.subtitle = tk.Label(title_box,
                                 text="REAL-TIME CAMERA PROCESSING  •  DIRECT WORKFLOW",
                                 bg=c["BG"], fg=c["MUTED"], font=("Segoe UI", 9, "bold"))
        self.subtitle.pack(anchor="w")

        self.theme_btn = tk.Button(header, text="☼  LIGHT", command=self.toggle_theme,
                                   bg="#ffffff", fg="#111111", activebackground="#e5e5e5",
                                   activeforeground="#000000", relief="flat", bd=0,
                                   font=("Segoe UI", 10, "bold"), padx=18, pady=10, cursor="hand2")
        self.theme_btn.pack(side="right", padx=(10, 0), pady=14)
        self._smooth_hover(self.theme_btn, "#ffffff", "#dddddd")
        self._button_motion(self.theme_btn)

        self.status_badge = tk.Label(header, text="●  READY", bg=c["CARD_2"], fg=c["MUTED"],
                                     font=("Segoe UI", 10, "bold"), padx=16, pady=9)
        self.status_badge.pack(side="right", pady=14)

        # Main split: scrollable settings on the left, embedded preview on the right.
        main = tk.Frame(self.root, bg=c["BG"])
        main.pack(fill="both", expand=True, padx=24, pady=(0, 18))
        main.grid_columnconfigure(0, weight=3, minsize=560)
        main.grid_columnconfigure(1, weight=2, minsize=340)
        main.grid_rowconfigure(0, weight=1)

        left = tk.Frame(main, bg=c["BG"])
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        left.grid_rowconfigure(0, weight=1)
        left.grid_columnconfigure(0, weight=1)

        self.settings_canvas = tk.Canvas(left, bg=c["BG"], highlightthickness=0, bd=0)
        self.settings_canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar = tk.Scrollbar(left, orient="vertical", command=self.settings_canvas.yview,
                                 bg=c["CARD_3"], troughcolor=c["BG"], activebackground=c["MUTED"],
                                 relief="flat", bd=0, width=11)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.settings_canvas.configure(yscrollcommand=scrollbar.set)
        self.settings_inner = tk.Frame(self.settings_canvas, bg=c["BG"])
        self.settings_window = self.settings_canvas.create_window((0, 0), window=self.settings_inner,
                                                                   anchor="nw")
        self.settings_inner.bind("<Configure>", lambda e: self.settings_canvas.configure(
            scrollregion=self.settings_canvas.bbox("all")))
        self.settings_canvas.bind("<Configure>", lambda e: self.settings_canvas.itemconfigure(
            self.settings_window, width=e.width))
        self.settings_canvas.bind_all("<MouseWheel>", self._on_mousewheel, add="+")

        outer = self.settings_inner

        flow_card = self._card(outer); flow_card.pack(fill="x", pady=(0, 12))
        self._label(flow_card, "LIVE PIPELINE", 10, True, c["CYAN"]).pack(anchor="w", padx=16, pady=(12, 6))
        flow = tk.Frame(flow_card, bg=c["CARD"]); flow.pack(fill="x", padx=14, pady=(0, 13))
        self.flow_input = self._flow_chip(flow, "CAMERA", "Waiting"); self.flow_input.pack(side="left", fill="x", expand=True)
        self.flow_arrows.append(self._flow_arrow(flow)); self.flow_arrows[-1].pack(side="left", padx=7)
        self.flow_process = self._flow_chip(flow, "PROCESS", "Face Blur"); self.flow_process.pack(side="left", fill="x", expand=True)
        self.flow_arrows.append(self._flow_arrow(flow)); self.flow_arrows[-1].pack(side="left", padx=7)
        self.flow_output = self._flow_chip(flow, "OUTPUT", "Unity Capture"); self.flow_output.pack(side="left", fill="x", expand=True)

        io_card = self._card(outer); io_card.pack(fill="x", pady=(0, 12))
        self._label(io_card, "CAMERA", 10, True, c["ACCENT"]).grid(row=0, column=0, columnspan=3, sticky="w", padx=16, pady=(14, 10))
        self._label(io_card, "Input camera", 9, True, c["MUTED"]).grid(row=1, column=0, sticky="w", padx=16)
        self.camera_combo = ttk.Combobox(io_card, textvariable=self.camera_var, state="readonly", width=30)
        self.camera_combo.grid(row=2, column=0, sticky="ew", padx=16, pady=(4, 12))
        self._focus_transition(self.camera_combo)
        refresh = tk.Button(io_card, text="↻  Refresh", command=self.refresh_cameras, bg=c["CARD_2"], fg=c["TEXT"],
                            activebackground=c["CARD_3"], activeforeground=c["TEXT"], relief="flat", bd=0,
                            font=("Segoe UI", 9, "bold"), padx=13, pady=9, cursor="hand2")
        refresh.grid(row=2, column=1, padx=(0, 8), pady=(4, 12)); self._smooth_hover(refresh, c["CARD_2"], c["CARD_3"])
        self._button_motion(refresh)
        self._label(io_card, "Processed camera", 9, True, c["MUTED"]).grid(row=1, column=2, sticky="w", padx=8)
        self.output_status = tk.Label(io_card, text="AUTO • Unity Video Capture", bg=c["CARD_2"], fg=c["ACCENT_2"],
                                      font=("Segoe UI", 9, "bold"), padx=10, pady=9)
        self.output_status.grid(row=2, column=2, sticky="ew", padx=(8, 16), pady=(4, 12))
        self._label(io_card, "Resolution", 9, True, c["MUTED"]).grid(row=3, column=0, sticky="w", padx=16)
        ttk.Combobox(io_card, textvariable=self.resolution_var, state="readonly",
                     values=["1280x720", "1920x1080", "960x540", "640x480"], width=18).grid(row=4, column=0, sticky="w", padx=16, pady=(4, 14))
        self.camera_hint = tk.Label(io_card, text="Select any available camera. Iriun is preferred automatically.",
                                    bg=c["CARD"], fg=c["MUTED"], font=("Segoe UI", 9))
        self.camera_hint.grid(row=4, column=1, columnspan=2, sticky="w", padx=8, pady=(4, 14))
        io_card.columnconfigure(0, weight=1); io_card.columnconfigure(2, weight=1)

        proc = self._card(outer); proc.pack(fill="x", pady=(0, 12))
        self._label(proc, "PROCESSING", 10, True, c["ACCENT"]).grid(row=0, column=0, columnspan=4, sticky="w", padx=16, pady=(14, 8))
        self._label(proc, "Effect", 9, True, c["MUTED"]).grid(row=1, column=0, sticky="w", padx=16)
        ttk.Combobox(proc, textvariable=self.mode_var, state="readonly", values=["Blur", "Pixelate", "Solid Mask"], width=16).grid(row=2, column=0, sticky="w", padx=16, pady=(4, 10))
        self._label(proc, "Mask", 9, True, c["MUTED"]).grid(row=1, column=1, sticky="w", padx=8)
        ttk.Combobox(proc, textvariable=self.shape_var, state="readonly", values=["Ellipse", "Rectangle"], width=14).grid(row=2, column=1, sticky="w", padx=8, pady=(4, 10))
        ttk.Checkbutton(proc, text="Mirror", variable=self.mirror_var).grid(row=2, column=2, sticky="w", padx=8, pady=(4, 10))
        ttk.Checkbutton(proc, text="Show preview", variable=self.preview_var).grid(row=2, column=3, sticky="w", padx=8, pady=(4, 10))
        self._slider(proc, "Blur intensity", self.blur_var, 3, 99, 3, 3)
        self._slider(proc, "Face padding", self.padding_var, 0, 100, 1, 5)
        self._slider(proc, "Detection threshold", self.threshold_var, 0.30, 0.90, 0.01, 7)
        self._slider(proc, "Pixel size", self.pixel_var, 2, 30, 1, 9)
        for col in range(4): proc.columnconfigure(col, weight=1)

        adj = self._card(outer); adj.pack(fill="x", pady=(0, 12))
        self._label(adj, "IMAGE", 10, True, c["ACCENT"]).grid(row=0, column=0, columnspan=2, sticky="w", padx=16, pady=(14, 8))
        self._slider(adj, "Brightness", self.brightness_var, -60, 60, 1, 1)
        self._slider(adj, "Contrast", self.contrast_var, 0.50, 1.80, 0.05, 3)

        controls = tk.Frame(outer, bg=c["BG"]); controls.pack(fill="x", pady=(2, 10))
        self.start_btn = tk.Button(controls, text="▶  START CAMERA", command=self.toggle, bg=c["ACCENT"], fg="white",
                                   activebackground=c["ACCENT"], activeforeground="white", relief="flat", bd=0,
                                   font=("Segoe UI", 12, "bold"), padx=30, pady=13, cursor="hand2")
        self.start_btn.pack(side="left"); self._smooth_hover(self.start_btn, c["ACCENT"], "#a78bfa")
        self._button_motion(self.start_btn)
        ttk.Checkbutton(controls, text="Auto reconnect", variable=self.auto_reconnect_var).pack(side="left", padx=18)
        self.stats = tk.Label(controls, text="0.0 FPS  •  0 faces", bg=c["BG"], fg=c["MUTED"], font=("Consolas", 10))
        self.stats.pack(side="right")

        footer = self._card(outer); footer.pack(fill="x", pady=(0, 20))
        self._label(footer, "DIRECT MODE", 10, True, c["CYAN"]).pack(anchor="w", padx=16, pady=(12, 5))
        tk.Label(footer, text=("No OBS is required. Face Blur Studio processes the selected camera and exposes the result "
                               "through Unity Video Capture. Discord/OmeTV can select that processed camera directly."),
                 bg=c["CARD"], fg=c["MUTED"], justify="left", wraplength=650, font=("Segoe UI", 9)).pack(fill="x", padx=16, pady=(0, 13))

        # Embedded preview panel
        preview_card = self._card(main)
        preview_card.grid(row=0, column=1, sticky="nsew")
        preview_card.grid_rowconfigure(1, weight=1); preview_card.grid_columnconfigure(0, weight=1)
        self._label(preview_card, "LIVE PREVIEW", 11, True, c["CYAN"]).grid(row=0, column=0, sticky="w", padx=18, pady=(16, 10))
        preview_area = tk.Frame(preview_card, bg="#000000", highlightthickness=2, highlightbackground=c["BORDER"])
        preview_area.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 12))
        preview_area.grid_rowconfigure(0, weight=1); preview_area.grid_columnconfigure(0, weight=1)
        preview_area.bind("<Configure>", self._on_preview_area_configure)
        self.preview_area = preview_area
        self.preview_frame = tk.Label(preview_area, text="CAMERA OFFLINE\n\nPress START CAMERA", bg="#000000", fg="#888888",
                                      font=("Segoe UI", 13, "bold"), justify="center")
        self.preview_frame.grid(row=0, column=0, sticky="nsew")
        self.preview_info = tk.Label(preview_card, text="Waiting for camera...", bg=c["CARD"], fg=c["MUTED"],
                                     font=("Consolas", 9))
        self.preview_info.grid(row=2, column=0, sticky="ew", padx=16, pady=(0, 16))

        self._update_flow("Waiting", "Face Blur", "Unity Capture")

    def _on_preview_area_configure(self, event):
        # Cheap, event-driven: only updates when the panel is actually resized,
        # so the capture thread always knows the right size to downscale to
        # without the UI thread doing any per-frame work.
        self.preview_target_size = (max(160, event.width), max(90, event.height))

    def _on_mousewheel(self, event):
        try:
            if self.settings_canvas.winfo_containing(event.x_root, event.y_root) is not None:
                self.settings_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        except tk.TclError:
            pass

    def _label(self, parent, text, size=10, bold=False, color=None):
        """Create a themed Tk label that follows the current app theme."""
        if color is None:
            color = self.colors["TEXT"]
        return tk.Label(
            parent,
            text=text,
            bg=parent.cget("bg"),
            fg=color,
            font=("Segoe UI", size, "bold" if bold else "normal"),
        )

    def _card(self, parent):
        """Create a themed card container used throughout the interface."""
        c = self.colors
        return tk.Frame(
            parent,
            bg=c["CARD"],
            highlightthickness=1,
            highlightbackground=c["BORDER"],
            highlightcolor=c["BORDER"],
            bd=0,
        )

    def _flow_chip(self, parent, title, value):
        c = self.colors
        box = tk.Frame(parent, bg=c["CARD_2"], highlightthickness=1, highlightbackground=c["BORDER"])
        tk.Label(box, text=title, bg=c["CARD_2"], fg=c["MUTED"], font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=10, pady=(7, 0))
        label = tk.Label(box, text=value, bg=c["CARD_2"], fg=c["TEXT"], font=("Segoe UI", 9, "bold"))
        label.pack(anchor="w", padx=10, pady=(1, 7)); box.value_label = label
        self._smooth_hover(box, c["CARD_2"], c["CARD_3"], steps=5, delay=10)
        self._smooth_hover(label, c["CARD_2"], c["CARD_3"], steps=5, delay=10)
        return box

    def _flow_arrow(self, parent):
        return tk.Label(parent, text="›", bg=self.colors["CARD"], fg=self.colors["ACCENT"], font=("Segoe UI", 20, "bold"))

    def _update_flow(self, camera, process, output):
        if hasattr(self, "flow_input"):
            self.flow_input.value_label.config(text=camera)
            self.flow_process.value_label.config(text=process)
            self.flow_output.value_label.config(text=output)

    def _smooth_hover(self, widget, normal, hover, steps=8, delay=12):
        """Eased color-fade hover transition instead of an instant color swap."""
        state = {"job": None}

        def animate(step, entering):
            t = step / steps
            t = t if entering else 1 - t
            try:
                widget.config(bg=_lerp_color(normal, hover, t))
            except tk.TclError:
                return
            if step < steps:
                state["job"] = widget.after(delay, lambda: animate(step + 1, entering))

        def on_enter(_e):
            if state["job"]:
                widget.after_cancel(state["job"])
            animate(0, True)

        def on_leave(_e):
            if state["job"]:
                widget.after_cancel(state["job"])
            animate(0, False)

        widget.bind("<Enter>", on_enter)
        widget.bind("<Leave>", on_leave)

    def _button_motion(self, widget):
        """Small press/release motion for a more modern desktop UI."""
        base_y = widget.winfo_y()

        def press(_event):
            try:
                widget.place_configure(y=base_y + 1) if widget.winfo_manager() == "place" else None
                widget.config(relief="sunken")
            except tk.TclError:
                pass

        def release(_event):
            try:
                widget.place_configure(y=base_y) if widget.winfo_manager() == "place" else None
                widget.config(relief="flat")
            except tk.TclError:
                pass

        widget.bind("<ButtonPress-1>", press, add="+")
        widget.bind("<ButtonRelease-1>", release, add="+")

    def _focus_transition(self, widget):
        """Subtle focus highlight for comboboxes and other interactive controls."""
        try:
            widget.bind("<FocusIn>", lambda _e: self._set_widget_border(widget, self.colors["ACCENT"]), add="+")
            widget.bind("<FocusOut>", lambda _e: self._set_widget_border(widget, self.colors["BORDER"]), add="+")
        except tk.TclError:
            pass

    @staticmethod
    def _set_widget_border(widget, color):
        try:
            widget.config(highlightbackground=color, highlightcolor=color)
        except tk.TclError:
            pass

    def _theme_transition(self):
        """Safely switch themes with a short UI feedback animation.

        Tkinter widgets are not alpha-composited, so a full-window Canvas
        overlay can hide the application and appear as a black screen.
        Instead, change the theme normally and animate the theme button/header
        accent for a polished transition without covering the UI.
        """
        if self._theme_transitioning:
            return

        self._theme_transitioning = True
        old_name = self.theme_name
        new_name = "light" if old_name == "dark" else "dark"

        self.theme_btn.config(state="disabled")

        # Switch the actual theme immediately so the window is never covered
        # by an opaque animation layer.
        self.theme_name = new_name
        self.colors = self.themes[new_name]
        self._setup_styles()
        self._retheme(self.root)

        if new_name == "dark":
            normal = "#ffffff"
            hover = "#dddddd"
            self.theme_btn.config(
                text="☼  LIGHT", bg=normal, fg="#111111",
                activebackground=hover, activeforeground="#000000"
            )
        else:
            normal = "#111111"
            hover = "#2a2a2a"
            self.theme_btn.config(
                text="☾  DARK", bg=normal, fg="#ffffff",
                activebackground=hover, activeforeground="#ffffff"
            )

        self._smooth_hover(self.theme_btn, normal, hover)

        # Small modern accent pulse after the theme change. This gives the
        # click a visible transition without blocking or hiding the window.
        base_accent = self.colors["ACCENT"]
        pulse_steps = 10

        def pulse(i=0):
            try:
                if i <= pulse_steps:
                    t = i / pulse_steps
                    # Fade from a brighter accent back to the normal accent.
                    bright = _lerp_color(base_accent, self.colors["CYAN"], 0.45)
                    color = _lerp_color(bright, base_accent, t)
                    self.theme_btn.config(highlightthickness=2, highlightbackground=color)
                    self.root.after(18, lambda: pulse(i + 1))
                else:
                    self.theme_btn.config(highlightthickness=0)
                    self.theme_btn.config(state="normal")
                    self._theme_transitioning = False
            except tk.TclError:
                self._theme_transitioning = False

        pulse()

    def _slider(self, parent, title, variable, low, high, resolution, row):
        base_row = row + 2
        tk.Label(parent, text=title, bg=parent.cget("bg"), fg=self.colors["MUTED"],
                 font=("Segoe UI", 9, "bold")).grid(row=base_row, column=0, sticky="w", padx=16, pady=(3, 0))
        value_label = tk.Label(parent, text="", bg=parent.cget("bg"), fg=self.colors["TEXT"], font=("Consolas", 9))
        value_label.grid(row=base_row, column=1, sticky="e", padx=16)
        scale = ttk.Scale(parent, from_=low, to=high, variable=variable, orient="horizontal",
                          command=lambda _v, lbl=value_label, var=variable: self._set_value_label(lbl, var))
        scale.grid(row=base_row + 1, column=0, columnspan=2, sticky="ew", padx=16, pady=(0, 8))
        self._set_value_label(value_label, variable)
        parent.columnconfigure(0, weight=1); parent.columnconfigure(1, weight=1)

    @staticmethod
    def _set_value_label(label, variable):
        v = variable.get()
        label.config(text=f"{v:.2f}" if abs(v - round(v)) > 0.001 else str(int(v)))

    def toggle_theme(self):
        self._theme_transition()

    def _retheme(self, widget):
        """Update native Tk widgets without rebuilding the UI."""
        old = self.themes["light" if self.theme_name == "dark" else "dark"]
        new = self.colors
        mapping = {v: new[k] for k, v in old.items()}
        try:
            bg = widget.cget("bg")
            if bg in mapping:
                widget.config(bg=mapping[bg])
            fg = widget.cget("fg")
            if fg in mapping:
                widget.config(fg=mapping[fg])
            if widget.cget("highlightbackground") in mapping:
                widget.config(highlightbackground=mapping[widget.cget("highlightbackground")])
            if widget.winfo_class() == "Canvas":
                widget.config(bg=new["BG"])
        except (tk.TclError, TypeError):
            pass
        for child in widget.winfo_children():
            self._retheme(child)

    # ---------- Camera discovery ----------
    def refresh_cameras(self):
        devices = []

        if HAS_PYGRABBER:
            try:
                graph = FilterGraph()
                names = graph.get_input_devices()
                devices = [(i, name) for i, name in enumerate(names)]
            except Exception:
                devices = []

        # If pygrabber is unavailable, probe DirectShow indexes.
        if not devices:
            for i in range(12):
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if cap.isOpened():
                    ok, frame = cap.read()
                    cap.release()
                    if ok and frame is not None:
                        devices.append((i, f"Camera {i}"))
                else:
                    cap.release()

        self.camera_devices = devices
        labels = [f"{name}  [#{idx}]" for idx, name in devices]

        self.camera_combo["values"] = labels

        # Prefer Iriun automatically.
        preferred = None
        for pos, (idx, name) in enumerate(devices):
            if "iriun" in name.lower():
                preferred = pos
                break

        if preferred is not None:
            self.camera_combo.current(preferred)
        elif labels:
            self.camera_combo.current(0)
        else:
            self.camera_var.set("No camera detected")

    def selected_camera_index(self):
        selected = self.camera_var.get()
        if not selected:
            return None
        try:
            return int(selected.rsplit("[#", 1)[1].rstrip("]"))
        except Exception:
            # Fall back to current combo index.
            pos = self.camera_combo.current()
            if 0 <= pos < len(self.camera_devices):
                return self.camera_devices[pos][0]
        return None

    # ---------- Camera / processed output ----------
    # A normal webcam device cannot be modified in-place by OpenCV. Other
    # applications need a separate Windows camera endpoint for the processed
    # frames, so Unity Capture is used automatically and invisibly in this UI.
    def open_camera(self):
        idx = self.selected_camera_index()
        if idx is None:
            raise RuntimeError("No camera selected. Start Iriun Webcam and click Refresh.")

        width, height = map(int, self.resolution_var.get().split("x"))

        # DirectShow is generally the best choice for named Windows webcam drivers.
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(idx, cv2.CAP_MSMF)

        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open camera #{idx}. Close other apps that may be using Iriun."
            )

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        # Keep the driver buffer as shallow as possible so we always read the
        # freshest frame instead of a queued, stale one (this is the single
        # biggest lever for perceived latency with DirectShow webcams).
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass

        # Warm up the driver.
        frame = None
        for _ in range(10):
            ok, frame = cap.read()
            if ok and frame is not None:
                break
            time.sleep(0.05)

        if frame is None:
            cap.release()
            raise RuntimeError("Camera opened but did not provide video frames.")

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or width
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or height
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps < 5 or fps > 120:
            fps = 30

        return cap, actual_w, actual_h, float(fps)

    def open_virtual_camera(self, width, height, fps):
        output = self.output_var.get()

        if output == "None / Preview Only":
            return None

        if not HAS_VIRTUALCAM:
            raise RuntimeError(
                "pyvirtualcam is not installed. Run: pip install -U pyvirtualcam"
            )

        # Unity Capture is the automatic output. We intentionally do not
        # hard-code "Unity Video Capture" as the device name because some
        # installations register a custom name. The backend can select its
        # first available Unity Capture device.
        # Browser compatibility profile:
        # Some WebRTC chat sites are much stricter than normal camera apps and
        # reject/ignore virtual cameras when they expose unusual resolutions or
        # high-resolution modes. Keep the virtual endpoint at a very common
        # webcam format while the app can still process the source at full
        # resolution.
        self.vcam_width = 640
        self.vcam_height = 480
        self.vcam_fps = 30

        # Windows/Chrome compatibility: Unity Capture is the only pyvirtualcam
        # backend we use here because pyvirtualcam cannot write processed frames
        # directly into a physical Iriun device. The app therefore keeps the
        # virtual endpoint deliberately boring: 640x480, RGB, 30 FPS.
        try:
            cam = pyvirtualcam.Camera(
                width=self.vcam_width,
                height=self.vcam_height,
                fps=self.vcam_fps,
                fmt=pyvirtualcam.PixelFormat.RGB,
                backend="unitycapture",
            )
            return cam
        except Exception as exc:
            raise RuntimeError(
                "Could not open the Unity Capture virtual camera.\n\n"
                "IMPORTANT: Iriun Webcam itself cannot be used as the processed "
                "output device because OpenCV cannot write modified frames back "
                "into another camera driver's output stream. A virtual-camera "
                "driver is required for the processed video.\n\n"
                "Make sure Unity Capture is installed and registered, and close "
                "OBS/Zoom/Teams/Chrome camera pages while testing.\n\n"
                f"Technical error: {exc}"
            ) from exc

    # ---------- Detection / effects ----------
    def make_detector(self, width, height):
        if self.use_yunet:
            return cv2.FaceDetectorYN.create(
                self.model_path,
                "",
                (width, height),
                0.60,
                0.30,
                5000,
            )

        return cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def detect_faces(self, image, detector):
        h, w = image.shape[:2]
        threshold = float(self.threshold_var.get())

        if self.use_yunet:
            detector.setInputSize((w, h))
            detector.setScoreThreshold(threshold)
            _, faces = detector.detect(image)
            if faces is None:
                return []
            return [
                tuple(map(int, face[:4]))
                for face in faces
                if face[2] > 0 and face[3] > 0
            ]

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.10,
            minNeighbors=5,
            minSize=(45, 45),
        )
        return [tuple(map(int, f)) for f in faces]

    def apply_effect(self, image, box):
        x, y, w, h = box
        pad = int(self.padding_var.get())

        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(image.shape[1], x + w + pad)
        y2 = min(image.shape[0], y + h + pad)

        if x2 <= x1 or y2 <= y1:
            return

        roi = image[y1:y2, x1:x2]
        rh, rw = roi.shape[:2]

        mode = self.mode_var.get()
        shape = self.shape_var.get()

        if mode == "Blur":
            k = int(self.blur_var.get())
            k = max(3, k)
            if k % 2 == 0:
                k += 1
            filtered = cv2.GaussianBlur(roi, (k, k), 0)
        elif mode == "Pixelate":
            p = max(2, int(self.pixel_var.get()))
            small = cv2.resize(
                roi,
                (max(1, rw // p), max(1, rh // p)),
                interpolation=cv2.INTER_LINEAR,
            )
            filtered = cv2.resize(
                small,
                (rw, rh),
                interpolation=cv2.INTER_NEAREST,
            )
        else:
            filtered = np.zeros_like(roi)

        if shape == "Rectangle":
            image[y1:y2, x1:x2] = filtered
            return

        # Elliptical mask with soft edge.
        mask = np.zeros((rh, rw), dtype=np.uint8)
        cv2.ellipse(
            mask,
            (rw // 2, rh // 2),
            (max(1, rw // 2), max(1, rh // 2)),
            0,
            0,
            360,
            255,
            -1,
        )
        mask = cv2.GaussianBlur(mask, (0, 0), 4)
        alpha = mask.astype(np.float32) / 255.0
        alpha = alpha[..., None]

        base = roi.astype(np.float32)
        fx = filtered.astype(np.float32)
        image[y1:y2, x1:x2] = np.clip(
            base * (1.0 - alpha) + fx * alpha, 0, 255
        ).astype(np.uint8)

    def adjust_image(self, image):
        brightness = float(self.brightness_var.get())
        contrast = float(self.contrast_var.get())

        if abs(brightness) > 0.1 or abs(contrast - 1.0) > 0.01:
            image = cv2.convertScaleAbs(
                image,
                alpha=contrast,
                beta=brightness,
            )
        return image

    def _build_preview_ppm(self, image):
        """Downscale + convert to a raw PPM buffer entirely off the UI thread.

        This replaces the old approach of PNG-encoding a full-res frame on
        the Tk main loop every ~33ms, which was the main source of lag: PNG
        compression is expensive and it was blocking the UI thread. Raw PPM
        needs zero compression, so tk.PhotoImage can parse it almost
        instantly on the main thread.
        """
        tw, th = self.preview_target_size
        h, w = image.shape[:2]
        scale = min(tw / w, th / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        small = cv2.resize(image, (nw, nh), interpolation=interp)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        header = f"P6 {nw} {nh} 255 ".encode("ascii")
        return header + rgb.tobytes(), (nw, nh)

    # ---------- Streaming ----------
    def toggle(self):
        if self.running:
            self.stop()
        else:
            self.start()

    def start(self):
        if self.running:
            return

        self.last_error = ""
        self._cached_faces = []
        self._processed_frames = 0
        self.stop_event.clear()
        self.running = True
        self.start_btn.config(text="STOP CAMERA", bg=DANGER)
        self._set_status("●  STARTING", "#f59e0b")

        self.worker = threading.Thread(
            target=self.video_loop,
            name="FaceBlurVideo",
            daemon=True,
        )
        self.worker.start()

    def stop(self):
        self.stop_event.set()
        self.running = False
        self.start_btn.config(text="START CAMERA", bg=self.colors["ACCENT"])
        self._set_status("●  STOPPING", "#f59e0b")

    def _set_status(self, text, color):
        self.root.after(
            0,
            lambda: self.status_badge.config(text=text, fg=color),
        )

    def _show_error(self, title, text):
        self.root.after(0, lambda: messagebox.showerror(title, text))

    def video_loop(self):
        cap = None
        vcam = None

        try:
            cap, width, height, fps = self.open_camera()
            self.cap = cap
            detector = self.make_detector(width, height)

            # Unity Capture receives the already-processed frames. Keep this
            # endpoint at 640x480/30 because many WebRTC pages are more
            # reliable with conservative webcam modes.
            vcam = self.open_virtual_camera(width, height, fps)
            self.vcam = vcam

            accent_2 = self.colors["ACCENT_2"]
            if vcam:
                output_name = getattr(vcam, "device", "Unity Video Capture")
                self.output_name = output_name
                self.root.after(
                    0,
                    lambda: self.output_status.config(
                        text=f"AUTO • {output_name}",
                        fg=accent_2,
                    ),
                )
                self._set_status(f"●  LIVE • {output_name}", accent_2)
            else:
                self._set_status("●  LIVE • PREVIEW", accent_2)

            frame_counter = 0
            fps_started = time.perf_counter()

            while not self.stop_event.is_set():
                ok, image = cap.read()

                if not ok or image is None:
                    if self.auto_reconnect_var.get() and not self.stop_event.is_set():
                        time.sleep(0.25)
                        continue
                    break

                if self.mirror_var.get():
                    image = cv2.flip(image, 1)

                # Run the detector every second frame and reuse the last
                # detection in between. This is a major FPS/latency improvement
                # on typical webcams while still updating faces frequently.
                self._processed_frames += 1
                if self._processed_frames % self.detect_interval == 0:
                    self._cached_faces = self.detect_faces(image, detector)
                faces = self._cached_faces

                for face in faces:
                    self.apply_effect(image, face)

                image = self.adjust_image(image)
                self.face_count = len(faces)

                if vcam:
                    # Always feed the virtual camera a standard webcam frame
                    # (640x480 @ 30 FPS). This improves compatibility with
                    # WebRTC sites that fail to initialize non-standard virtual
                    # camera modes even though the browser can enumerate them.
                    output_frame = cv2.resize(
                        image,
                        (self.vcam_width, self.vcam_height),
                        interpolation=cv2.INTER_AREA,
                    )
                    rgb = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
                    rgb = np.ascontiguousarray(rgb)
                    vcam.send(rgb)
                    vcam.sleep_until_next_frame()

                if self.preview_var.get():
                    ppm_bytes, dims = self._build_preview_ppm(image)
                    with self.preview_lock:
                        self.latest_preview = ppm_bytes
                        self.latest_preview_dims = dims
                else:
                    with self.preview_lock:
                        self.latest_preview = None

                frame_counter += 1
                elapsed = time.perf_counter() - fps_started
                if elapsed >= 0.75:
                    self.fps_value = frame_counter / elapsed
                    frame_counter = 0
                    fps_started = time.perf_counter()

        except Exception as exc:
            self.last_error = str(exc)
            self._show_error("Camera / Virtual Camera Error", self.last_error)

        finally:
            try:
                if cap is not None:
                    cap.release()
            except Exception:
                pass

            try:
                if vcam is not None:
                    vcam.close()
            except Exception:
                pass

            self.cap = None
            self.vcam = None
            with self.preview_lock:
                self.latest_preview = None
            self.preview_photo = None
            self.root.after(0, self._clear_preview)

            self.running = False
            self.root.after(
                0,
                lambda: (
                    self.start_btn.config(text="START CAMERA", bg=self.colors["ACCENT"]),
                    self.status_badge.config(text="●  READY", fg=self.colors["MUTED"]),
                ),
            )

    # ---------- Animation ----------
    def _animate_ui(self):
        try:
            self._anim_phase = (self._anim_phase + 0.09) % 6.28318
            pulse = (np.sin(self._anim_phase) + 1.0) / 2.0

            if self.theme_name == "dark":
                r = int(210 + 45 * pulse); g = int(210 + 45 * pulse); b = int(210 + 45 * pulse)
            else:
                r = int(60 + 35 * pulse); g = int(60 + 35 * pulse); b = int(60 + 35 * pulse)
            if hasattr(self, "hero_title"):
                self.hero_title.config(fg=f"#{r:02x}{g:02x}{b:02x}")

            c = self.colors
            if self.running:
                if hasattr(self, "status_badge"):
                    live_bg = "#102719" if self.theme_name == "dark" else "#dcfce7"
                    self.status_badge.config(bg=live_bg)
                # Pulsing glow around the live preview panel.
                if hasattr(self, "preview_area"):
                    glow = _lerp_color(c["ACCENT"], c["CYAN"], pulse)
                    self.preview_area.config(highlightbackground=glow)
                # Pulsing flow arrows to suggest data actively moving through the pipeline.
                for arrow in self.flow_arrows:
                    arrow.config(fg=_lerp_color(c["ACCENT"], c["ACCENT_2"], pulse))
            else:
                if hasattr(self, "preview_area"):
                    self.preview_area.config(highlightbackground=c["BORDER"])
                for arrow in self.flow_arrows:
                    arrow.config(fg=c["ACCENT"])
        except tk.TclError:
            return
        self.root.after(20, self._animate_ui)

    def _preview_tick(self):
        if not hasattr(self, "preview_frame"):
            return
        try:
            with self.preview_lock:
                data = self.latest_preview
                dims = self.latest_preview_dims

            if data is not None and self.preview_var.get():
                nw, nh = dims
                # Parsing raw PPM data is cheap; no per-frame resize/encode
                # happens on the UI thread anymore.
                photo = tk.PhotoImage(width=nw, height=nh, data=data, format="PPM")
                self.preview_photo = photo
                self.preview_frame.config(image=photo, text="", bg="#000000")
                self.preview_info.config(
                    text=f"LIVE  •  {self.fps_value:.1f} FPS  •  {self.face_count} face"
                    + ("" if self.face_count == 1 else "s")
                )
            elif not self.running:
                self.preview_frame.config(image="", text="CAMERA OFFLINE\n\nPress START CAMERA", fg="#888888", bg="#000000")
                self.preview_info.config(text="Waiting for camera...")
        except (tk.TclError, ValueError):
            pass
        self.root.after(33, self._preview_tick)

    def _clear_preview(self):
        if hasattr(self, "preview_frame"):
            self.preview_frame.config(image="", text="CAMERA OFFLINE\n\nPress START CAMERA", fg="#888888", bg="#000000")
        if hasattr(self, "preview_info"):
            self.preview_info.config(text="Waiting for camera...")

    def _tick_status(self):
        self.stats.config(
            text=f"{self.fps_value:.1f} FPS  •  {self.face_count} face"
            + ("" if self.face_count == 1 else "s")
        )
        self.root.after(300, self._tick_status)

    def close(self):
        self.stop_event.set()
        self.running = False

        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass

        try:
            if self.vcam is not None:
                self.vcam.close()
        except Exception:
            pass

        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = FaceBlurStudio(root)
    root.mainloop()