"""
Simple matplotlib-based labeling tool for parking spaces segmentation.
Windows-compatible, no complex dependencies.

Usage:
    python scripts/labeling_tool_simple.py
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
from scipy import ndimage
from skimage.segmentation import felzenszwalb
import time
import threading
import shutil
import os
import labeling.core.segment as segment_core

# Import from existing codebase
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from app.core.boundary import estimate_boundary_from_overlay


# Semantic class definitions
CLASSES = [
    "parking_stalls",   # 1. Individual parking stalls
    "driveway_road",    # 2. Driveway / Road (replaces parking_access)
    "building",         # 3. All building-related labels combined
    "vegetation",       # 4.
    "sidewalk",         # 5.
    "pool_equipment",   # 6. Pool / Equipment areas
]

# Human-friendly display names for radio buttons (allows slashes and nicer labels)
DISPLAY_NAMES = {
    "parking_stalls": "Parking Stalls",
    "driveway_road": "Driveway/Road",
    "building": "Building",
    "vegetation": "Vegetation",
    "sidewalk": "Sidewalk",
    "pool_equipment": "Pool/Equipment",
}

CLASS_COLORS = {
    "parking_stalls": [0.39, 0.58, 0.93], # Cornflower blue - parking stalls
    "driveway_road": [1.0, 0.84, 0.0],   # Gold - driveway/road
    "building": [0.85, 0.3, 0.3],         # Building - warm red
    "sidewalk": [0.78, 0.78, 0.78],       # Light gray
    "vegetation": [0.13, 0.55, 0.13],     # Forest green
    "pool_equipment": [0.6, 0.4, 0.8],    # Purple-ish for pool/equipment
} 


class LabelingTool:
    """Main labeling tool application."""
    
    CONFIG_FILE = Path(__file__).parent.parent / "data" / ".labeling_tool_config.json"
    
    def __init__(self):
        try:
            self.root = tk.Tk()
            self.root.title("Parking Spaces Labeling Tool")
            self.root.geometry("1400x950")
            # Prevent users from resizing too small and causing widgets to be clipped
            self.root.minsize(1100, 800)
            self._tk_available = True
        except Exception:
            # Running in headless/test environment without Tk available. Use a minimal stub.
            from types import SimpleNamespace
            def _noop(*a, **kw):
                return None
            stub = SimpleNamespace()
            stub.withdraw = _noop
            stub.title = _noop
            stub.geometry = _noop
            stub.minsize = _noop
            stub.after = _noop
            stub.mainloop = _noop
            self.root = stub
            self._tk_available = False
        
        # State
        self.images_folder: Optional[Path] = None
        self.output_folder: Optional[Path] = None
        self.last_input_folder: Optional[Path] = None  # Track for parent navigation
        self.last_output_folder: Optional[Path] = None  # Track for parent navigation
        self.image_files: List[Path] = []
        self.current_idx: int = 0
        
        # Current image state
        self.current_image: Optional[np.ndarray] = None  # Original BGR with red border
        self.clean_image: Optional[np.ndarray] = None  # BGR with red inpainted out
        self.current_boundary: Optional[np.ndarray] = None
        self.original_boundary: Optional[np.ndarray] = None  # Store detected boundary
        self.boundary_approved: bool = False
        self.segments: Optional[np.ndarray] = None
        self.segment_labels: Dict[int, str] = {}
        # Boundary access state
        self.boundary_access_mode: bool = False
        self.access_click_start: Optional[Dict] = None  # Stores {'frac': float, 'pt': (x,y)}
        self.boundary_access_segments: List[Dict] = []  # Each: {start_frac, end_frac, label, start_px, end_px}
        # Buffer settings: buffer outside boundary will be included in segmentation and labeling
        self.road_buffer_px: int = 96
        self.road_buffer_pct: float = 0.05
        self.buffer_mode: str = "px"  # 'px' or 'pct'
        # Note: road painting feature removed; buffer area is included in segmentation instead
        
        # Export mapping for interior combined classes
        # 0=ignore,1=parking_or_road,2=building_immobile,3=vegetation
        self.interior_export_map = {
            'parking_stalls': 1,
            'driveway_road': 1,   # driveways/roads contribute to parking_or_road class
            'sidewalk': 1,
            'building': 2,
            'vegetation': 3,
            'pool_equipment': 0   # pool/equipment - ignored by default in interior export
        }
        # Legacy field preserved for backward compatibility (kept for save/load)
        self.road_access_lines: List[Dict] = []
        self.n_segments: int = 50  # Actual segment count
        self.target_segments: int = 50  # User's desired target
        self.selected_class: str = CLASSES[0]
        # Collapsible section frames registry
        self.section_frames: Dict[str, Tuple[tk.Frame, tk.Button]] = {}

        # Drag redraw throttle (time in seconds)
        self._last_drag_render_time: float = 0.0
        self._drag_render_interval: float = 0.03  # 30 ms between heavy redraws

        # Toggle profiling during drag to find bottlenecks (default False)
        self._profile_drag: bool = False
        # Coalesced draw state
        self._pending_draw: bool = False
        self._coalesced_draw_delay_ms: int = 30
        
        # Preserve labels when editing boundary
        self.preserved_labels: Optional[Dict[int, str]] = None
        
        # Boundary transformation
        self.boundary_dx: float = 0.0
        self.boundary_dy: float = 0.0
        self.boundary_theta: float = 0.0  # degrees
        self.boundary_vertices_artists: List = []
        
        # Drag state for boundary transformation
        self.dragging_boundary: bool = False
        self.drag_start_pos: Optional[Tuple[float, float]] = None
        self.drag_start_boundary: Optional[np.ndarray] = None
        
        # Vertex dragging removed - only whole-boundary translate/rotate supported
        self.dragging_vertex: bool = False  # kept for compatibility (unused)
        self.dragging_vertex_idx: Optional[int] = None  # kept for compatibility (unused) 
        
        # Manual segmentation mode
        self.manual_mode: bool = False
        self.manual_line_points: List[Tuple[int, int]] = []  # Current polyline being drawn
        self.manual_polylines: List[List[Tuple[int, int]]] = []  # List of completed polylines
        self.manual_line_artists = []  # Artists for all polylines
        self.split_history: List[np.ndarray] = []  # Stack of previous segment states for undo
        self.splitting_segment_id: Optional[int] = None  # Currently selected segment for splitting

        # Road painting (removed from main UI but keep backend stubs for compatibility/tests)
        try:
            self.road_paint_var = tk.StringVar(value='public_road')
        except Exception:
            self.road_paint_var = None
        self.road_mask = None
        self.road_mode_active = False
        self.painting_road = False
        self.paint_erase_mode = False
        self.road_brush_size = 15
        
        # Display enhancement (cached)
        self.enhanced_image: Optional[np.ndarray] = None
        
        # Track last clicked segment for visual feedback
        self.last_clicked_segment: Optional[int] = None
        
        # Track unsaved changes
        self.has_unsaved_changes: bool = False
        
        # Build UI only when a real Tk root was successfully created. In headless/test environments
        # we skip building Tk widgets and provide minimal placeholders for attributes used in logic/tests.
        if self._tk_available:
            self._build_ui()
            # Load last used folders and auto-load if valid (deferred to after UI is ready)
            self.root.after(100, self._load_folder_preferences)
        else:
            # Headless stub - create minimal placeholders to avoid attribute errors in tests
            from types import SimpleNamespace
            self.mode = 'boundary'
            self.mode_buttons = {}
            self.section_frames = {}
            self.footer_frame = SimpleNamespace()
            # Simple submit_frame stub with pack methods
            self.submit_frame = SimpleNamespace(pack=lambda *a, **k: None, pack_forget=lambda *a, **k: None)
            self.canvas = SimpleNamespace(get_tk_widget=lambda: None, draw=lambda: None, draw_idle=lambda: None)
            # Minimal StringVar-like stubs for flags and controls used in logic
            self.buffer_mode_var = SimpleNamespace(get=lambda: 'px')
            self.segment_smoothing_var = SimpleNamespace(get=lambda: 'med')
            self.shadow_robust_var = SimpleNamespace(get=lambda: False)
            self.pre_smooth_var = SimpleNamespace(get=lambda: False)
            self.auto_merge_var = SimpleNamespace(get=lambda: False)
            # Do not attempt to auto-load preferences in headless mode
        
    def _build_ui(self):
        """Build the user interface."""
        # Left panel container with scrollbar (widened to avoid clipping)
        left_container = tk.Frame(self.root, width=400, bg='#f0f0f0')
        left_container.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        left_container.pack_propagate(False)
        # make accessible for toggle button
        self.left_container = left_container
        self._left_panel_expanded = False
        
        # Top toolbar (always visible) to host critical quick actions like Save Draft
        top_toolbar = tk.Frame(left_container, bg='#f0f0f0')
        top_toolbar.pack(side=tk.TOP, fill='x')
        # Add quick Save Draft button in top toolbar (duplicate of footer save to ensure visibility)
        self.quick_save_btn = tk.Button(top_toolbar, text="💾 Save Draft", command=self._save_draft, bg='#E0E0E0', width=12)
        self.quick_save_btn.pack(side=tk.RIGHT, padx=6, pady=2)
        # Add left panel expand toggle so users can enlarge the left panel if controls are obscured
        def _toggle_left_panel():
            try:
                if not getattr(self, '_left_panel_expanded', False):
                    self.left_container.config(width=540)
                    self._left_panel_expanded = True
                    self.quick_expand_btn.config(text='⇤')
                else:
                    self.left_container.config(width=400)
                    self._left_panel_expanded = False
                    self.quick_expand_btn.config(text='⇥')
                # force re-layout
                try:
                    self.root.update_idletasks()
                except Exception:
                    pass
            except Exception:
                pass
        self.quick_expand_btn = tk.Button(top_toolbar, text='⇥', command=_toggle_left_panel, bg='#F0F0F0', width=3)
        self.quick_expand_btn.pack(side=tk.RIGHT, padx=(0,4), pady=2)

        # Canvas and scrollbar for left panel
        canvas = tk.Canvas(left_container, bg='#f0f0f0', highlightthickness=0)
        scrollbar = tk.Scrollbar(left_container, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='#f0f0f0')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Footer frame (fixed) below the scrollable content - submit always visible here
        footer_frame = tk.Frame(left_container, bg='#f0f0f0')
        footer_frame.pack(side=tk.BOTTOM, fill='x')
        self.footer_frame = footer_frame
        
        # Use scrollable_frame as the parent for all controls
        left_panel = scrollable_frame
        
        # === FILE SELECTION ===
        folders_content = self._make_section(left_panel, "1. Folders")
        
        tk.Button(folders_content, text="📁 Input Folder", command=self._select_input_folder, bg='#D0E0FF').pack(fill='x', pady=1)
        self.input_label = tk.Label(folders_content, text="None", bg='#f0f0f0', wraplength=360, font=("Arial", 8))
        self.input_label.pack(fill='x')
        
        tk.Button(folders_content, text="💾 Output Folder", command=self._select_output_folder, bg='#FFD0D0').pack(fill='x', pady=1)
        self.output_label = tk.Label(folders_content, text="None", bg='#f0f0f0', wraplength=320, font=("Arial", 8))
        self.output_label.pack(fill='x')
        
        # === NAVIGATION ===
        tk.Label(left_panel, text="2. Navigate", font=("Arial", 10, "bold"), bg='#f0f0f0').pack(pady=1)
        
        nav_frame = tk.Frame(left_panel, bg='#f0f0f0')
        nav_frame.pack()
        tk.Button(nav_frame, text="◀ Prev", command=self._prev_image, width=10).pack(side=tk.LEFT, padx=2)
        tk.Button(nav_frame, text="Next ▶", command=self._next_image, width=10).pack(side=tk.LEFT, padx=2)
        # Quick jump to image number (1-based index)
        self.goto_entry = tk.Entry(nav_frame, width=6)
        self.goto_entry.pack(side=tk.LEFT, padx=(6,2))
        tk.Button(nav_frame, text="Go", command=self._goto_image, width=6).pack(side=tk.LEFT)
        
        self.progress_label = tk.Label(left_panel, text="No images", bg='#f0f0f0', font=("Arial", 8))
        self.progress_label.pack()
        
        # Current image indicator
        self.current_image_label = tk.Label(left_panel, text="", bg='#f0f0f0', font=("Arial", 8, "bold"), fg='#0066cc')
        self.current_image_label.pack()
        
        # Mode selector - a compact row of buttons to control which toolset is visible
        mode_bar = tk.Frame(left_panel, bg='#f0f0f0')
        mode_bar.pack(pady=4, fill='x')
        self.mode_buttons = {}
        modes = [('boundary','Border Adjust'), ('segments','Segments'), ('label','Label'), ('access','Access')]
        for m_key, m_label in modes:
            b = tk.Button(mode_bar, text=m_label, command=lambda mk=m_key: self._set_mode(mk), width=12)
            b.pack(side=tk.LEFT, padx=2, ipadx=2)
            self.mode_buttons[m_key] = b
        # Initialize mode to boundary by default
        try:
            self._set_mode('boundary')
        except Exception:
            pass

        # === BOUNDARY REVIEW ===
        tk.Label(left_panel, text="3. Boundary", font=("Arial", 10, "bold"), bg='#f0f0f0').pack(pady=1)
        
        # Group boundary controls so they can be shown/hidden as a unit when switching modes
        self.boundary_tools_frame = tk.Frame(left_panel, bg='#f0f0f0')
        self.boundary_tools_frame.pack(pady=1, fill='x')

        boundary_frame = tk.Frame(self.boundary_tools_frame, bg='#f0f0f0')
        boundary_frame.pack(pady=1)
        self.approve_btn = tk.Button(boundary_frame, text="✓ Good", command=self._approve_boundary, width=10, bg='#90EE90', state=tk.DISABLED)
        self.approve_btn.pack(side=tk.LEFT, padx=1)
        self.reject_btn = tk.Button(boundary_frame, text="✗ Bad", command=self._reject_boundary, width=10, bg='#FFB6C1', state=tk.DISABLED)
        self.reject_btn.pack(side=tk.LEFT, padx=1)
        
        # Edit/retry/regenerate buttons and status, all inside the grouped boundary frame
        self.edit_boundary_btn = tk.Button(self.boundary_tools_frame, text="✏️ Edit Boundary", command=self._unapprove_boundary, bg='#FFFFE0', state=tk.DISABLED)
        self.edit_boundary_btn.pack(fill='x', pady=1)
        
        self.retry_boundary_btn = tk.Button(self.boundary_tools_frame, text="🔄 Retry Detection", command=self._retry_boundary, bg='#FFA500', state=tk.DISABLED)
        self.retry_boundary_btn.pack(fill='x', pady=1)
        
        self.regenerate_boundary_btn = tk.Button(self.boundary_tools_frame, text="🔄 Regenerate", command=self._regenerate_boundary, bg='#FFB347', state=tk.DISABLED)
        self.regenerate_boundary_btn.pack(fill='x', pady=1)
        
        self.boundary_status = tk.Label(self.boundary_tools_frame, text="Load image", bg='#f0f0f0', wraplength=360, font=("Arial", 8), fg='black')
        self.boundary_status.pack(fill='x')

        # Mode selector moved earlier to appear above Boundary section
        # (original block removed - recreated near the Boundary header)        
        # === SEGMENTATION ===
        seg_content = self._make_section(left_panel, "4. Segments", default_open=False)
        
        seg_buttons_frame = tk.Frame(seg_content, bg='#f0f0f0')
        seg_buttons_frame.pack(pady=1)
        tk.Button(seg_buttons_frame, text="- Fewer", command=self._coarsen_segments, width=10).pack(side=tk.LEFT, padx=1)
        tk.Button(seg_buttons_frame, text="+ More", command=self._refine_segments, width=10).pack(side=tk.LEFT, padx=1)
        
        manual_frame = tk.Frame(seg_content, bg='#f0f0f0')
        manual_frame.pack(pady=1)
        # NOTE: Clicking a segment in 'Segments' mode will automatically enter Split mode
        tk.Label(manual_frame, text="Split:", bg='#f0f0f0', font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=(0,2))
        self.manual_btn = tk.Button(manual_frame, text="Split Mode", command=self._toggle_manual_mode, width=18, bg='#FFD700', relief=tk.RAISED, font=("Arial", 8, "bold"))
        self.manual_btn.pack(side=tk.LEFT, padx=1)
        self.finalize_btn = tk.Button(manual_frame, text="Apply", command=self._finalize_manual_split, width=8, bg='#ADD8E6', state=tk.DISABLED)
        self.finalize_btn.pack(side=tk.LEFT, padx=1)
        
        undo_frame = tk.Frame(seg_content, bg='#f0f0f0')
        undo_frame.pack(pady=1)
        self.undo_split_btn = tk.Button(undo_frame, text="↶ Undo Split", command=self._undo_split, width=20, bg='#FFE4B5', state=tk.DISABLED)
        self.undo_split_btn.pack()
        
        self.manual_status = tk.Label(seg_content, text="", bg='#f0f0f0', fg='#0066cc', font=("Arial", 8, "bold"))
        self.manual_status.pack()
        
        self.seg_status = tk.Label(seg_content, text="Not generated", bg='#f0f0f0', font=("Arial", 8))
        self.seg_status.pack()

        # Smoothing control for segment simplification
        smoothing_frame = tk.Frame(seg_content, bg='#f0f0f0')
        smoothing_frame.pack(pady=2)
        tk.Label(smoothing_frame, text="Smoothing:", bg='#f0f0f0').pack(side=tk.LEFT)
        self.segment_smoothing_var = tk.StringVar(value='med')
        smoothing_options = [('Off','off'), ('Low','low'), ('Med','med'), ('High','high')]
        for text, val in smoothing_options:
            tk.Radiobutton(smoothing_frame, text=text, variable=self.segment_smoothing_var, value=val, bg='#f0f0f0', command=self._on_smoothing_change).pack(side=tk.LEFT, padx=4)

        # Min segment size control
        size_frame = tk.Frame(seg_content, bg='#f0f0f0')
        size_frame.pack(pady=2)
        tk.Label(size_frame, text="Min segment px:", bg='#f0f0f0').pack(side=tk.LEFT)
        self.min_segment_entry = tk.Entry(size_frame, width=6)
        self.min_segment_entry.insert(0, "1000")
        self.min_segment_entry.pack(side=tk.LEFT, padx=4)
        tk.Button(size_frame, text="Apply", command=self._apply_min_segment_px).pack(side=tk.LEFT)
        self.min_segment_px = 1000

        # Shadow-robust toggle (reduces sensitivity to shadows by using chromaticity + gradient)
        shadow_frame = tk.Frame(seg_content, bg='#f0f0f0')
        shadow_frame.pack(pady=2)
        self.shadow_robust_var = tk.BooleanVar(value=False)
        tk.Checkbutton(shadow_frame, text="Shadow-robust segmentation", variable=self.shadow_robust_var, bg='#f0f0f0', command=self._on_shadow_robust_toggle).pack(side=tk.LEFT)

        # Pre-smooth, fast-preview and auto-merge options
        opts_frame = tk.Frame(seg_content, bg='#f0f0f0')
        opts_frame.pack(pady=2)
        self.pre_smooth_var = tk.BooleanVar(value=True)
        tk.Checkbutton(opts_frame, text="Pre-smooth (bilateral)", variable=self.pre_smooth_var, bg='#f0f0f0').pack(side=tk.LEFT, padx=4)
        self.fast_preview_var = tk.BooleanVar(value=True)
        tk.Checkbutton(opts_frame, text="Fast preview (lower quality, faster)", variable=self.fast_preview_var, bg='#f0f0f0').pack(side=tk.LEFT, padx=4)
        # Keep auto-merge off by default to avoid overly-aggressive merging during high-target tests
        self.auto_merge_var = tk.BooleanVar(value=False)
        tk.Checkbutton(opts_frame, text="Auto-merge small regions", variable=self.auto_merge_var, bg='#f0f0f0').pack(side=tk.LEFT, padx=4)

        # Manual start: Segment Myself button
        manual_start_frame = tk.Frame(seg_content, bg='#f0f0f0')
        manual_start_frame.pack(pady=4)
        self.segment_myself_btn = tk.Button(manual_start_frame, text="Segment Myself", command=self._segment_myself, bg='#EEE8AA')
        self.segment_myself_btn.pack(pady=1)

        # User hint explaining unified Segments mode (label, split, refine)
        self.segments_hint_label = tk.Label(seg_content, text=(
            "Hint: In Segments mode you can: left-click a segment to label it; "
            "right-click to remove a label; click 'Split Mode' to draw split lines; "
            "use + / - to change segmentation granularity. Use smoothing to reduce squiggles."),
            bg='#f0f0f0', fg='#666666', wraplength=320, font=("Arial", 8))
        self.segments_hint_label.pack(pady=(4,2))
        # Initialize smoothing control state
        try:
            self.segment_smoothing_level = self.segment_smoothing_var.get()
        except Exception:
            self.segment_smoothing_level = 'med'

        # Access & Roads section moved below to preserve numeric ordering
        
        # === CLASS SELECTION ===
        class_content = self._make_section(left_panel, "5. Class (click to label)", default_open=False)
        
        # Create 2-column grid for class labels
        class_frame = tk.Frame(class_content, bg='#f0f0f0')
        class_frame.pack(pady=1)
        
        self.class_var = tk.StringVar(value=CLASSES[0])
        for idx, cls in enumerate(CLASSES):
            color = CLASS_COLORS[cls]
            rgb_color = f'#{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}'
            # Use friendly display name if provided
            display_text = DISPLAY_NAMES.get(cls, cls.replace("_", " ").title())
            rb = tk.Radiobutton(class_frame, text=display_text, 
                               variable=self.class_var, value=cls,
                               command=self._on_class_selected,
                               bg=rgb_color, activebackground=rgb_color,
                               font=("Arial", 8, "bold"),
                               width=18, anchor="w")
            # Grid: row = idx // 2, column = idx % 2
            rb.grid(row=idx // 2, column=idx % 2, sticky="w", padx=1, pady=0)
        
        # === ACCESS & ROADS ===
        access_content = self._make_section(left_panel, "6. Access & Roads", default_open=False)
        access_frame = tk.Frame(access_content, bg='#f0f0f0')
        access_frame.pack(pady=1)
        self.boundary_access_btn = tk.Button(access_frame, text="Boundary Access Mode", command=self._toggle_boundary_access_mode, width=22, bg='#FFE4B5')
        self.boundary_access_btn.pack(pady=1)
        # Buffer controls (include outside buffer in segmentation and labeling)
        buffer_frame = tk.Frame(access_content, bg='#f0f0f0')
        buffer_frame.pack(pady=1)
        tk.Label(buffer_frame, text="Buffer (px or pct):", bg='#f0f0f0').pack(side=tk.LEFT)
        self.buffer_mode_var = tk.StringVar(value='px')
        tk.Radiobutton(buffer_frame, text="px", variable=self.buffer_mode_var, value='px', command=self._on_buffer_mode_change, bg='#f0f0f0').pack(side=tk.LEFT)
        tk.Radiobutton(buffer_frame, text="pct", variable=self.buffer_mode_var, value='pct', command=self._on_buffer_mode_change, bg='#f0f0f0').pack(side=tk.LEFT)
        self.buffer_entry = tk.Entry(buffer_frame, width=6)
        self.buffer_entry.insert(0, str(self.road_buffer_px))
        self.buffer_entry.pack(side=tk.LEFT, padx=4)
        self.apply_buffer_btn = tk.Button(buffer_frame, text="Apply", command=self._apply_buffer_value)
        self.apply_buffer_btn.pack(side=tk.LEFT, padx=4)
        tk.Label(buffer_frame, text="(Buffer outside boundary will be included in segmentation)", bg='#f0f0f0', fg='#666666').pack(side=tk.LEFT, padx=6)
        # Legend
        legend_frame = tk.Frame(access_content, bg='#f0f0f0')
        legend_frame.pack(pady=1)
        tk.Label(legend_frame, text="Legend:", bg='#f0f0f0', font=("Arial", 8, "bold")).pack(anchor='w')
        tk.Label(legend_frame, text="Boundary no access", bg='#FFD700', width=20, wraplength=300).pack(pady=1)
        tk.Label(legend_frame, text="Boundary access allowed", bg='#00FF00', width=20, wraplength=300).pack(pady=1)
        tk.Label(legend_frame, text="Note: Buffer area will be included in segmentation", bg='#E8E8E8', width=30, wraplength=300).pack(pady=1)

        # === PROGRESS ===
        self.labeling_progress = tk.Label(left_panel, text="Labeled: 0/0", bg='#f0f0f0', wraplength=280, font=("Arial", 8))
        # Progress label is packed into submit_frame so it is shown as part of Label mode.
        # --- smoothing helpers exposed for scripting/tests
        self.SMOOTHING_EPS = {
            'off': 0.0,
            'low': 0.002,
            'med': 0.01,
            'high': 0.02
        }
        
        # === PROPERTY NAME ===
        self.property_label = tk.Label(left_panel, text="", bg='#f0f0f0', wraplength=360, font=("Arial", 8, "bold"), fg='#0066cc')
        self.property_label.pack()
        
        # === SUBMIT (fixed footer) ===
        self.submit_frame = tk.Frame(self.footer_frame, bg='#f0f0f0')
        self.submit_frame.pack(pady=2, fill='x')
        tk.Label(self.submit_frame, text="6. Submit", font=("Arial", 10, "bold"), bg='#f0f0f0').pack(pady=1)
        self.submit_btn = tk.Button(self.submit_frame, text="✓ Submit Annotation", command=self._submit_annotation, 
                 bg='#ADD8E6', font=("Arial", 9, "bold"))
        self.submit_btn.pack(fill='x', pady=4)
        # Save Draft button always visible in footer
        self.save_draft_btn = tk.Button(self.submit_frame, text="Save Draft", command=self._save_draft, bg='#E0E0E0')
        self.save_draft_btn.pack(fill='x', pady=(0,4))
        # Progress label shown in footer
        self.labeling_progress.pack(pady=1)
        
# (Access & Roads section moved to appear earlier as item 6 - see insertion near property label)
        
        # Right panel - Image display
        right_panel = tk.Frame(self.root)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.fig = Figure(figsize=(14, 10))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, right_panel)
        self.canvas.mpl_connect('button_release_event', self._on_release)
        self.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.canvas.mpl_connect('button_press_event', self._on_click)
        self.canvas.mpl_connect('key_press_event', self._on_key_press)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.ax.set_title("Choose input folder to begin")
        self.ax.axis('off')
        self.canvas.draw()

        # === GLOBAL SAVE BAR ===
        # Add a persistent, always-visible Save button across the bottom of the application
        # so users can always commit segmentation/labels even if left panel content is clipped.
        try:
            self.global_footer = tk.Frame(self.root, bg='#e8e8e8')
            # Pack at the root level bottom so it is always visible
            self.global_footer.pack(side=tk.BOTTOM, fill='x')
            self.save_btn = tk.Button(self.global_footer, text="✓ Save", command=self._submit_annotation, bg='#4CAF50', fg='white', font=("Arial", 10, "bold"))
            self.save_btn.pack(side=tk.RIGHT, padx=8, pady=6)
            # Hide the in-footer Save Draft button (keeps method intact for tests) to avoid duplicates
            try:
                self.save_draft_btn.pack_forget()
            except Exception:
                pass
        except Exception:
            pass
        # Floating save button in the top-right corner for guaranteed visibility
        try:
            self.floating_save_btn = tk.Button(self.root, text="💾 Save", command=self._save_draft, bg='#E0E0E0')
            # place relative to root so it's visible regardless of panel clipping
            self.floating_save_btn.place(relx=0.98, rely=0.02, anchor='ne')
            self.floating_save_visible = True
        except Exception:
            self.floating_save_btn = None
            self.floating_save_visible = False
        # Add a small toggle to the top toolbar to hide/show the floating button
        try:
            def _toggle_floating_save():
                try:
                    if getattr(self, 'floating_save_visible', False):
                        if self.floating_save_btn is not None:
                            self.floating_save_btn.place_forget()
                        self.floating_save_visible = False
                        self.quick_floating_toggle.config(text='☆')
                    else:
                        if self.floating_save_btn is not None:
                            self.floating_save_btn.place(relx=0.98, rely=0.02, anchor='ne')
                        self.floating_save_visible = True
                        self.quick_floating_toggle.config(text='★')
                except Exception:
                    pass
            self.quick_floating_toggle = tk.Button(top_toolbar, text='★', command=_toggle_floating_save, bg='#F0F0F0', width=3)
            self.quick_floating_toggle.pack(side=tk.RIGHT, padx=(0,2), pady=2)
        except Exception:
            pass
        
    def _select_input_folder(self):
        """Select input folder."""
        initialdir = self.last_input_folder.parent if self.last_input_folder else None
        folder = filedialog.askdirectory(title="Select Input Folder", initialdir=initialdir)
        if folder:
            self.images_folder = Path(folder)
            self.last_input_folder = Path(folder)
            self._save_folder_preferences()
            self.image_files = sorted(list(self.images_folder.glob("*.png")) + 
                                     list(self.images_folder.glob("*.jpg")))
            
            if not self.image_files:
                messagebox.showerror("Error", "No images found")
                return
            
            self.input_label.config(text=f"{len(self.image_files)} images found")
            
            # Enable boundary buttons if output folder also selected
            if self.output_folder is not None:
                self.approve_btn.config(state=tk.NORMAL)
                self.reject_btn.config(state=tk.NORMAL)
                self.retry_boundary_btn.config(state=tk.DISABLED)
            
            # Load first image
            try:
                self._load_image(0)
            except Exception as e:
                messagebox.showerror("Error loading image", str(e))
                import traceback
                traceback.print_exc()
            
    def _select_output_folder(self):
        """Select output folder."""
        initialdir = self.last_output_folder.parent if self.last_output_folder else None
        folder = filedialog.askdirectory(title="Select Output Folder", initialdir=initialdir)
        if folder:
            self.output_folder = Path(folder)
            self.last_output_folder = Path(folder)
            self.output_folder.mkdir(parents=True, exist_ok=True)
            self._save_folder_preferences()
            self.output_label.config(text=f"Output: {self.output_folder.name}")
            
            # Enable boundary buttons now that both folders are selected
            if self.images_folder is not None:
                self.approve_btn.config(state=tk.NORMAL)
                self.reject_btn.config(state=tk.NORMAL)
            
    def _prev_image(self):
        """Load previous image."""
        if not self.image_files:
            return
        if self.current_idx > 0:
            if self._check_unsaved_changes():
                self._load_image(self.current_idx - 1)
        else:
            print(f"Already at first image (index {self.current_idx})")
            
    def _next_image(self):
        """Load next image."""
        if not self.image_files:
            return
        if self.current_idx < len(self.image_files) - 1:
            if self._check_unsaved_changes():
                self._load_image(self.current_idx + 1)
        else:
            print(f"Already at last image (index {self.current_idx})")

    def _goto_image(self):
        """Go to a specific image number (1-based)."""
        if not self.image_files:
            return
        val = None
        try:
            val = self.goto_entry.get()
            idx = int(val) - 1
            if idx < 0 or idx >= len(self.image_files):
                messagebox.showwarning("Invalid index", f"Enter an image number between 1 and {len(self.image_files)}")
                return
            if self._check_unsaved_changes():
                self._load_image(idx)
        except Exception:
            messagebox.showwarning("Invalid value", "Enter a valid integer image number")
            return
            
    def _load_image(self, idx: int):
        """Load image and detect boundary."""
        try:
            self.current_idx = idx
            image_path = self.image_files[idx]
            
            # Update navigation display
            self.current_image_label.config(text=f"Image {idx + 1} of {len(self.image_files)}")
            print(f"\n=== Loading image {idx + 1}/{len(self.image_files)}: {image_path.name} ===")
            
            # Save current position to config
            self._save_folder_preferences()
            
            # Reset state
            self.has_unsaved_changes = False
            self.boundary_approved = False
            self.segments = None
            self.segment_labels = {}
            self.road_access_lines = []
            self.enhanced_image = None  # Clear enhanced image cache
            
            # Reset boundary status color to default
            self.boundary_status.config(fg='black')
        except Exception as e:
            print(f"ERROR in _load_image setup: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # Reset progress display
        self.labeling_progress.config(text="Labeled: 0% of pixels")
        
        # Update property name display from filename
        property_name = image_path.stem.replace('_', ' ')
        self.property_label.config(text=property_name)
        
        # Check if this image already has saved labels
        existing_labels = None
        is_bad_boundary = False
        if self.output_folder is not None:
            label_file = self.output_folder / f"{image_path.stem}_labels.json"
            print(f"Checking for labels at: {label_file}")
            if label_file.exists():
                try:
                    with open(label_file, 'r') as f:
                        existing_labels = json.load(f)
                    print(f"Found existing labels")
                    
                    # Check if boundary was flagged as bad
                    if existing_labels.get("boundary_status") == "rejected":
                        is_bad_boundary = True
                        self.boundary_status.config(text="🚫 Bad boundary flagged", fg='red')
                        print("Image has bad boundary flag")
                    else:
                        self.boundary_status.config(text="⚠ Previously labeled - reloading...")
                except Exception as e:
                    print(f"Error loading existing labels: {e}")
        
        # Load image at original resolution
        # Automatically use raw_images for boundary detection if adjusted_images was selected
        try:
            self.current_image = cv2.imread(str(image_path))
            if self.current_image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            # Check if image has red boundary - if not, try to find raw version
            b, g, r = cv2.split(self.current_image)
            red_pixel_count = np.sum((r > 200) & (g < 100) & (b < 100))
            
            if red_pixel_count == 0:
                # Try to find raw image version
                raw_path = None
                if "adjusted_images" in str(image_path):
                    raw_path = Path(str(image_path).replace("adjusted_images", "raw_images"))
                elif "labeled_images" in str(image_path):
                    raw_path = Path(str(image_path).replace("labeled_images", "raw_images"))
                
                if raw_path and raw_path.exists():
                    print(f"⚠️  No red boundary in {image_path.parent.name}/{image_path.name}")
                    print(f"✓  Found raw version at {raw_path.parent.name}/{raw_path.name}")
                    raw_image = cv2.imread(str(raw_path))
                    if raw_image is not None:
                        # Check raw has red pixels
                        b_raw, g_raw, r_raw = cv2.split(raw_image)
                        raw_red_count = np.sum((r_raw > 200) & (g_raw < 100) & (b_raw < 100))
                        if raw_red_count > 0:
                            print(f"✓  Using raw image for boundary detection ({raw_red_count} red pixels)")
                            self.current_image = raw_image
                        else:
                            print(f"⚠️  Raw image also has no red boundary!")
                else:
                    print(f"⚠️  No red boundary found and no raw_images version available")
                    print(f"⚠️  Current path: {image_path}")
            
            rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            print(f"Loaded image: {self.current_image.shape}")
        except Exception as e:
            print(f"ERROR loading image file: {e}")
            import traceback
            traceback.print_exc()
            self.ax.clear()
            self.ax.set_title(f"Error loading image: {image_path.name}")
            self.ax.axis('off')
            self.canvas.draw()
            return
        
        # Scale to max 1500px (increased from 1000px for better quality)
        h, w = rgb.shape[:2]
        scale = min(1500 / w, 1500 / h, 1.0)
        if scale < 1.0:
            new_w, new_h = int(w * scale), int(h * scale)
            rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            self.current_image = cv2.resize(self.current_image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Detect boundary
        print("Starting boundary detection...")
        boundary_result = estimate_boundary_from_overlay(self.current_image)
        print(f"Boundary detection result: polygon={boundary_result.polygon_px is not None}, confidence={boundary_result.confidence}, warnings={boundary_result.warnings}")
        
        # Save debug image for boundary detection
        if self.output_folder is not None:
            from app.core.boundary import save_boundary_debug_image
            debug_folder = self.output_folder.parent / "debug"
            debug_path = debug_folder / f"{image_path.stem}_boundary_debug.png"
            save_boundary_debug_image(self.current_image, boundary_result.polygon_px, str(debug_path))
        
        # Inpaint out the red boundary for clean display
        hsv = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        
        # Inpaint to remove red pixels
        self.clean_image = cv2.inpaint(self.current_image, red_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        rgb = cv2.cvtColor(self.clean_image, cv2.COLOR_BGR2RGB)
        
        # Pre-generate enhanced image for boundary adjustment (item 8 fix)
        lab = cv2.cvtColor(self.clean_image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        self.enhanced_image = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
        # Initialize smoothing level (controls polygon simplification)
        if not hasattr(self, 'segment_smoothing_level'):
            self.segment_smoothing_level = 'med'  # default: med
        
        # Only display clean image initially if we don't have existing labels
        # (if we do have labels, _update_display() will handle the display)
        if existing_labels is None:
            try:
                self.ax.clear()
                self.ax.imshow(self.enhanced_image)  # Use enhanced image from the start
                self.ax.axis('off')
                self.canvas.draw()
                print("Displayed image successfully")
            except Exception as e:
                print(f"ERROR displaying image: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("Existing labels found - will display after loading segments")
        
        # Add boundary if detected (skip if bad boundary flagged)
        if is_bad_boundary:
            # Show the image but don't process boundary
            self.ax.clear()
            self.ax.imshow(self.enhanced_image)
            self.ax.set_title("Bad boundary - manually flagged by user")
            self.ax.axis('off')
            self.canvas.draw()
            print("Skipping boundary processing - bad boundary flagged")
            
            # Enable retry button, disable other boundary controls
            self.approve_btn.config(state=tk.DISABLED)
            self.reject_btn.config(state=tk.DISABLED)
            self.edit_boundary_btn.config(state=tk.DISABLED)
            self.retry_boundary_btn.config(state=tk.NORMAL)
            self.regenerate_boundary_btn.config(state=tk.DISABLED)
            self.boundary_status.config(text="Bad boundary flagged - click Retry to re-detect", fg='red')
            return  # Don't process boundary further
        elif boundary_result.polygon_px is not None or existing_labels is not None:
            # If we have existing labels, use the SAVED boundary, not the newly detected one
            if existing_labels is not None and "boundary_polygon_px" in existing_labels and existing_labels["boundary_polygon_px"] is not None:
                # Use saved boundary vertices
                self.original_boundary = np.array(existing_labels["boundary_polygon_px"])
                print(f"Using saved boundary with {len(self.original_boundary)} vertices")
            elif boundary_result.polygon_px is not None:
                # Use newly detected boundary
                self.original_boundary = np.array(boundary_result.polygon_px)
                print(f"Using newly detected boundary with {len(self.original_boundary)} vertices")
            else:
                # No boundary available
                self.current_boundary = None
                self.original_boundary = None
                self.boundary_status.config(text="⚠ No boundary detected")
                self.approve_btn.config(state=tk.DISABLED)
                self.reject_btn.config(state=tk.DISABLED)
                self.retry_boundary_btn.config(state=tk.DISABLED)
                return
            
            self.current_boundary = self.original_boundary.copy()
            self.boundary_dx = 0.0
            self.boundary_dy = 0.0
            self.boundary_theta = 0.0

            # Default boundary access: new images should start as 'no_access' across entire boundary
            if existing_labels is None and self.current_boundary is not None and not self.boundary_access_segments:
                self.boundary_access_segments = [{
                    'start_frac': 0.0,
                    'end_frac': 1.0,
                    'label': 'no_access'
                }]

            # If we have existing labels, auto-approve boundary and reload saved segments
            if existing_labels is not None and "boundary_polygon_px" in existing_labels:
                self.boundary_status.config(text=f"Previously labeled - reloading")
                self.boundary_approved = True

                # Enable edit boundary button for previously labeled images
                self.edit_boundary_btn.config(state=tk.NORMAL)
                self.approve_btn.config(state=tk.DISABLED)
                self.reject_btn.config(state=tk.DISABLED)
                self.retry_boundary_btn.config(state=tk.DISABLED)
                self.regenerate_boundary_btn.config(state=tk.NORMAL)  # Enable regenerate

                # Restore boundary transform BEFORE drawing
                boundary_transform = existing_labels.get("boundary_transform", {})
                self.boundary_dx = boundary_transform.get("dx", 0.0)
                self.boundary_dy = boundary_transform.get("dy", 0.0)
                self.boundary_theta = boundary_transform.get("theta_deg", 0.0)

                # Apply transform to boundary
                if self.boundary_dx != 0.0 or self.boundary_dy != 0.0 or self.boundary_theta != 0.0:
                    self._apply_boundary_transform(self.boundary_dx, self.boundary_dy, self.boundary_theta)
                    print(f"Applied boundary transform: dx={self.boundary_dx}, dy={self.boundary_dy}, theta={self.boundary_theta}")

                # Try to load saved segments array
                segments_file = self.output_folder / f"{image_path.stem}_segments.npy"
                if segments_file.exists():
                    try:
                        self.segments = np.load(str(segments_file))
                        self.n_segments = self.segments.max()
                        self.target_segments = self.n_segments
                        print(f"Loaded saved segments: {self.n_segments} segments")
                    except Exception as e:
                        print(f"Error loading segments: {e}")
                        # Fallback to regenerating
                        self.target_segments = existing_labels.get("slic_params", {}).get("n_segments", 50)
                        self._generate_segments()
                else:
                    # Fallback to regenerating
                    self.target_segments = existing_labels.get("slic_params", {}).get("n_segments", 50)
                    self._generate_segments()
                
                # Restore segment labels (excluding auto-filled ones)
                for seg_id_str, seg_data in existing_labels.get("segments", {}).items():
                    seg_id = int(seg_id_str)
                    class_name = seg_data.get("class")
                    labeled_by = seg_data.get("labeled_by", "user")  # Default to user if missing
                    # Only restore user-labeled segments, not auto-filled ones
                    if labeled_by == "user" and seg_id <= self.n_segments:
                        # Normalize legacy class names to the updated set
                        if class_name == 'parking_surface':
                            class_name = 'parking_stalls'
                        if class_name not in CLASSES:
                            print(f"Warning: Unknown class in saved labels: {class_name}. Mapping to 'parking_stalls'.")
                            class_name = 'parking_stalls'
                        self.segment_labels[seg_id] = class_name

                # Restore boundary access segments if present
                for seg in existing_labels.get("boundary_access", {}).get("segments", []):
                    # Accept either new fractional format or legacy pixel coords
                    if 'start_frac' in seg and 'end_frac' in seg:
                        self.boundary_access_segments.append(seg)
                    elif 'start_px' in seg and 'end_px' in seg and self.current_boundary is not None:
                        start_px = seg['start_px']
                        end_px = seg['end_px']
                        _, s_frac = self._closest_point_on_boundary(start_px[0], start_px[1])
                        _, e_frac = self._closest_point_on_boundary(end_px[0], end_px[1])
                        if s_frac is not None and e_frac is not None:
                            seg_new = {'start_frac': float(s_frac), 'end_frac': float(e_frac), 'label': seg.get('label', 'access_allowed')}
                            self.boundary_access_segments.append(seg_new)


                
                # Mark as saved (no unsaved changes since we just loaded)
                self.has_unsaved_changes = False
                self.submit_btn.config(text="✓ Saved!", bg='#90EE90')
                
                print("Calling _update_display() for previously labeled image")
                self._update_display()
                self._update_progress()
                print("Display updated for previously labeled image")
            else:
                # New image - draw the detected boundary for user to review
                self._draw_editable_boundary()
                self.boundary_status.config(text=f"Detected (conf: {boundary_result.confidence:.2f})\nDrag to adjust")
                self.ax.set_title("Adjust boundary (arrows=move, </>=rotate) and press Enter to proceed")
                
                # Enable boundary approval buttons
                self.approve_btn.config(state=tk.NORMAL)
                self.reject_btn.config(state=tk.NORMAL)
                self.retry_boundary_btn.config(state=tk.DISABLED)
                self.regenerate_boundary_btn.config(state=tk.DISABLED)
        else:
            self.current_boundary = None
            self.original_boundary = None
            self.boundary_status.config(text="⚠ No boundary detected")
            self.ax.set_title(f"Image {idx + 1}/{len(self.image_files)}: {image_path.name}")
            
            # Disable boundary buttons when no boundary detected
            self.approve_btn.config(state=tk.DISABLED)
            self.reject_btn.config(state=tk.DISABLED)
            self.retry_boundary_btn.config(state=tk.DISABLED)
            self.regenerate_boundary_btn.config(state=tk.DISABLED)
        
        if existing_labels is None and boundary_result.polygon_px is None:
            self.ax.set_title(f"Image {idx + 1}/{len(self.image_files)}: {image_path.name}")
        self.ax.axis('off')
        self.canvas.draw()
        # Ensure submit frame remains packed in the footer so Save is always visible
        try:
            self.submit_frame.pack_forget()
            self.submit_frame.pack(in_=self.footer_frame, pady=1, fill='x')
        except Exception:
            pass
        # Ensure canvas has focus for keyboard events
        self.canvas.get_tk_widget().focus_set()
        
        self.progress_label.config(text=f"Image {idx + 1} of {len(self.image_files)}")
    
    def _apply_boundary_transform(self, dx: float, dy: float, theta_deg: float):
        """Apply transformation to original boundary."""
        if self.original_boundary is None:
            return
        
        # Get image dimensions
        h, w = self.clean_image.shape[:2] if self.clean_image is not None else (1000, 1000)
        
        # Compute centroid of original boundary
        centroid = self.original_boundary.mean(axis=0)
        
        # Translate to origin
        boundary = self.original_boundary - centroid
        
        # Rotate
        theta_rad = np.deg2rad(theta_deg)
        cos_t = np.cos(theta_rad)
        sin_t = np.sin(theta_rad)
        rotation_matrix = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        boundary = boundary @ rotation_matrix.T
        
        # Translate back and apply offset
        boundary = boundary + centroid + np.array([dx, dy])
        
        # Check if any point goes outside image bounds
        min_x, min_y = boundary.min(axis=0)
        max_x, max_y = boundary.max(axis=0)
        
        # Constrain to keep entire boundary within image
        if min_x < 0:
            dx -= min_x
        if min_y < 0:
            dy -= min_y
        if max_x >= w:
            dx -= (max_x - w + 1)
        if max_y >= h:
            dy -= (max_y - h + 1)
        
        # Recompute boundary with constrained offset
        boundary = self.original_boundary - centroid
        boundary = boundary @ rotation_matrix.T
        boundary = boundary + centroid + np.array([dx, dy])
        
        self.current_boundary = boundary
        self.boundary_dx = dx
        self.boundary_dy = dy
        self.boundary_theta = theta_deg
    
    def _reset_boundary_transform(self):
        """Reset boundary to original detected position."""
        if self.original_boundary is None:
            return
        
        self.current_boundary = self.original_boundary.copy()
        self.boundary_dx = 0.0
        self.boundary_dy = 0.0
        self.boundary_theta = 0.0
        
        # Redraw
        rgb = cv2.cvtColor(self.clean_image, cv2.COLOR_BGR2RGB)
        self.ax.clear()
        self.ax.imshow(rgb)
        self._draw_editable_boundary()
        self.ax.axis('off')
        self.canvas.draw()
        
    def _approve_boundary(self):
        """Approve boundary and generate segments."""
        if self.current_boundary is None:
            messagebox.showwarning("Warning", "No boundary detected")
            return
        
        print(f"DEBUG _approve_boundary: current_boundary has {len(self.current_boundary)} vertices")
        print(f"DEBUG _approve_boundary: boundary transform - dx={self.boundary_dx}, dy={self.boundary_dy}, theta={self.boundary_theta}")
        
        self.boundary_approved = True
        self.boundary_status.config(text="✓ Approved - generating...")
        self.edit_boundary_btn.config(state=tk.NORMAL)  # Enable edit button
        self.retry_boundary_btn.config(state=tk.DISABLED)  # Disable retry button
        self.regenerate_boundary_btn.config(state=tk.DISABLED)  # Disable regenerate button
        self._generate_segments()
        # Default to Segments mode after approval so user can label/split seamlessly
        try:
            self._set_mode('segments')
        except Exception:
            pass

        
    def _reject_boundary(self):
        """Reject boundary."""
        image_name = self.image_files[self.current_idx].stem
        output_file = self.output_folder / f"{image_name}_labels.json"
        
        data = {
            "image_id": image_name,
            "timestamp": datetime.now().isoformat(),
            "boundary_status": "rejected",
            "reason": "User marked boundary as bad"
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        messagebox.showinfo("Saved", f"Bad boundary marker saved")
        
        if self.current_idx < len(self.image_files) - 1:
            self._next_image()
    
    def _retry_boundary(self):
        """Retry boundary detection after flagging as bad."""
        # Delete the bad boundary marker file
        if self.output_folder is None:
            messagebox.showwarning("Warning", "No output folder selected")
            return
        
        image_name = self.image_files[self.current_idx].stem
        output_file = self.output_folder / f"{image_name}_labels.json"
        
        if output_file.exists():
            try:
                output_file.unlink()
                print(f"Deleted bad boundary marker: {output_file}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to delete marker file: {e}")
                return
        
        # Reload the image (will re-run boundary detection)
        self._load_image(self.current_idx)
        messagebox.showinfo("Retry", "Boundary detection rerun. Adjust if needed and approve.")
    
    def _regenerate_boundary(self):
        """Regenerate boundary using current algorithm, discarding saved boundary."""
        if self.output_folder is None:
            messagebox.showwarning("Warning", "No output folder selected")
            return
        
        # Warn user about losing labels
        response = messagebox.askyesno(
            "Warning: Progress Will Be Lost",
            "This will delete the saved boundary and labels,\n"
            "and re-run boundary detection with the current algorithm.\n\n"
            "All labeling progress for this image will be lost.\n\n"
            "Continue?"
        )
        if not response:
            return
        
        # Delete saved files
        image_name = self.image_files[self.current_idx].stem
        files_to_delete = [
            self.output_folder / f"{image_name}_labels.json",
            self.output_folder / f"{image_name}_segments.npy",
            self.output_folder / f"{image_name}_segmented.png",
            self.output_folder / f"{image_name}_mask.png"
        ]
        
        for file_path in files_to_delete:
            if file_path.exists():
                try:
                    file_path.unlink()
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
        
        # Reload image (will run fresh boundary detection)
        self._load_image(self.current_idx)
        messagebox.showinfo("Regenerated", "Boundary regenerated with current algorithm. Adjust if needed and approve.")
    
    def _unapprove_boundary(self):
        """Return to boundary adjustment mode after approval."""
        if not self.boundary_approved:
            return
        
        # Warn user that labels will be lost
        if self.segment_labels:
            response = messagebox.askyesno(
                "Warning: Labels Will Be Lost",
                f"You have {len(self.segment_labels)} labeled segments.\n\n"
                "Adjusting the boundary will regenerate segments,\n"
                "and your current labels cannot be transferred to\n"
                "the new segments (they cover different areas).\n\n"
                "Continue and lose labels?"
            )
            if not response:
                return  # User cancelled
        
        # Clear segments and labels
        self.boundary_approved = False
        self.segments = None
        self.segment_labels = {}
        self.has_unsaved_changes = False
        self.preserved_labels = None  # Clear any preserved labels
        
        # Make sure we're not in manual/split mode
        if self.manual_mode:
            self.manual_mode = False
            self._clear_manual_line()
        
        # Reset mode button to show Split Mode (click a segment to start splitting)
        self.manual_btn.config(text="Split Mode", bg='#FFD700', relief=tk.RAISED, font=("Arial", 8, "bold"))
        
        # Reset UI
        self.edit_boundary_btn.config(state=tk.DISABLED)
        self.retry_boundary_btn.config(state=tk.DISABLED)
        self.regenerate_boundary_btn.config(state=tk.DISABLED)
        self.boundary_status.config(text="Adjust boundary (arrows=move, </>=rotate)")
        self.submit_btn.config(text="✓ Submit Annotation", bg='#ADD8E6')
        self.labeling_progress.config(text="Labeled: 0% of pixels")
        self.seg_status.config(text="Not generated")
        
        # Redraw with boundary adjustment view using enhanced image or fallback
        self.ax.clear()
        if self.enhanced_image is not None:
            try:
                self.ax.imshow(self.enhanced_image)
            except Exception:
                pass
        elif self.clean_image is not None:
            try:
                self.ax.imshow(cv2.cvtColor(self.clean_image, cv2.COLOR_BGR2RGB))
            except Exception:
                pass
        # Draw boundary overlays even if an image is not available
        self._draw_editable_boundary()
        self.ax.set_title("Adjust boundary (use arrow keys to move and </> to rotate)")
        self.ax.axis('off')
        try:
            self.canvas.draw()
        except Exception:
            pass
        try:
            self.canvas.get_tk_widget().focus_set()
        except Exception:
            pass
            
    def _generate_segments(self):
        """Generate segments that follow edges with straight boundaries."""
        if self.current_boundary is None:
            return
        
        print(f"DEBUG _generate_segments: Using boundary with {len(self.current_boundary)} vertices")
        print(f"DEBUG _generate_segments: Boundary corners: {self.current_boundary[:2]}")
        
        # Use clean image for segmentation
        rgb = cv2.cvtColor(self.clean_image, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        
        # Fast preview mode uses a smaller scale for quick interactive feedback
        if getattr(self, 'fast_preview_var', None) and self.fast_preview_var.get():
            scale_factor = 0.25  # very small for fast previews
            max_attempts_local = 2
        else:
            scale_factor = 0.5
            max_attempts_local = 6
        small_h, small_w = int(h * scale_factor), int(w * scale_factor)
        rgb_small = cv2.resize(rgb, (small_w, small_h), interpolation=cv2.INTER_AREA)
        
        # Enhance contrast using CLAHE for better edge detection
        bgr_small = cv2.cvtColor(rgb_small, cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2LAB)
        l, a, b_channel = cv2.split(lab)

        # Apply CLAHE only moderately to reduce shadow impact
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)

        # Normalize color channels to reduce shadow impact and emphasize material differences
        # Boost color channels (a, b) relative to lightness to prioritize material over shadow
        l_normalized = l_enhanced.astype(float) / 255.0
        a_normalized = (a.astype(float) - 128) / 128.0  # Center and normalize
        b_normalized = (b_channel.astype(float) - 128) / 128.0

        # Default feature image (legacy): emphasize color over brightness
        feature_img_default = np.stack([
            l_normalized * 0.3,  # Reduce lightness weight (shadows)
            a_normalized * 1.2,  # Boost green-red
            b_normalized * 1.2   # Boost blue-yellow
        ], axis=-1)

        # Shadow-robust feature composition: chromaticity + gradient magnitude + color axes
        if getattr(self, 'shadow_robust_var', None) and self.shadow_robust_var.get():
            # Compute chromaticity channels (r/(r+g+b), g/(r+g+b)) to reduce brightness effect
            arr = rgb_small.astype(np.float32)
            denom = arr.sum(axis=2, keepdims=True) + 1e-6
            chroma_r = (arr[:, :, 0:1] / denom).squeeze()
            chroma_g = (arr[:, :, 1:2] / denom).squeeze()
            # Use a and b channels from LAB for material color
            # Compute gradient magnitude from grayscale (lightness) to emphasize edges
            gray = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2GRAY).astype(np.float32)
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            grad = np.sqrt(gx * gx + gy * gy)
            # Normalize gradients to 0..1
            if grad.max() > 0:
                grad_norm = grad / grad.max()
            else:
                grad_norm = grad
            # Compose feature image with a, b, and gradient (higher priority) - stack into 3 channels
            feature_img = np.stack([
                a_normalized,     # material color a
                b_normalized,     # material color b
                grad_norm         # edges/texture
            ], axis=-1)
        else:
            feature_img = feature_img_default

        # Scale back to 0-1 range and convert to uint8
        feature_img = (feature_img - feature_img.min()) / (feature_img.max() - feature_img.min())
        rgb_enhanced = (feature_img * 255).astype(np.uint8)
        
        # Create downsampled ROI mask from boundary
        boundary_small = self.current_boundary * scale_factor
        roi_mask_small = np.zeros((small_h, small_w), dtype=np.uint8)
        cv2.fillPoly(roi_mask_small, [boundary_small.astype(np.int32)], 255)
        roi_area = np.sum(roi_mask_small > 0)
        # Compute buffer mask (in full-resolution coordinates) and map to small scale
        self._compute_buffer_mask()  # ensures self.buffer_mask exists
        if self.buffer_mask is not None:
            buffer_small = cv2.resize(self.buffer_mask.astype(np.uint8), (small_w, small_h), interpolation=cv2.INTER_NEAREST) > 0
        else:
            buffer_small = np.zeros_like(roi_mask_small, dtype=bool)
        # Build expanded ROI including interior + outside buffer area
        roi_expanded_small = (roi_mask_small > 0) | buffer_small
        # Create an edge/boundary mask (1-pixel) in the small scale and exclude it to make the boundary a hard split
        edge_mask_small = np.zeros((small_h, small_w), dtype=np.uint8)
        cv2.polylines(edge_mask_small, [boundary_small.astype(np.int32)], isClosed=True, color=255, thickness=1)
        edge_mask_small = edge_mask_small > 0
        # Exclude the boundary pixels from ROI to enforce split
        roi_expanded_small[edge_mask_small] = False
        # If shadow_robust was toggled, include its state in the debug prints
        print(f"DEBUG _generate_segments: shadow_robust={getattr(self, 'shadow_robust_var', False).get() if hasattr(self, 'shadow_robust_var') else False}")
        
        # Build the feature image used for Felzenszwalb segmentation
        # This abstracts the previous enhancement + gives us a shadow-robust path
        rgb_enhanced = self._build_segment_features(rgb_small)

        # Optionally pre-smooth to reduce texture/shadow noise
        if getattr(self, 'pre_smooth_var', None) and self.pre_smooth_var.get():
            try:
                # bilateral on BGR image (preserve edges)
                bgr_small = cv2.bilateralFilter(bgr_small, d=9, sigmaColor=75, sigmaSpace=75)
                rgb_small = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2RGB)
            except Exception:
                pass

        # Use Felzenszwalb segmentation - follows edges with straighter boundaries
        # Adjust scale based on ROI size and target segments. The 'scale' parameter controls coarseness; smaller -> more segments.
        estimated_scale = float(roi_area) / max(self.target_segments, 1)
        # Apply a moderate multiplier and enforce a small lower bound to allow fine segmentation
        init_scale = max(int(max(estimated_scale * 0.8, 10)), 10)
        # Adjust min_size based on ROI area - for fast preview use larger minimum to reduce compute
        if getattr(self, 'fast_preview_var', None) and self.fast_preview_var.get():
            min_size_base = max(50, int(max(roi_area * 0.001, 50)))
        else:
            min_size_base = max(20, int(max(roi_area * 0.0005, 20)))  # Allow fairly small segments (>=20 px)

        # Adaptive segmentation: try multiple attempts with decreasing scale/min_size if results are too coarse
        scale = init_scale
        min_size = min_size_base
        max_attempts = max_attempts_local
        attempt = 0
        target_threshold = max(5, int(self.target_segments // 20))  # aim for at least this many segments
        segments_small = None
        segments_full = None
        while attempt < max_attempts:
            segments_small = felzenszwalb(rgb_enhanced, scale=scale, sigma=0, min_size=min_size)
            # Upscale for evaluation
            segments_full = cv2.resize(segments_small, (w, h), interpolation=cv2.INTER_NEAREST)
            unique_full = len(np.unique(segments_full))
            print(f"DEBUG _generate_segments attempt={attempt}, scale={scale}, min_size={min_size}, unique={unique_full}")
            # If segmentation meets threshold, accept
            if unique_full >= target_threshold:
                break
            # Otherwise, make segmentation finer and retry (more aggressive reductions)
            attempt += 1
            old_scale, old_min = scale, min_size
            # Reduce scale by 40% (more gradual) and allow min_size down to 1
            scale = max(1, int(max(1, scale * 0.6)))
            min_size = max(1, int(max(1, min_size * 0.5)))
            print(f"DEBUG _generate_segments: retry={attempt}, scale {old_scale}->{scale}, min_size {old_min}->{min_size}, unique_full={unique_full}")
        # segments_small and segments_full set from last attempt

        
        # Mask out segments outside expanded ROI (interior + buffer) and renumber
        # Compute full-resolution ROI mask (interior) and buffer mask (precomputed by _compute_buffer_mask)
        roi_mask_full = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(roi_mask_full, [self.current_boundary.astype(np.int32)], 255)
        roi_mask_bool = (roi_mask_full > 0)
        # Ensure buffer mask computed
        self._compute_buffer_mask()
        if getattr(self, 'buffer_mask', None) is not None:
            buffer_mask_full = self.buffer_mask.astype(bool)
        else:
            buffer_mask_full = np.zeros((h, w), dtype=bool)
        # Build expanded ROI and exclude boundary edge pixels to enforce hard split
        roi_expanded = roi_mask_bool | buffer_mask_full
        edge_mask_full = np.zeros((h, w), dtype=np.uint8)
        cv2.polylines(edge_mask_full, [self.current_boundary.astype(np.int32)], isClosed=True, color=255, thickness=1)
        edge_mask_full = edge_mask_full > 0
        roi_expanded[edge_mask_full] = False

        segments = np.zeros_like(segments_full)
        segment_map = {}  # Map old IDs to new IDs
        new_id = 1
        
        # Kernels for morphological operations
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # Smooth edges
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))   # Break thin connections

        # Preallocate gradient image for boundary strength checks (used in merging)
        gray_full = cv2.cvtColor(self.clean_image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx = cv2.Sobel(gray_full, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_full, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag_full = np.sqrt(gx * gx + gy * gy)
        
        for old_id in np.unique(segments_full):
            seg_mask = (segments_full == old_id) & (roi_expanded)
            # Allow much smaller segments to survive when target is high
            if seg_mask.sum() < getattr(self, 'min_segment_px', 1000):  # Skip tiny segments
                continue
            
            seg_mask_uint8 = seg_mask.astype(np.uint8) * 255
            
            # First: Apply morphological closing to smooth squiggly edges (fill indentations from parking lines)
            closed = cv2.morphologyEx(seg_mask_uint8, cv2.MORPH_CLOSE, kernel_close)
            
            # Second: Apply morphological opening to break thin connections
            opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)
            
            # Find separate connected components after smoothing and breaking connections
            labeled_components, num_components = ndimage.label(opened > 0)
            
            # Assign each disconnected chunk its own segment ID
            for component_id in range(1, num_components + 1):
                component_mask = labeled_components == component_id
                if component_mask.sum() >= getattr(self, 'min_segment_px', 1000):  # Only keep chunks >= min_segment_px
                    # Optionally simplify the component contour to reduce squiggles
                    simp_mask = component_mask
                    try:
                        simp_mask = self._simplify_component_mask(component_mask, level=self.segment_smoothing_level)
                    except Exception:
                        pass
                    if simp_mask.sum() >= getattr(self, 'min_segment_px', 1000):
                        segments[simp_mask] = new_id
                        new_id += 1

        # After initial region construction, grad_mag_full is available for merging tests
        self._grad_mag_full = grad_mag_full
        
        self.segments = segments
        self.n_segments = segments.max()  # Actual segment count

        # Enforce boundary as hard split to ensure no segment crosses the boundary
        try:
            self._enforce_boundary_split()
        except Exception:
            pass

        # Optional auto-merge step to reduce spurious small regions (run after boundary split)
        if getattr(self, 'auto_merge_var', None) and self.auto_merge_var.get():
            try:
                self._postprocess_merge(target=self.target_segments)
            except Exception as e:
                print('Warning: postprocess merge failed', e)


        # Update smoothing level from current UI selection (if any)
        try:
            self.segment_smoothing_level = self.segment_smoothing_var.get()
        except Exception:
            pass

        # Display segments
        self._update_display()
        
        self.seg_status.config(text=f"✓ {self.n_segments} segments (target: {self.target_segments})")
        self._update_progress()

    def _segment_myself(self):
        """Start manual segmentation seeded by boundary+buffer (single region covering ROI+buffer)."""
        if self.current_boundary is None:
            messagebox.showwarning("Warning", "No boundary available")
            return
        h, w = self.clean_image.shape[:2]
        roi_mask_full = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(roi_mask_full, [self.current_boundary.astype(np.int32)], 255)
        # Include buffer if available
        self._compute_buffer_mask()
        if getattr(self, 'buffer_mask', None) is not None:
            roi_expanded = (roi_mask_full > 0) | self.buffer_mask
        else:
            roi_expanded = (roi_mask_full > 0)
        segments = np.zeros((h, w), dtype=np.int32)
        segments[roi_expanded] = 1
        self.segments = segments
        self.n_segments = 1
        self.segment_labels = {}
        self.manual_mode = True
        self.manual_status.config(text="Manual segmentation: Use Split Mode to draw splits")
        self._update_display()
        self._update_progress()

        
    def _refine_segments(self):
        """Increase segment count (more segments)."""
        self.target_segments = min(int(self.target_segments * 1.5), 2000)
        self._generate_segments()
    
    def _coarsen_segments(self):
        """Decrease segment count (fewer segments)."""
        self.target_segments = max(int(self.target_segments / 1.5), 10)
        self._generate_segments()
    
    def _update_title_for_mode(self):
        """Update title based on current mode."""
        # Mode selection has priority: ensure title reflects the actively selected mode
        try:
            if getattr(self, 'mode', None) == 'boundary':
                self.ax.set_title("Adjust boundary (drag corners to reshape, arrows=move, </>=rotate)")
            elif getattr(self, 'mode', None) == 'segments':
                if getattr(self, 'manual_mode', False):
                    self.ax.set_title("SPLIT MODE: Click segment, draw lines (Esc=reselect, Space=new line, Enter=apply)")
                else:
                    self.ax.set_title("SEGMENT MODE: Label, split, refine segments")
            elif getattr(self, 'mode', None) == 'label':
                self.ax.set_title("LABEL MODE: Click segment to label (right-click to remove)")
            elif getattr(self, 'mode', None) == 'access':
                self.ax.set_title("ACCESS MODE: Boundary access and road painting (use buttons to enable)")
            else:
                # Fallback to boundary-aware title
                if not self.boundary_approved:
                    self.ax.set_title("Adjust boundary (drag corners to reshape, arrows=move, </>=rotate)")
                elif self.manual_mode:
                    self.ax.set_title("SPLIT MODE: Click segment, draw lines (Esc=reselect, Space=new line, Enter=apply)")
                else:
                    self.ax.set_title("LABEL MODE: Click segment to label (right-click to remove)")
        except Exception:
            pass
        try:
            self.canvas.draw()
        except Exception:
            pass

    def _set_mode(self, mk: str):
        """Set the current UI mode and update visibility/state consistently.
        This overrides any fragile previous implementations and ensures tests (and UX)
        get deterministic behavior when switching modes.
        """
        try:
            self.mode = mk
        except Exception:
            pass
        # Update button visuals
        try:
            for k, b in getattr(self, 'mode_buttons', {}).items():
                try:
                    if k == mk:
                        b.config(relief=tk.SUNKEN, bg='#D3D3D3')
                    else:
                        b.config(relief=tk.RAISED, bg='#f0f0f0')
                except Exception:
                    pass
        except Exception:
            pass
        # Show/hide section frames (use labels registered in self.section_frames where possible)
        try:
            for name, (frame, btn) in getattr(self, 'section_frames', {}).items():
                try:
                    frame.pack_forget()
                except Exception:
                    pass
            # Conservative show sets based on mode
            if mk == 'boundary':
                try:
                    self.boundary_tools_frame.pack(pady=1, fill='x')
                except Exception:
                    pass
            elif mk == 'segments':
                # Ensure segments, class and access sections are visible
                for nstarts in ('4. Segments', '5. Class', '6. Access'):
                    for n, (f, _) in self.section_frames.items():
                        if n.startswith(nstarts):
                            try:
                                f.pack(pady=1, fill='x')
                            except Exception:
                                pass
                # Disable label button while in segments (explicit test expectation)
                try:
                    self.mode_buttons['label'].config(state=tk.DISABLED)
                except Exception:
                    pass
            elif mk == 'label':
                for nstarts in ('5. Class', '6. Access'):
                    for n, (f, _) in self.section_frames.items():
                        if n.startswith(nstarts):
                            try:
                                f.pack(pady=1, fill='x')
                            except Exception:
                                pass
                try:
                    self.mode_buttons['label'].config(state=tk.NORMAL)
                except Exception:
                    pass
            elif mk == 'access':
                for n, (f, _) in self.section_frames.items():
                    if n.startswith('6. Access'):
                        try:
                            f.pack(pady=1, fill='x')
                        except Exception:
                            pass
            else:
                # fallback: show boundary tools by default
                try:
                    self.boundary_tools_frame.pack(pady=1, fill='x')
                except Exception:
                    pass
        except Exception:
            pass
        # Update title and redraw
        try:
            self._update_title_for_mode()
        except Exception:
            pass
        try:
            self.canvas.draw()
        except Exception:
            pass
        # Enforce label button disabled state for segments mode (defensive)
        try:
            if getattr(self, 'mode', None) == 'segments':
                self.mode_buttons['label'].config(state=tk.DISABLED)
            else:
                self.mode_buttons['label'].config(state=tk.NORMAL)
        except Exception:
            pass
    
    def _on_class_selected(self):
        """Called when user selects a class - auto-switch to labeling mode."""
        # Auto-approve boundary if not already approved but boundary exists
        if not self.boundary_approved and self.current_boundary is not None:
            self._approve_boundary()
        
        if self.manual_mode:
            self._toggle_manual_mode()
        # Update display to show we're in labeling mode
        if self.boundary_approved and self.segments is not None:
            self.ax.set_title("Click segment to label (right-click to remove)")
            self.canvas.draw()
    
    def _toggle_manual_mode(self):
        """Toggle manual segmentation mode."""
        # Auto-approve boundary if switching to split mode and boundary not approved yet
        if not self.manual_mode and not self.boundary_approved and self.current_boundary is not None:
            self._approve_boundary()
        
        self.manual_mode = not self.manual_mode
        if self.manual_mode:
            # SPLIT mode active
            self.manual_btn.config(text="Split Mode (On)", bg='#90EE90', relief=tk.RAISED, font=("Arial", 8, "bold"))
            self.finalize_btn.config(state=tk.NORMAL)
            self.ax.set_title("SPLIT MODE: Click segment, draw lines (Esc=reselect, Space=new line, Enter=apply)")
            self.manual_status.config(text="Click a segment to select it for splitting")
        else:
            # SPLIT mode inactive
            self.manual_btn.config(text="Split Mode (Off)", bg='#FFD700', relief=tk.RAISED, font=("Arial", 8, "bold"))
            self.finalize_btn.config(state=tk.DISABLED)
            self.ax.set_title("Adjust boundary (arrows=move, </>=rotate) and press Enter to proceed" if not self.boundary_approved else "LABEL MODE: Click segment to label (right-click to remove)")
            self.manual_status.config(text="")
            self._clear_manual_line()
            self.splitting_segment_id = None  # Reset highlighted segment
            self._update_display()  # Restore normal display
        self.canvas.draw()
    
    def _clear_manual_line(self):
        """Clear manual line points and artists."""
        self.manual_line_points.clear()
        self.manual_polylines.clear()
        self.splitting_segment_id = None  # Reset highlighted segment
        for artist in self.manual_line_artists:
            try:
                artist.remove()
            except:
                pass
        self.manual_line_artists.clear()
    
    def _reset_submit_button(self):
        """Reset submit button to unsaved state."""
        self.submit_btn.config(text="✓ Submit Annotation", bg='#ADD8E6')
        self.has_unsaved_changes = True
    
    def _check_unsaved_changes(self):
        """Check if there are unsaved changes and prompt user.
        Returns True if OK to proceed (no changes or user chose to discard).
        Returns False if user wants to stay and save.
        """
        if not self.has_unsaved_changes:
            return True
        
        response = messagebox.askyesnocancel(
            "Unsaved Changes",
            "You have unsaved work on this image.\n\n"
            "Yes - Save changes and continue\n"
            "No - Discard changes and continue\n"
            "Cancel - Stay on current image"
        )
        
        if response is True:  # Yes - save and continue
            self._submit_annotation()
            return True
        elif response is False:  # No - discard and continue
            return True
        else:  # Cancel - stay on current image
            return False
    
    def _undo_split(self):
        """Undo the last manual split."""
        if not self.split_history:
            return
        
        # Restore previous segment state
        self.segments = self.split_history.pop()
        self.n_segments = self.segments.max()
        
        # Disable undo button if no more history
        if not self.split_history:
            self.undo_split_btn.config(state=tk.DISABLED)
        
        # Clear any labels for segments that no longer exist
        valid_seg_ids = set(range(1, self.n_segments + 1))
        self.segment_labels = {k: v for k, v in self.segment_labels.items() if k in valid_seg_ids}
        
        self._update_display()
        self._update_progress()
        self.seg_status.config(text=f"✓ {self.n_segments} segments (split undone)")
        print(f"Undone split - restored to {self.n_segments} segments")
    
    def _finalize_manual_split(self):
        """Finalize and apply the manual split."""
        # Add current polyline to completed list if it has points
        if len(self.manual_line_points) >= 2:
            self.manual_polylines.append(self.manual_line_points.copy())
            self.manual_line_points.clear()
        
        if len(self.manual_polylines) == 0:
            self.manual_status.config(text="Need at least 2 points!")
            messagebox.showwarning("Not enough points", "Click at least 2 points before finalizing")
            return

        # Update UI and then run split in background on GUI-enabled environments to keep UI responsive
        self.manual_status.config(text="Applying split...")
        try:
            if self._tk_available:
                # Disable buttons to prevent re-entry while processing
                try:
                    self.finalize_btn.config(state=tk.DISABLED)
                    self.manual_btn.config(state=tk.DISABLED)
                except Exception:
                    pass
                t = threading.Thread(target=self._apply_manual_split_background_safe, daemon=True)
                t.start()
            else:
                # Headless/test mode - run synchronously for tests
                self._apply_manual_split()
                self.manual_status.config(text="Split applied! Draw another or toggle off")
        except Exception:
            # Fallback to synchronous call on unexpected errors
            self._apply_manual_split()
            self.manual_status.config(text="Split applied! Draw another or toggle off")
    
    def _simplify_component_mask(self, mask, level='med'):
        """Simplify the polygon of a boolean component mask.
        level: 'off'|'low'|'med'|'high'
        Returns a boolean mask of the simplified polygon (same shape as mask).
        """
        eps_frac = self.SMOOTHING_EPS.get(level, 0.01)
        if eps_frac <= 0.0:
            return mask
        mask_uint8 = mask.astype('uint8') * 255
        # Apply a small morphological closing to fill jagged teeth before contour extraction
        kernel_size_map = {'off':1, 'low':3, 'med':5, 'high':7}
        k = kernel_size_map.get(level, 5)
        try:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            closed0 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
        except Exception:
            closed0 = mask_uint8
        contours, _ = cv2.findContours(closed0, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return mask
        c = max(contours, key=lambda x: cv2.contourArea(x))
        perim = cv2.arcLength(c, True)
        eps = max(1.0, perim * eps_frac)

        # Optional contour smoothing (Chaikin corner cutting) to remove spike extremes
        smoothing_iters_map = {'off': 0, 'low': 1, 'med': 2, 'high': 3}
        iters = smoothing_iters_map.get(level, 2)

        def chaikin(points, iterations=1):
            pts = points.copy()
            for _ in range(max(0, iterations)):
                if len(pts) < 2:
                    break
                new_pts = []
                n = len(pts)
                for i in range(n):
                    p0 = pts[i]
                    p1 = pts[(i + 1) % n]
                    q = 0.75 * p0 + 0.25 * p1
                    r = 0.25 * p0 + 0.75 * p1
                    new_pts.append(q)
                    new_pts.append(r)
                pts = np.array(new_pts)
            return pts

        try:
            pts = c.reshape(-1, 2).astype(float)
            if iters > 0 and len(pts) >= 3:
                smoothed = chaikin(pts, iterations=iters)
                # Remove consecutive duplicates and ensure at least 3 points
                # Round to integer and remove duplicates preserving order
                rounded = np.round(smoothed).astype(int)
                # Remove duplicates by checking consecutive equal rows
                keep_idx = [0]
                for i in range(1, len(rounded)):
                    if not np.array_equal(rounded[i], rounded[i - 1]):
                        keep_idx.append(i)
                rounded = rounded[keep_idx]
                if len(rounded) >= 3:
                    smooth_cnt = rounded.reshape(-1, 1, 2).astype(np.int32)
                    approx = cv2.approxPolyDP(smooth_cnt, eps, True)
                else:
                    approx = cv2.approxPolyDP(c, eps, True)
            else:
                approx = cv2.approxPolyDP(c, eps, True)
        except Exception:
            approx = cv2.approxPolyDP(c, eps, True)

        poly_mask = np.zeros_like(mask_uint8)
        try:
            cv2.fillPoly(poly_mask, [approx], 255)
            poly_bool = poly_mask.astype(bool)
            # Do not allow simplification to increase area beyond original - intersect with original mask
            poly_bool = poly_bool & mask
            # Ensure we didn't dramatically shrink area; otherwise fallback
            area_thresholds = {'off': 1.0, 'low': 0.2, 'med': 0.5, 'high': 0.3}
            frac = area_thresholds.get(level, 0.3)
            if poly_bool.sum() >= max( int(mask.sum() * frac), 1 ):
                return poly_bool
            else:
                return mask
        except Exception:
            return mask

    def _postprocess_merge(self, target:int=None, min_size:int=None, boundary_grad_thresh:float=20.0):
        if min_size is None:
            min_size = getattr(self, 'min_segment_px', 100)
        """Merge small regions towards neighbors based on color similarity and weak boundary strength.
        - target: desired approximate number of segments. Merge small regions until reaching target or no merges possible.
        - min_size: size below which a region is considered 'small' and eligible to be merged.
        - boundary_grad_thresh: mean gradient magnitude threshold; don't merge across strong boundaries.
        """
        if self.segments is None:
            return
        seg = self.segments.copy()
        h, w = seg.shape
        unique_ids = [int(x) for x in np.unique(seg) if x != 0]
        if len(unique_ids) <= 1:
            return
        # compute region areas and mean color
        rgb = cv2.cvtColor(self.clean_image, cv2.COLOR_BGR2RGB).astype(float)
        areas = {}
        means = {}
        for uid in unique_ids:
            mask = (seg == uid)
            areas[uid] = int(mask.sum())
            if areas[uid] > 0:
                means[uid] = rgb[mask].mean(axis=0)
            else:
                means[uid] = np.array([0.0, 0.0, 0.0])
        # precompute adjacency and boundary gradients using edge pairs (right and down neighbors)
        from collections import defaultdict
        boundary_pixels = defaultdict(list)  # (a,b) -> list of coords
        for dy, dx in [(0,1),(1,0)]:
            a = seg[:, :-dx or None]
            b = seg[:, dx or None:]
            if dx == 1:
                coords = np.argwhere(a != b)
                for y,x in coords:
                    id_a = int(seg[y,x])
                    id_b = int(seg[y,x+1])
                    if id_a == id_b or id_a == 0 or id_b == 0:
                        continue
                    key = tuple(sorted((id_a, id_b)))
                    boundary_pixels[key].append((y,x))
            else:
                coords = np.argwhere(a != b)
                for y,x in coords:
                    id_a = int(seg[y,x])
                    id_b = int(seg[y+1,x])
                    if id_a == id_b or id_a == 0 or id_b == 0:
                        continue
                    key = tuple(sorted((id_a, id_b)))
                    boundary_pixels[key].append((y,x))
        # iterative small-region merging (merge small slivers even if current count <= target)
        current_unique = set(unique_ids)
        target = target or self.target_segments
        iters = 0
        while iters < 1000:
            iters += 1
            # find smallest region
            small_id = min(current_unique, key=lambda u: areas.get(u, 0))
            # Stop condition: smallest region large enough and we are at or below target
            if areas.get(small_id, 0) >= min_size and len(current_unique) <= max(1, int(target)):
                break
            # find neighbors
            neighbor_candidates = []
            for pair, coords in boundary_pixels.items():
                if small_id in pair:
                    other = pair[0] if pair[1] == small_id else pair[1]
                    neighbor_candidates.append((other, coords))
            if not neighbor_candidates:
                # isolated small region; remove it by assigning to largest region
                largest = max(current_unique, key=lambda u: areas.get(u,0))
                if largest == small_id:
                    break
                seg[seg==small_id] = largest
                current_unique.remove(small_id)
            else:
                # evaluate best neighbor by color distance and boundary gradient
                best_n = None
                best_score = float('inf')
                for other, coords in neighbor_candidates:
                    grads = [self._grad_mag_full[y,x] for (y,x) in coords]
                    mean_grad = float(np.mean(grads)) if grads else 0.0
                    if mean_grad > boundary_grad_thresh:
                        continue
                    # color distance (Euclidean in RGB)
                    dist = np.linalg.norm(means.get(small_id, np.zeros(3)) - means.get(other, np.zeros(3)))
                    # prefer neighbor with small distance and reasonable area
                    score = dist / (1 + areas.get(other,1))
                    if score < best_score:
                        best_score = score
                        best_n = other
                if best_n is None:
                    # no neighbor suitable (strong borders) - stop
                    break
                seg[seg==small_id] = best_n
                current_unique.remove(small_id)
            # recompute adjacency/areas/means for next iteration
            boundary_pixels = defaultdict(list)
            unique_ids2 = set([int(x) for x in np.unique(seg) if x != 0])
            for dy, dx in [(0,1),(1,0)]:
                a = seg[:, :-dx or None]
                b = seg[:, dx or None:]
                if dx == 1:
                    coords2 = np.argwhere(a != b)
                    for y,x in coords2:
                        id_a = int(seg[y,x])
                        id_b = int(seg[y,x+1])
                        if id_a == id_b or id_a == 0 or id_b == 0:
                            continue
                        key = tuple(sorted((id_a, id_b)))
                        boundary_pixels[key].append((y,x))
                else:
                    coords2 = np.argwhere(a != b)
                    for y,x in coords2:
                        id_a = int(seg[y,x])
                        id_b = int(seg[y+1,x])
                        if id_a == id_b or id_a == 0 or id_b == 0:
                            continue
                        key = tuple(sorted((id_a, id_b)))
                        boundary_pixels[key].append((y,x))
            areas = {}
            means = {}
            for uid in unique_ids2:
                mask = (seg == uid)
                areas[uid] = int(mask.sum())
                if areas[uid] > 0:
                    means[uid] = rgb[mask].mean(axis=0)
                else:
                    means[uid] = np.array([0.0,0.0,0.0])
        # reassign compact ids
        new_seg = np.zeros_like(seg, dtype=np.int32)
        new_id = 1
        for uid in sorted([int(x) for x in np.unique(seg) if x != 0]):
            new_seg[seg == uid] = new_id
            new_id += 1
        self.segments = new_seg
        self.n_segments = int(self.segments.max())
        print(f"_postprocess_merge: reduced to {self.n_segments} segments (target {target}) after {iters} iterations")
        return


    def _on_smoothing_change(self):
        """Handle smoothing radio changes."""
        try:
            self.segment_smoothing_level = self.segment_smoothing_var.get()
        except Exception:
            self.segment_smoothing_level = 'med'
        # If segments exist, re-render to reflect smoothing choice
        try:
            if self.segments is not None:
                self._update_display()
        except Exception:
            pass

    def _on_shadow_robust_toggle(self):
        """Handle toggle for shadow-robust segmentation."""
        try:
            self.shadow_robust = self.shadow_robust_var.get()
        except Exception:
            self.shadow_robust = False
        # Recompute segments if boundary approved
        if self.boundary_approved:
            self.seg_status.config(text="Regenerating segments with new settings...")
            self._generate_segments()

    def _apply_min_segment_px(self):
        """Apply min segment size from the entry control."""
        val = self.min_segment_entry.get()
        try:
            v = int(val)
            v = max(1, v)
            self.min_segment_px = v
            messagebox.showinfo("Min segment updated", f"Min segment set to {v} px")
            # If segments exist, re-run to apply threshold
            if self.boundary_approved:
                self.seg_status.config(text="Regenerating segments with new min size...")
                self._generate_segments()
        except Exception:
            messagebox.showwarning("Invalid value", "Enter a valid integer for min segment px")

    def _build_segment_features(self, rgb_small: np.ndarray) -> np.ndarray:
        """Thin wrapper calling implementation moved to labeling.core.segment._build_segment_features
        (implementation was copied verbatim into that module as part of the atomic move).
        """
        return segment_core._build_segment_features(self, rgb_small)

    def _apply_manual_split_background_safe(self):
        """Background-safe split handler for common closed-loop polygons.
        If closed-loop polygons are detected in the manual polylines, handle them quickly
        in a background thread and schedule final UI updates on the main thread. For
        non-closed line splits we fall back to performing the full split on the main
        thread (to keep GUI actions safe).
        """
        self._running_in_background = True
        try:
            manual_polylines = [p.copy() for p in getattr(self, 'manual_polylines', [])]
            if len(manual_polylines) == 0 or self.segments is None:
                try:
                    self.root.after(0, lambda: self.manual_status.config(text="No lines to apply"))
                except Exception:
                    pass
                self._running_in_background = False
                return

            segs_local = self.segments.copy()
            seg_id = self.splitting_segment_id
            seg_mask = (segs_local == seg_id)
            seg_area = int(seg_mask.sum())
            h, w = segs_local.shape
            if seg_area == 0:
                try:
                    self.root.after(0, lambda: self.manual_status.config(text="Selected segment empty"))
                except Exception:
                    pass
                self._running_in_background = False
                return

            # compute edge coords for this segment
            seg_uint8 = seg_mask.astype(np.uint8)
            kernel = np.ones((3, 3), np.uint8)
            eroded = cv2.erode(seg_uint8, kernel, iterations=1)
            edge_mask = seg_uint8 - eroded
            edge_coords = np.argwhere(edge_mask > 0)
            if edge_coords.size == 0:
                try:
                    contours, _ = cv2.findContours((seg_uint8 * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    if contours:
                        c = max(contours, key=lambda x: cv2.contourArea(x))
                        edge_coords = np.array([[pt[0][1], pt[0][0]] for pt in c.reshape(-1,1,2)])
                except Exception:
                    edge_coords = np.empty((0,2), dtype=int)

            segments_added_local = 0
            new_segment_selected = None

            # quick path: handle closed polygons only
            for manual_line_points in manual_polylines:
                pts = np.array(manual_line_points, dtype=np.int32)
                if pts.shape[0] < 3:
                    continue
                p0 = pts[0]; p_last = pts[-1]
                dist_endpoints = np.linalg.norm(np.array([p0[1], p0[0]]) - np.array([p_last[1], p_last[0]]))
                if edge_coords.size == 0:
                    continue
                dist_to_edges_0 = np.linalg.norm(edge_coords - np.array([p0[1], p0[0]]), axis=1)
                nearest_edge_dist_0 = dist_to_edges_0.min()
                dist_to_edges_last = np.linalg.norm(edge_coords - np.array([p_last[1], p_last[0]]), axis=1)
                nearest_edge_dist_last = dist_to_edges_last.min()
                is_closed_loop = (dist_endpoints < nearest_edge_dist_0 and dist_endpoints < nearest_edge_dist_last)
                if not is_closed_loop:
                    # Not a closed polygon; fallback to main-thread full split
                    try:
                        self.root.after(0, lambda: self._apply_manual_split())
                    except Exception:
                        pass
                    self._running_in_background = False
                    return

                # closed polygon: fill and assign if large enough
                poly_pts = pts.astype(np.int32)
                poly_mask_local = np.zeros((h, w), dtype=np.uint8)
                try:
                    cv2.fillPoly(poly_mask_local, [poly_pts], 255)
                except Exception:
                    continue
                constrained_poly = (poly_mask_local > 0) & seg_mask
                area_poly = int(constrained_poly.sum())
                poly_min = max(100, int(seg_area * 0.01))
                if area_poly >= poly_min:
                    new_seg_id_local = int(segs_local.max()) + 1
                    segs_local[constrained_poly] = new_seg_id_local
                    segments_added_local += 1
                    new_segment_selected = new_seg_id_local

            # Finalize on main thread if any closed-loop handled
            if segments_added_local > 0:
                def ui_finalize():
                    try:
                        self.segments = segs_local
                        old_n = self.n_segments
                        self.n_segments = int(self.segments.max())
                        print(f"\nSegment count: {old_n} -> {self.n_segments} (added {segments_added_local})")
                        self.manual_status.config(text="Split applied! Draw another or toggle off")
                        try:
                            self.finalize_btn.config(state=tk.NORMAL)
                            self.manual_btn.config(state=tk.NORMAL)
                        except Exception:
                            pass
                        if new_segment_selected is not None:
                            self.splitting_segment_id = int(new_segment_selected)
                            try:
                                self._update_display_with_highlight(self.splitting_segment_id)
                            except Exception:
                                pass
                        else:
                            self.splitting_segment_id = None
                        self._clear_manual_line()
                        self._reset_submit_button()
                        self._update_display()
                        self._update_progress()
                    except Exception as e:
                        print("Error finalizing background split:", e)
                try:
                    self.root.after(0, ui_finalize)
                except Exception:
                    ui_finalize()
            else:
                # Nothing created; report to user
                try:
                    self.root.after(0, lambda: self.manual_status.config(text="Split did not create any region"))
                except Exception:
                    pass
        finally:
            self._running_in_background = False

    def _apply_manual_split(self):
        """Apply manual split lines to segments - divides by multiple polylines."""
        if len(self.manual_polylines) == 0 or self.segments is None:
            return
        
        # Check if user selected a segment to split
        if not hasattr(self, 'splitting_segment_id') or self.splitting_segment_id is None:
            self.manual_status.config(text="No segment selected! Click a segment first")
            return
        
        # Save current state for undo
        self.split_history.append(self.segments.copy())
        if len(self.split_history) > 10:  # Limit history to last 10 splits
            self.split_history.pop(0)
        self.undo_split_btn.config(state=tk.NORMAL)
        
        h, w = self.segments.shape
        print(f"\n=== MANUAL SPLIT DEBUG ===")
        print(f"Image size: {h}x{w}")
        print(f"Target segment: {self.splitting_segment_id}")
        print(f"Number of polylines: {len(self.manual_polylines)}")
        
        # Create line mask from ALL polylines
        # Use 1-pixel line - just needs to create a gap for flood fill
        line_thickness = 1  # 1 pixel wide
        endpoint_radius = 2  # 2 pixel radius at endpoints to ensure connection
        
        # Only split the selected segment (not all segments the line touches)
        seg_id = self.splitting_segment_id
        new_seg_id = self.segments.max() + 1
        print(f"Splitting segment {seg_id} into multiple regions")
        
        seg_mask = (self.segments == seg_id)
        seg_area = np.sum(seg_mask)
        print(f"Processing segment {seg_id} (area: {seg_area} pixels)")
        
        # Create debug visualization
        debug_img = cv2.cvtColor(self.clean_image, cv2.COLOR_BGR2RGB).copy()
        # Dim everything except the segment
        mask_3ch = np.stack([seg_mask] * 3, axis=2)
        debug_img[~mask_3ch] = (debug_img[~mask_3ch] * 0.3).astype(np.uint8)
        
        # Find edge pixels of the segment (constrained to selected segment only)
        seg_uint8 = seg_mask.astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(seg_uint8, kernel, iterations=1)
        edge_mask = seg_uint8 - eroded
        edge_coords = np.argwhere(edge_mask > 0)  # (y, x) format
        
        # Fallback: if erosion didn't produce edge pixels (rare), extract contour points
        if len(edge_coords) == 0:
            try:
                contours, _ = cv2.findContours((seg_uint8 * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                if contours:
                    c = max(contours, key=lambda x: cv2.contourArea(x))
                    # contour points are (x,y); convert to (y,x)
                    edge_coords = np.array([ [pt[0][1], pt[0][0]] for pt in c.reshape(-1,1,2) ])
            except Exception:
                edge_coords = np.empty((0,2), dtype=int)

        if len(edge_coords) == 0:
            print(f"  → No edges found - segment too small")
            self.manual_status.config(text="Segment too small to split!")
            return
        
        # Create combined line mask from all polylines
        line_mask = np.zeros((h, w), dtype=np.uint8)
        segments_added = 0  # Counter for how many new segments were created during this operation
        # Prepare vars used for debug visualization to avoid UnboundLocalError when closed-loops are handled
        edge_pt_0 = None
        edge_pt_last = None
        extended_points_array = None
        
        for line_idx, manual_line_points in enumerate(self.manual_polylines):
            print(f"  Processing line {line_idx + 1} with {len(manual_line_points)} points")
            points = np.array(manual_line_points, dtype=np.int32)
            
            # Check if this is a closed polyline (endpoints close to each other)
            p0 = points[0]
            p_last = points[-1]
            dist_endpoints = np.linalg.norm(np.array([p0[1], p0[0]]) - np.array([p_last[1], p_last[0]]))
            
            # Snap first point - consider both edge pixels AND the last point
            dist_to_edges_0 = np.linalg.norm(edge_coords - np.array([p0[1], p0[0]]), axis=1)
            nearest_edge_dist_0 = dist_to_edges_0.min()
            
            # Snap last point - consider both edge pixels AND the first point
            dist_to_edges_last = np.linalg.norm(edge_coords - np.array([p_last[1], p_last[0]]), axis=1)
            nearest_edge_dist_last = dist_to_edges_last.min()
            
            # Determine if this should be a closed loop
            is_closed_loop = (dist_endpoints < nearest_edge_dist_0 and dist_endpoints < nearest_edge_dist_last)
            
            if is_closed_loop:
                # Treat a closed polyline as a filled polygon: if the drawn polygon lies inside the
                # selected segment, create a new segment from that polygon rather than stretching to the boundary.
                poly_pts = points.astype(np.int32)
                poly_mask_local = np.zeros((h, w), dtype=np.uint8)
                try:
                    cv2.fillPoly(poly_mask_local, [poly_pts], 255)
                except Exception:
                    poly_mask_local = np.zeros((h, w), dtype=np.uint8)
                constrained_poly = (poly_mask_local > 0) & seg_mask
                area_poly = int(np.sum(constrained_poly))
                print(f"    Closed polygon: area inside segment = {area_poly} px")
                # Minimum meaningful size threshold (consistent with other splits)
                poly_min = max(100, int(seg_area * 0.01))
                if area_poly >= poly_min:
                    new_seg_id_local = self.segments.max() + 1
                    self.segments[constrained_poly] = new_seg_id_local
                    segments_added += 1
                    print(f"  → Closed-loop split: created new seg {new_seg_id_local} with area {area_poly}")
                    # Visualize successful polygon assignment in debug image (green fill)
                    try:
                        cv2.polylines(debug_img, [poly_pts], isClosed=True, color=(0,255,0), thickness=2)
                        mask_rgb = np.zeros_like(debug_img)
                        mask_rgb[constrained_poly] = (0,255,0)
                        alpha = 0.4
                        debug_img = (debug_img * (1-alpha) + mask_rgb * alpha).astype(np.uint8)
                    except Exception:
                        pass
                    # Keep the newly created segment selected so user can continue edits
                    self.splitting_segment_id = new_seg_id_local
                    try:
                        self._update_display_with_highlight(self.splitting_segment_id)
                    except Exception:
                        pass
                else:
                    print(f"    Closed-loop too small (min {poly_min}) - ignoring polygon")
                    try:
                        cv2.polylines(debug_img, [poly_pts], isClosed=True, color=(255,0,0), thickness=2)
                    except Exception:
                        pass
                # Move on to next manual polyline (closed-loop handled)
                continue
            else:
                # Smart snapping: prefer nearby drawn endpoints, but otherwise snap to nearest edge pixel
                # and avoid extremely long diagonals by limiting snap distance.
                max_snap_dist = max(20, int(min(h, w) * 0.07))  # 7% of smaller dim, at least 20px

                # Build list of other endpoints (x,y) from all other polylines
                other_endpoints = []
                for other_idx, ol in enumerate(self.manual_polylines):
                    if other_idx == line_idx:
                        continue
                    if len(ol) >= 1:
                        other_endpoints.append(tuple(ol[0]))
                        other_endpoints.append(tuple(ol[-1]))
                other_eps_arr = np.array([[pt[1], pt[0]] for pt in other_endpoints]) if other_endpoints else np.empty((0, 2))

                # Start point snapping
                nearest_edge_0_idx = np.argmin(dist_to_edges_0)
                nearest_edge_0 = edge_coords[nearest_edge_0_idx]  # (y, x)
                nearest_edge_dist_0 = float(dist_to_edges_0[nearest_edge_0_idx])

                chosen0 = None
                chosen0_source = 'edge'
                # If any other drawn endpoint is closer (and within reasonable snap dist), prefer it
                if other_eps_arr.shape[0] > 0:
                    d_eps0 = np.linalg.norm(other_eps_arr - np.array([p0[1], p0[0]]), axis=1)
                    ep0_idx = int(np.argmin(d_eps0))
                    ep0_dist = float(d_eps0[ep0_idx])
                    if ep0_dist <= min(nearest_edge_dist_0, max_snap_dist):
                        # use other drawn endpoint
                        chosen0 = other_endpoints[ep0_idx]
                        chosen0_source = 'endpoint'
                if chosen0 is None:
                    # If nearest edge is within allowed snap distance, use it; otherwise still use it but mark as distant
                    chosen0 = (int(nearest_edge_0[1]), int(nearest_edge_0[0]))
                    chosen0_source = 'edge' if nearest_edge_dist_0 <= max_snap_dist else 'edge_distant'
                edge_pt_0 = chosen0

                # Last point snapping - avoid choosing the same exact edge pixel as start
                nearest_edge_last_idx = np.argmin(dist_to_edges_last)
                # If the nearest is same as start, try second-best
                if nearest_edge_last_idx == nearest_edge_0_idx and len(dist_to_edges_last) > 1:
                    sorted_idx = np.argsort(dist_to_edges_last)
                    nearest_edge_last_idx = int(sorted_idx[1])
                nearest_edge_last = edge_coords[nearest_edge_last_idx]
                nearest_edge_dist_last = float(dist_to_edges_last[nearest_edge_last_idx])

                chosen_last = None
                chosen_last_source = 'edge'
                if other_eps_arr.shape[0] > 0:
                    d_eplast = np.linalg.norm(other_eps_arr - np.array([p_last[1], p_last[0]]), axis=1)
                    ep_last_idx = int(np.argmin(d_eplast))
                    ep_last_dist = float(d_eplast[ep_last_idx])
                    if ep_last_dist <= min(nearest_edge_dist_last, max_snap_dist):
                        chosen_last = other_endpoints[ep_last_idx]
                        chosen_last_source = 'endpoint'
                if chosen_last is None:
                    chosen_last = (int(nearest_edge_last[1]), int(nearest_edge_last[0]))
                    chosen_last_source = 'edge' if nearest_edge_dist_last <= max_snap_dist else 'edge_distant'
                edge_pt_last = chosen_last

                # Debug print what snapping source we chose and store for tests/debugging
                try:
                    print(f"    Snap sources: start={chosen0_source}, end={chosen_last_source}, max_snap_dist={max_snap_dist}")
                except Exception:
                    pass
                try:
                    # Save last snap info for tests / debugging
                    self._last_snap_info = {
                        'start': {'pt': tuple(edge_pt_0), 'source': chosen0_source},
                        'end': {'pt': tuple(edge_pt_last), 'source': chosen_last_source}
                    }
                except Exception:
                    pass

                # If snapping caused endpoints to collapse (very close together), try a contour-intersection or farthest-projection fallback
                def _stretch_endpoints_if_collapsed(ep0, ep1, seg_uint8_local):
                    dx = ep0[0] - ep1[0]
                    dy = ep0[1] - ep1[1]
                    if dx*dx + dy*dy > 25:
                        return ep0, ep1
                    try:
                        contours, _ = cv2.findContours(seg_uint8_local*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                        if contours:
                            c = max(contours, key=lambda x: cv2.contourArea(x)).reshape(-1,2)
                            p0f = np.array(p0, dtype=float)
                            p1f = np.array(p_last, dtype=float)
                            line_dir = p1f - p0f
                            if np.linalg.norm(line_dir) > 1e-6:
                                inters = []
                                for i in range(len(c)):
                                    A = c[i]
                                    B = c[(i+1) % len(c)]
                                    A = np.array([float(A[0]), float(A[1])])
                                    B = np.array([float(B[0]), float(B[1])])
                                    s = B - A
                                    r = line_dir
                                    M = np.column_stack((s, -r))
                                    bvec = (p0f - A)
                                    try:
                                        sol, *_ = np.linalg.lstsq(M, bvec, rcond=None)
                                        t = float(sol[0]); u = float(sol[1])
                                        if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
                                            X = A + t * s
                                            inters.append((float(X[0]), float(X[1])))
                                    except Exception:
                                        continue
                                if len(inters) >= 2:
                                    us = [np.dot(np.array(p)-p0f, line_dir)/np.dot(line_dir,line_dir) for p in inters]
                                    sidx = np.argsort(us)
                                    pA = inters[sidx[0]]
                                    pB = inters[sidx[-1]]
                                    return (int(round(pA[0])), int(round(pA[1]))), (int(round(pB[0])), int(round(pB[1])))
                            # If contour intersection failed, fallback to projection extremes along the line within the segment
                            pts = np.argwhere(seg_uint8_local > 0)  # (y,x)
                            if pts.shape[0] > 0:
                                pts_xy = np.vstack([pts[:,1], pts[:,0]]).T.astype(float)
                                p0f = np.array(p0, dtype=float)
                                p1f = np.array(p_last, dtype=float)
                                r = p1f - p0f
                                denom = np.dot(r,r)
                                if denom > 1e-6:
                                    us = np.dot(pts_xy - p0f, r) / denom
                                    min_idx = np.argmin(us)
                                    max_idx = np.argmax(us)
                                    pA = pts_xy[min_idx]
                                    pB = pts_xy[max_idx]
                                    return (int(round(pA[0])), int(round(pA[1]))), (int(round(pB[0])), int(round(pB[1])))
                    except Exception:
                        pass
                    return ep0, ep1

                edge_pt_0, edge_pt_last = _stretch_endpoints_if_collapsed(edge_pt_0, edge_pt_last, seg_uint8)
                
                print(f"    Snapping: {tuple(p0)} -> {edge_pt_0}, {tuple(p_last)} -> {edge_pt_last}")
            
            # Create extended points array
            extended_points = [edge_pt_0] + [tuple(pt) for pt in points[1:-1]] + [edge_pt_last]
            extended_points_array = np.array(extended_points, dtype=np.int32)
            
            # Add this polyline to the combined line mask
            cv2.polylines(line_mask, [extended_points_array], isClosed=False, color=255, thickness=line_thickness)
            for point in [extended_points_array[0], extended_points_array[-1]]:
                cv2.circle(line_mask, tuple(point), radius=endpoint_radius, color=255, thickness=-1)
            
            # Visualize this line in debug image (different colors for different lines)
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
            color = colors[line_idx % len(colors)]
            cv2.polylines(debug_img, [extended_points_array], isClosed=False, color=color, thickness=3)
        
        # Keep a copy of the unconstrained combined line mask so we can split multiple segments
        full_line_mask = line_mask.copy()
        # CRITICAL: Constrain line_mask to only pixels within the selected segment
        # This prevents the line from extending into neighboring segments
        line_mask = line_mask & seg_mask.astype(np.uint8) * 255
        
        print(f"Combined line mask pixels (within segment): {np.sum(line_mask > 0)}")
        
        # Add visualization to debug image
        # Show edge pixels in blue
        debug_img[edge_mask > 0] = [0, 0, 255]
        # Draw user's original points in yellow
        for pt in points:
            cv2.circle(debug_img, tuple(pt), 5, (255, 255, 0), -1)
        # Draw snapped endpoints in cyan (if available)
        if edge_pt_0 is not None:
            try:
                cv2.circle(debug_img, edge_pt_0, 8, (0, 255, 255), -1)
            except Exception:
                pass
        if edge_pt_last is not None:
            try:
                cv2.circle(debug_img, edge_pt_last, 8, (0, 255, 255), -1)
            except Exception:
                pass
        # Draw extended line in red (if available)
        if extended_points_array is not None:
            try:
                cv2.polylines(debug_img, [extended_points_array], isClosed=False, color=(255, 0, 0), thickness=3)
            except Exception:
                pass
        
        # If the constrained line is tiny (e.g., endpoints collapsed), try to compute true boundary intersections
        if line_mask.sum() < 20:
            try:
                # find contour of segment and intersect with infinite line through original endpoints
                contours, _ = cv2.findContours((seg_mask.astype(np.uint8)*255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                if contours:
                    c = max(contours, key=lambda x: cv2.contourArea(x)).reshape(-1, 2)
                    p0f = np.array(points[0], dtype=float)
                    p1f = np.array(points[-1], dtype=float)
                    line_dir = p1f - p0f
                    if np.linalg.norm(line_dir) > 1e-6:
                        inters = []
                        for i in range(len(c)):
                            a = c[i]
                            b = c[(i+1) % len(c)]
                            A = np.array([a[0], a[1]], dtype=float)
                            B = np.array([b[0], b[1]], dtype=float)
                            s = B - A
                            r = p1f - p0f
                            M = np.column_stack((s, -r))  # solves M * [t,u] = (p0 - A)
                            bvec = (p0f - A)
                            try:
                                # Solve linear system robustly
                                sol, *_ = np.linalg.lstsq(M, bvec, rcond=None)
                                t = float(sol[0]); u = float(sol[1])
                                if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
                                    X = A + t * s
                                    inters.append((float(X[0]), float(X[1])))
                            except Exception:
                                continue
                        # after checking all contour segments, if we found 2 or more intersections, pick farthest apart
                        if len(inters) >= 2:
                            # pick two that are farthest along the line
                            # project to line parameter u
                            us = [np.dot(np.array(p)-p0f, line_dir)/np.dot(line_dir,line_dir) for p in inters]
                            sorted_idx = np.argsort(us)
                            pA = inters[sorted_idx[0]]
                            pB = inters[sorted_idx[-1]]
                            edge_pt_0 = (int(round(pA[0])), int(round(pA[1])))
                            edge_pt_last = (int(round(pB[0])), int(round(pB[1])))
                            # rebuild extended points array between these two exact intersection endpoints
                            extended_points = [edge_pt_0, edge_pt_last]
                            extended_points_array = np.array(extended_points, dtype=np.int32)
                            # rebuild full and constrained masks
                            cv2.polylines(line_mask, [extended_points_array], isClosed=False, color=255, thickness=line_thickness)
                            for point in [extended_points_array[0], extended_points_array[-1]]:
                                cv2.circle(line_mask, tuple(point), radius=endpoint_radius, color=255, thickness=-1)
            except Exception:
                pass

        # Subtract the line from the segment to create a gap
        seg_without_line = seg_mask.copy()
        seg_without_line[line_mask > 0] = False
        
        # Find connected components on each side of the line
        labeled_regions, num_regions = ndimage.label(seg_without_line)
        
        print(f"  After removing line, found {num_regions} regions")
        
        # If constrained intersection was tiny, try widening the line to force a split
        if num_regions < 2 and line_mask.sum() > 0:
            for widen in (3,5,9):
                try:
                    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (widen, widen))
                    dil = cv2.dilate(line_mask, k, iterations=1)
                    seg_without_line2 = seg_mask.copy()
                    seg_without_line2[dil > 0] = False
                    labeled2, num2 = ndimage.label(seg_without_line2)
                    print(f"  Retry widen={widen}: found {num2} regions")
                    if num2 >= 2:
                        labeled_regions = labeled2
                        num_regions = num2
                        line_mask = dil
                        break
                except Exception:
                    continue
        
        if num_regions < 2:
            # If this line didn't split the selected segment and we are running in the
            # background worker for UI responsiveness, prefer to fall back to the main-thread
            # full split routine (safer for interactions). If we are in the main thread,
            # continue with seed-split attempts as before.
            if getattr(self, '_running_in_background', False):
                # Schedule the full synchronous split to run in the main thread (fallback)
                try:
                    self.root.after(0, lambda: self._apply_manual_split())
                except Exception:
                    # As last resort, do nothing here to avoid blocking
                    pass
                return

            # 1) Sample along the drawn line and perform orthogonal seed floods to find two opposite seeds within the segment.
            p0f = np.array(points[0], dtype=float)
            p1f = np.array(points[-1], dtype=float)
            line_dir = p1f - p0f
            found_seed_split = False
            if np.linalg.norm(line_dir) > 1e-6:
                norm = np.array([-line_dir[1], line_dir[0]])
                for t_frac in (0.2, 0.35, 0.5, 0.65, 0.8):
                    p_mid = p0f + t_frac * line_dir
                    if np.linalg.norm(norm) < 1e-6:
                        continue
                    n = norm / np.linalg.norm(norm)
                    for delta in (8, 16, 32, 64):
                        sA = np.round(p_mid + n * delta).astype(int)
                        sB = np.round(p_mid - n * delta).astype(int)
                        h, w = seg_mask.shape
                        if not (0 <= sA[1] < h and 0 <= sA[0] < w and 0 <= sB[1] < h and 0 <= sB[0] < w):
                            continue
                        if not (seg_mask[sA[1], sA[0]] and seg_mask[sB[1], sB[0]]):
                            continue
                        # flood fill from seeds
                        from collections import deque
                        def flood(seed):
                            visited = np.zeros_like(seg_mask, dtype=bool)
                            q = deque()
                            q.append((seed[1], seed[0]))
                            visited[seed[1], seed[0]] = True
                            while q:
                                y,x = q.popleft()
                                for ny in range(max(0,y-1), min(seg_mask.shape[0], y+2)):
                                    for nx in range(max(0,x-1), min(seg_mask.shape[1], x+2)):
                                        if not visited[ny,nx] and seg_mask[ny,nx]:
                                            visited[ny,nx] = True
                                            q.append((ny,nx))
                            return visited
                        maskA = flood(sA)
                        maskB = flood(sB)
                        inter = np.logical_and(maskA, maskB).sum()
                        if inter == 0 and maskA.sum() > 0 and maskB.sum() > 0:
                            if maskA.sum() < maskB.sum():
                                new_seg_id = self.segments.max() + 1
                                self.segments[maskA] = new_seg_id
                            else:
                                new_seg_id = self.segments.max() + 1
                                self.segments[maskB] = new_seg_id
                            segments_added += 1
                            found_seed_split = True
                            print(f"  → Seed-split applied at t={t_frac}, delta={delta}")
                            break
                    if found_seed_split:
                        break
            if found_seed_split:
                # Recompute segment counts and keep selection on newly created segment
                self.n_segments = int(self.segments.max())
                self.splitting_segment_id = new_seg_id
                try:
                    self._update_display_with_highlight(self.splitting_segment_id)
                except Exception:
                    pass
            else:
                # If seed attempt failed, attempt to split any touched segments by the unconstrained full line
                touched_seg_ids = np.unique(self.segments[(full_line_mask > 0)])
                touched_seg_ids = [int(x) for x in touched_seg_ids if x != 0]
                touched_success = 0
                for tid in touched_seg_ids:
                    if tid == seg_id:
                        continue
                    t_mask = (self.segments == tid)
                    t_seg_area = int(t_mask.sum())
                    if t_seg_area == 0:
                        continue
                    # Constrain the full_line_mask to this segment region and try split
                    t_line_mask = (full_line_mask & (t_mask.astype(np.uint8) * 255))
                    if t_line_mask.sum() == 0:
                        continue
                    t_seg_without_line = t_mask.copy()
                    t_seg_without_line[t_line_mask > 0] = False
                    t_labeled, t_num = ndimage.label(t_seg_without_line)
                    if t_num < 2:
                        continue
                    # take two largest
                    t_region_sizes = []
                    for region_id in range(1, t_num + 1):
                        t_region_sizes.append((int((t_labeled == region_id).sum()), region_id))
                    t_region_sizes.sort(reverse=True)
                    t_side1_count, t_side1_id = t_region_sizes[0]
                    t_side2_count, t_side2_id = t_region_sizes[1]
                    t_min_side = max(100, int(t_seg_area * 0.01))
                    if t_side1_count >= t_min_side and t_side2_count >= t_min_side:
                        # perform the split for this tid
                        new_seg_id_local = self.segments.max() + 1
                        side2_mask_local = (t_labeled == t_side2_id)
                        self.segments[side2_mask_local] = new_seg_id_local
                        touched_success += 1
                        segments_added += 1
                        print(f"  → Split successful on touched segment {tid}: {t_side1_count} vs {t_side2_count} -> new id {new_seg_id_local}")
                if touched_success == 0:
                    # Try performing a global line (intersection with property boundary) and split all segments it crosses
                    try:
                        p0f = np.array(points[0], dtype=float)
                        p1f = np.array(points[-1], dtype=float)
                        bpoly = self.current_boundary.astype(float)
                        inters = []
                        for i in range(len(bpoly)):
                            A = bpoly[i]
                            B = bpoly[(i+1) % len(bpoly)]
                            s = B - A
                            r = p1f - p0f
                            M = np.column_stack((s, -r))
                            bvec = (p0f - A)
                            try:
                                sol, *_ = np.linalg.lstsq(M, bvec, rcond=None)
                                t = float(sol[0]); u = float(sol[1])
                                if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
                                    X = A + t * s
                                    inters.append((float(X[0]), float(X[1])))
                            except Exception:
                                continue
                        # If polygon intersections not found, fall back to intersecting the image bounding box to get a global cut
                        box_inters = []
                        W = self.clean_image.shape[1]
                        H = self.clean_image.shape[0]
                        # box edges as segment pairs (A,B)
                        box_edges = [ (np.array([0.0,0.0]), np.array([W-1.0,0.0])),
                                      (np.array([W-1.0,0.0]), np.array([W-1.0,H-1.0])),
                                      (np.array([W-1.0,H-1.0]), np.array([0.0,H-1.0])),
                                      (np.array([0.0,H-1.0]), np.array([0.0,0.0])) ]
                        for (A,B) in box_edges:
                            s = B - A
                            r = p1f - p0f
                            M = np.column_stack((s, -r))
                            bvec = (p0f - A)
                            try:
                                sol, *_ = np.linalg.lstsq(M, bvec, rcond=None)
                                t = float(sol[0]); u = float(sol[1])
                                if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
                                    X = A + t * s
                                    box_inters.append((float(X[0]), float(X[1])))
                            except Exception:
                                continue
                        if len(inters) < 2 and len(box_inters) >= 2:
                            inters = box_inters
                        if len(inters) >= 2:
                            # pick outermost two intersections along the line
                            line_dir = p1f - p0f
                            us = [np.dot(np.array(p)-p0f, line_dir)/np.dot(line_dir,line_dir) for p in inters]
                            sidx = np.argsort(us)
                            gA = inters[sidx[0]]
                            gB = inters[sidx[-1]]
                            global_line = np.array([(int(round(gA[0])), int(round(gA[1]))), (int(round(gB[0])), int(round(gB[1])))], dtype=np.int32)
                            global_mask = np.zeros_like(line_mask, dtype=np.uint8)
                            cv2.polylines(global_mask, [global_line], isClosed=False, color=255, thickness=line_thickness)
                            for point in [global_line[0], global_line[-1]]:
                                cv2.circle(global_mask, tuple(point), radius=endpoint_radius, color=255, thickness=-1)                        # For every segment intersected by the global line, attempt split with that constrained mask
                            touched_seg_ids2 = np.unique(self.segments[(global_mask > 0)])
                            touched_seg_ids2 = [int(x) for x in touched_seg_ids2 if x != 0]
                            for tid in touched_seg_ids2:
                                t_mask = (self.segments == tid)
                                t_seg_area = int(t_mask.sum())
                                if t_seg_area == 0:
                                    continue
                                t_line_mask = (global_mask & (t_mask.astype(np.uint8) * 255))
                                if t_line_mask.sum() == 0:
                                    continue
                                t_seg_without_line = t_mask.copy()
                                t_seg_without_line[t_line_mask > 0] = False
                                t_labeled, t_num = ndimage.label(t_seg_without_line)
                                if t_num < 2:
                                    continue
                                t_region_sizes = []
                                for region_id in range(1, t_num + 1):
                                    t_region_sizes.append((int((t_labeled == region_id).sum()), region_id))
                                t_region_sizes.sort(reverse=True)
                                t_side1_count, t_side1_id = t_region_sizes[0]
                                t_side2_count, t_side2_id = t_region_sizes[1]
                                t_min_side = max(100, int(t_seg_area * 0.01))
                                if t_side1_count >= t_min_side and t_side2_count >= t_min_side:
                                    new_seg_id_local = self.segments.max() + 1
                                    side2_mask_local = (t_labeled == t_side2_id)
                                    self.segments[side2_mask_local] = new_seg_id_local
                                    touched_success += 1
                                    segments_added += 1
                                    print(f"  → Split successful on global cut for segment {tid}: {t_side1_count} vs {t_side2_count} -> new id {new_seg_id_local}")
                    except Exception:
                        pass
                    if touched_success == 0:
                        print(f"  → Split failed: line didn't divide segment into 2 parts (found {num_regions} regions)")
                        self.manual_status.config(text=f"Line didn't split segment (only {num_regions} region)")
                        return
        else:
            # Get the two largest regions (in case line created small debris)
            region_sizes = []
            for region_id in range(1, num_regions + 1):
                size = np.sum(labeled_regions == region_id)
                region_sizes.append((size, region_id))
            region_sizes.sort(reverse=True)  # Largest first
            
            side1_id = region_sizes[0][1]
            side2_id = region_sizes[1][1]
            side1_count = region_sizes[0][0]
            side2_count = region_sizes[1][0]
            
            print(f"  Line divides segment into: {side1_count} vs {side2_count} pixels")
            
            # Only split if both sides have meaningful pixels
            min_side_size = max(100, int(seg_area * 0.01))
            if side1_count >= min_side_size and side2_count >= min_side_size:
                # Assign second region to new segment
                side2_mask = (labeled_regions == side2_id)
                self.segments[side2_mask] = new_seg_id
                
                # Add split line visualization to debug image (green for successful split)
                cv2.polylines(debug_img, [points], isClosed=False, color=(0, 255, 0), thickness=line_thickness)
                
                print(f"  → Split successful: kept {side1_count} as seg {seg_id}, moved {side2_count} to seg {new_seg_id}")
                segments_added = 1
                # Keep the larger side selected so user can continue making cuts without re-selecting
                # Default to selecting the newly created segment to allow quick subsequent cuts on the other side
                self.splitting_segment_id = new_seg_id
                self.manual_status.config(text=f"Segment {self.splitting_segment_id} selected - draw another line to split further")
                # Highlight the selected segment immediately
                try:
                    self._update_display_with_highlight(self.splitting_segment_id)
                except Exception:
                    pass
            else:
                # Add failed split visualization (red X's at endpoints)
                cv2.drawMarker(debug_img, tuple(points[0]), (255, 0, 0), cv2.MARKER_TILTED_CROSS, 40, 4)
                cv2.drawMarker(debug_img, tuple(points[-1]), (255, 0, 0), cv2.MARKER_TILTED_CROSS, 40, 4)
                
                print(f"  → Split failed: one side too small (min {min_side_size} pixels)")
                segments_added = 0
                # Keep selection cleared on failure
                self.splitting_segment_id = None
                self.manual_status.config(text="Split failed: one side too small; redraw closer to midline")
        
        # Save debug image
        if self.images_folder:
            # Put debug folder at workspace level (parent of images folder)
            debug_folder = self.images_folder.parent / "debug"
            debug_folder.mkdir(exist_ok=True)
            debug_path = debug_folder / f"split_seg{seg_id}_{self.image_files[self.current_idx].stem}.png"
            cv2.imwrite(str(debug_path), cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))
            print(f"  Debug image saved: {debug_path}")
        
        old_n_segments = self.n_segments
        self.n_segments = self.segments.max()
        print(f"\nSegment count: {old_n_segments} -> {self.n_segments} (added {segments_added})")
        
        # Warn if no segments were actually split
        if segments_added == 0:
            print(f"WARNING: No new segments created - line may be too thin or grazing edges")
            print(f"TIP: Draw line through the middle/widest part of the segment")
        
        print(f"=== END DEBUG ===\n")
        
        # Clear the drawn line(s). Preserve selection if the split succeeded so user can continue.
        keep_sel = None
        if segments_added == 1 and getattr(self, 'splitting_segment_id', None) is not None:
            keep_sel = int(self.splitting_segment_id)
        self._clear_manual_line()
        # Restore selection if appropriate
        if keep_sel is not None:
            self.splitting_segment_id = keep_sel
            try:
                self._update_display_with_highlight(self.splitting_segment_id)
            except Exception:
                pass
        else:
            self.splitting_segment_id = None
        self._reset_submit_button()
        self._update_display()
        self._update_progress()

    # --------------------- New access & road helpers ---------------------
    def _toggle_boundary_access_mode(self):
        """Toggle segment-level boundary access annotation mode."""
        if not self.boundary_approved:
            messagebox.showwarning("Warning", "Approve boundary first to annotate access")
            return
        self.boundary_access_mode = not self.boundary_access_mode
        if self.boundary_access_mode:
            self.boundary_access_btn.config(bg='#90EE90')
            self.ax.set_title("BOUNDARY ACCESS MODE: Click two points on boundary to add access segment")
        else:
            self.boundary_access_btn.config(bg='#FFE4B5')
            self.ax.set_title("LABEL MODE: Click segment to label (right-click to remove)")
        self.canvas.draw()

    def _enforce_boundary_split(self, thickness: int = 1):
        """Ensure the boundary acts as a hard split between segments.
        This will zero out boundary pixels and relabel connected components so that
        any segment that crossed the boundary gets split into separate segments.
        """
        if self.current_boundary is None or self.segments is None:
            return
        h, w = self.segments.shape
        edge_mask_full = np.zeros((h, w), dtype=np.uint8)
        try:
            cv2.polylines(edge_mask_full, [self.current_boundary.astype(np.int32)], isClosed=True, color=255, thickness=thickness)
        except Exception:
            return
        edge_mask = edge_mask_full > 0
        if not edge_mask.any():
            return
        # Remove any pixels on the edge so components are separated
        self.segments[edge_mask] = 0
        # Relabel by processing each original segment independently to avoid merging adjacent segments
        new_segments = np.zeros_like(self.segments, dtype=np.int32)
        new_id = 1
        for old_id in np.unique(self.segments):
            if old_id == 0:
                continue
            mask = (self.segments == old_id)
            # Remove edge pixels from this mask (already zeroed but keep for clarity)
            mask[edge_mask] = False
            if mask.sum() == 0:
                continue
            labeled, num = ndimage.label(mask)
            for comp in range(1, num + 1):
                comp_mask = (labeled == comp)
                if comp_mask.sum() > 0:
                    new_segments[comp_mask] = new_id
                    new_id += 1
        self.segments = new_segments
        self.n_segments = int(self.segments.max()) if self.segments is not None else 0
        # Update status and display
        try:
            self.seg_status.config(text=f"✓ {self.n_segments} segments")
        except Exception:
            pass
        try:
            self._update_display()
        except Exception:
            pass    
    def _toggle_road_mode(self):
        """Toggle the (legacy) road painting mode. Kept for backward compatibility with tests.
        This will toggle an internal flag and update the manual status. Actual painting occurs
        only when the user draws on the canvas (or tests call _paint_road_at directly).
        """
        self.road_mode_active = not getattr(self, 'road_mode_active', False)
        if self.road_mode_active:
            self.manual_status.config(text="Road painting active (use mouse to paint; feature limited in UI)")
            try:
                self.road_mode_btn.config(bg='#90EE90')
            except Exception:
                pass
        else:
            self.manual_status.config(text="Road painting deactivated")
            try:
                self.road_mode_btn.config(bg='#E0FFFF')
            except Exception:
                pass
        self.canvas.draw()

    def _paint_road_at(self, x, y, erase=False):
        """Road painting removed - no-op."""
        return    
    def _set_brush_size(self, size: int):
        self.road_brush_size = size

    def _set_road_paint(self, choice: str):
        """Set the paint choice for road painting: 'public_road' or 'non_road'"""
        if choice in ('public_road', 'non_road'):
            self.road_paint_choice = choice
            try:
                self.road_paint_var.set(choice)
            except Exception:
                pass

    def _on_buffer_mode_change(self):
        self.buffer_mode = self.buffer_mode_var.get()

    def _apply_buffer_value(self):
        val = self.buffer_entry.get()
        try:
            if self.buffer_mode_var.get() == 'px':
                self.road_buffer_px = int(val)
            else:
                self.road_buffer_pct = float(val)
        except Exception:
            messagebox.showwarning("Invalid value", "Enter numeric value for buffer")
            return
        messagebox.showinfo("Buffer updated", f"Buffer set to {val} ({self.buffer_mode_var.get()})")
        # Recompute buffer mask now that buffer parameters changed
        self._compute_buffer_mask()

    def _closest_point_on_boundary(self, x, y):
        """Return closest projected point on boundary and fraction along boundary length."""
        if self.current_boundary is None:
            return None, None
        pts = self.current_boundary
        # compute segment lengths and cumulative
        seg_starts = pts
        seg_ends = np.vstack([pts[1:], pts[0]])
        seg_vecs = seg_ends - seg_starts
        seg_lens = np.linalg.norm(seg_vecs, axis=1)
        cum = np.concatenate([[0], np.cumsum(seg_lens)])
        total = cum[-1]
        best_dist = float('inf')
        best_pt = None
        best_frac = 0.0
        best_seg_idx = 0
        px = np.array([x, y])
        for i, (a, b, v, L) in enumerate(zip(seg_starts, seg_ends, seg_vecs, seg_lens)):
            if L == 0:
                continue
            t = np.dot(px - a, v) / (L * L)
            t_clamped = max(0.0, min(1.0, t))
            proj = a + t_clamped * v
            d = np.linalg.norm(proj - px)
            if d < best_dist:
                best_dist = d
                best_pt = proj
                frac = (cum[i] + L * t_clamped) / total if total > 0 else 0.0
                best_frac = frac % 1.0
                best_seg_idx = i
        return best_pt, best_frac
    
    def _add_boundary_access_click(self, x, y, click_button):
        """Handle click for boundary access annotation."""
        # Snap to boundary and store fraction
        proj, frac = self._closest_point_on_boundary(x, y)
        if proj is None:
            return
        px = (int(round(proj[0])), int(round(proj[1])))
        if self.access_click_start is None:
            self.access_click_start = {'frac': frac, 'pt': px}
            self.manual_status.config(text="First point set. Click second point to complete access segment.")
            self._draw_access_segments()
            return
        else:
            # Right-click during provisional state cancels the provisional point
            if click_button == 3:
                self.access_click_start = None
                self.manual_status.config(text='Access point canceled')
                self._draw_access_segments()
                return

            start_frac = self.access_click_start['frac']
            end_frac = frac
            # Choose shorter arc by default
            d = (end_frac - start_frac) % 1.0
            # If points are effectively the same, treat as accidental and ask user to retry
            if abs(d) < 1e-3 or abs(d - 1.0) < 1e-3:
                self.manual_status.config(text="Points too close or identical - try selecting a different second point.")
                self.access_click_start = None
                self._draw_access_segments()
                return
            if d > 0.5:
                # swap to take shorter arc
                start_frac, end_frac = end_frac, start_frac
            seg = {
                "start_frac": float(start_frac),
                "end_frac": float(end_frac),
                "label": "access_allowed"
            }
            self.boundary_access_segments.append(seg)
            self.access_click_start = None
            self._reset_submit_button()
            self.manual_status.config(text="Access segment added.")
            self._draw_access_segments()
            self._update_display()
            return

    def _draw_access_segments(self):
        """Draw access segments on the current axis (used in both edit and display modes)."""
        if self.current_boundary is None:
            return
        if not hasattr(self, 'access_artists'):
            self.access_artists = []
        # Remove old artists
        for a in getattr(self, 'access_artists', []):
            try:
                a.remove()
            except:
                pass
        self.access_artists = []
        pts = self.current_boundary
        # Build cumulative lengths
        seg_starts = pts
        seg_ends = np.vstack([pts[1:], pts[0]])
        seg_vecs = seg_ends - seg_starts
        seg_lens = np.linalg.norm(seg_vecs, axis=1)
        cum = np.concatenate([[0], np.cumsum(seg_lens)])
        total = cum[-1]
        def frac_to_point(frac):
            f = (frac % 1.0) * total
            # find segment containing f
            idx = np.searchsorted(cum, f, side='right') - 1
            idx = max(0, min(idx, len(seg_lens)-1))
            local_f = (f - cum[idx]) / (seg_lens[idx] if seg_lens[idx]>0 else 1e-6)
            pt = seg_starts[idx] + seg_vecs[idx] * local_f
            return pt
        for seg in self.boundary_access_segments:
            s, e = seg['start_frac'], seg['end_frac']
            # Skip full default 'no_access' to avoid a thick gold overlay for the entire boundary
            if seg.get('label') == 'no_access':
                # Compute segment length properly
                if e >= s:
                    seg_len = e - s
                else:
                    seg_len = (1.0 - s) + e
                if seg_len >= 0.99:
                    continue
            # sample along shorter arc
            samples = np.linspace(s, e, num=50) if s <= e else np.linspace(s, e+1, num=50)
            pts_samples = np.array([frac_to_point(ss % 1.0) for ss in samples])
            if seg.get('label') == 'access_allowed':
                color = '#00FF00'
                lw = 3
                style = {'linewidth': lw, 'solid_capstyle': 'round'}
            else:
                # Subtle rendering for explicit 'no_access' segments (non-full)
                color = '#FFD700'
                lw = 1.0
                style = {'linewidth': lw, 'linestyle': '--'}
            artist, = self.ax.plot(pts_samples[:,0], pts_samples[:,1], color=color, **style)
            self.access_artists.append(artist)
        # Draw provisional first-click point
        if self.access_click_start is not None:
            pt = self.access_click_start['pt']
            artist = self.ax.scatter([pt[0]], [pt[1]], c='white', s=60, edgecolor='black', zorder=20)
            self.access_artists.append(artist)
        self.canvas.draw()

    def _update_boundary_artists(self):
        """Fast update of existing boundary polygon only (vertex handles removed).
        If the polygon artist doesn't exist yet, create it via _draw_editable_boundary().
        """
        # If blit is available, try to blit the polygon update only
        if getattr(self, '_use_blit', False) and getattr(self, '_bg', None) is not None and getattr(self, '_renderer', None) is not None:
            try:
                self.canvas.restore_region(self._bg)
                try:
                    self._boundary_poly_artist.set_xy(self.current_boundary)
                except Exception:
                    try:
                        self._boundary_poly_artist.remove()
                    except Exception:
                        pass
                    self._boundary_poly_artist = mpatches.Polygon(self.current_boundary, fill=False, edgecolor='yellow', linewidth=1)
                    self.ax.add_patch(self._boundary_poly_artist)
                try:
                    self._boundary_poly_artist.draw(self._renderer)
                except Exception:
                    pass
                try:
                    self.canvas.blit(self.ax.bbox)
                except Exception:
                    self.canvas.draw_idle()
                return
            except Exception:
                # Fall through to non-blit update
                pass

        # Non-blit update: update or recreate polygon artist and schedule a coalesced draw
        try:
            self._boundary_poly_artist.set_xy(self.current_boundary)
        except Exception:
            try:
                self._boundary_poly_artist.remove()
            except Exception:
                pass
            self._boundary_poly_artist = mpatches.Polygon(self.current_boundary, fill=False, edgecolor='yellow', linewidth=1)
            self.ax.add_patch(self._boundary_poly_artist)
        # Use coalesced draw to avoid frequent heavy GUI redraws
        try:
            if self._profile_drag:
                t_draw0 = time.time()
            self._schedule_coalesced_draw()
            if self._profile_drag:
                t_draw1 = time.time(); print(f"PROFILE: schedule_draw {t_draw1-t_draw0:.4f}s")
        except Exception:
            if self._profile_drag:
                t_draw0 = time.time()
            self.canvas.draw()
            if self._profile_drag:
                t_draw1 = time.time(); print(f"PROFILE: canvas_draw {t_draw1-t_draw0:.4f}s")

    def _compute_buffer_mask(self):
        """Compute buffer mask (bool) outside the property boundary within buffer distance."""
        if self.current_boundary is None:
            self.buffer_mask = None
            return
        h, w = self.clean_image.shape[:2]
        # Create ROI mask for interior (filled polygon)
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(roi_mask, [self.current_boundary.astype(np.int32)], 255)
        # Create an edge mask of the boundary and compute distance transform outside
        edge_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.polylines(edge_mask, [self.current_boundary.astype(np.int32)], isClosed=True, color=255, thickness=1)
        invert = (edge_mask == 0).astype(np.uint8) * 255
        dist = cv2.distanceTransform(invert, cv2.DIST_L2, 5)
        if self.buffer_mode_var.get() == 'px':
            thresh = self.road_buffer_px
        else:
            val = self.road_buffer_pct
            thresh = min(h, w) * val
        buffer_mask = (dist <= thresh) & (roi_mask == 0)
        # Store as boolean mask
        self.buffer_mask = buffer_mask.astype(bool)

    def _schedule_coalesced_draw(self):
        """Schedule a single draw after a short delay to coalesce frequent updates."""
        if self._pending_draw:
            return
        self._pending_draw = True
        try:
            self.root.after(self._coalesced_draw_delay_ms, self._do_coalesced_draw)
        except Exception:
            # Fall back to immediate draw if scheduling fails
            self._do_coalesced_draw()

    def _do_coalesced_draw(self):
        """Perform the scheduled draw and clear the pending flag."""
        self._pending_draw = False
        try:
            self.canvas.draw()
        except Exception:
            try:
                self.canvas.draw_idle()
            except Exception:
                pass


    def _paint_road_at(self, x, y, erase=False):
        """Paint or erase circular brush at (x,y) constrained to buffer_mask outside boundary.
        Values: 0 = unlabeled/ignore (erased), 1 = public_road, 2 = non_road (negative)
        """
        if self.buffer_mask is None:
            self._compute_buffer_mask()
        if self.buffer_mask is None:
            return
        h, w = self.buffer_mask.shape
        r = int(self.road_brush_size // 2)
        x0, x1 = max(0, x - r), min(w, x + r + 1)
        y0, y1 = max(0, y - r), min(h, y + r + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask = (xx - x)**2 + (yy - y)**2 <= r*r
        region_allowed = self.buffer_mask[y0:y1, x0:x1]
        final_mask = mask & region_allowed
        if final_mask.sum() == 0:
            return
        if self.road_mask is None or self.road_mask.shape != (h, w):
            self.road_mask = np.zeros((h, w), dtype=np.uint8)
        if erase:
            self.road_mask[y0:y1, x0:x1][final_mask] = 0
        else:
            # determine paint value from current selection
            paint_choice = getattr(self, 'road_paint_var', None)
            if paint_choice is None:
                paint = 1
            else:
                pc = self.road_paint_var.get()
                paint = 1 if pc == 'public_road' else 2
            self.road_mask[y0:y1, x0:x1][final_mask] = paint
        self._reset_submit_button()
        self._update_display()
        
    def _draw_editable_boundary(self):
        """Draw boundary polygon (vertex dragging disabled)."""
        if self.current_boundary is None:
            return
        
        # Don't try to remove artists - just clear the list
        # The ax.clear() in redraw will handle removing them
        self.boundary_vertices_artists.clear()
        
        # Draw only the polygon (vertex dragging disabled)
        poly = mpatches.Polygon(self.current_boundary, fill=False, edgecolor='yellow', linewidth=1, joinstyle='round')
        # Remove any previous polygon artist if present
        try:
            if getattr(self, '_boundary_poly_artist', None) is not None:
                try:
                    self._boundary_poly_artist.remove()
                except Exception:
                    pass
        except Exception:
            pass
        self._boundary_poly_artist = poly
        self.ax.add_patch(poly)
        # Avoid drawing access overlays while user is actively dragging the whole boundary
        if not getattr(self, 'dragging_boundary', False):
            self._draw_access_segments() 
    
    def _on_click(self, event):
        """Handle click events for boundary dragging, manual splitting, or segment labeling."""
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        
        # ONLY check for whole-boundary dragging if boundary is NOT approved
        if self.current_boundary is not None and not self.boundary_approved:
            from matplotlib.path import Path
            poly_path = Path(self.current_boundary)
            if poly_path.contains_point((event.xdata, event.ydata)):
                # Start whole-boundary drag (translate/rotate only)
                self.dragging_boundary = True
                self.drag_start_pos = (event.xdata, event.ydata)
                self.drag_start_boundary = self.current_boundary.copy()
                return

        # === Boundary Access Mode (segment-level) ===
        # Treat Access mode (mode == 'access') as shorthand for boundary access editing.
        if (self.boundary_access_mode or self.mode == 'access') and self.boundary_approved:
            # If we were in mode 'access' but not yet 'boundary_access_mode', enable it visually
            if not self.boundary_access_mode:
                self.boundary_access_mode = True
                try:
                    self.boundary_access_btn.config(bg='#90EE90')
                except Exception:
                    pass
            # Prefer toggling existing segment if clicking on one
            proj, frac = self._closest_point_on_boundary(event.xdata, event.ydata)
            if proj is not None:
                # If a provisional first-click exists, treat any click as a completion (or cancel)
                if self.access_click_start is not None and event.button in (1, 3):
                    self._add_boundary_access_click(event.xdata, event.ydata, event.button)
                    return
                # Iterate segments from shortest to longest so that clicks hit small segments before the default full-boundary segment
                seg_indices = sorted(range(len(self.boundary_access_segments)),
                                     key=lambda idx: ((self.boundary_access_segments[idx]['end_frac'] - self.boundary_access_segments[idx]['start_frac']) % 1.0))
                for i in seg_indices:
                    seg = self.boundary_access_segments[i]
                    s, e = seg['start_frac'], seg['end_frac']
                    inside = (s <= e and frac >= s and frac <= e) or (s > e and (frac >= s or frac <= e))
                    if inside:
                        d = np.linalg.norm(proj - np.array([event.xdata, event.ydata]))
                        if d < 10:
                            if event.button == 1:
                                # Compute actual segment length along the closed boundary (handle wrap-around correctly)
                                if e >= s:
                                    seg_len = e - s
                                else:
                                    seg_len = (1.0 - s) + e
                                # If this segment effectively covers the full boundary and is 'no_access',
                                # start a provisional first-click here instead of toggling the whole boundary
                                if seg_len >= 0.99 and seg.get('label') == 'no_access' and self.access_click_start is None:
                                    # Start provisional access click here instead of toggling the whole boundary
                                    self.access_click_start = {'frac': frac, 'pt': (int(round(proj[0])), int(round(proj[1])))}
                                    self.manual_status.config(text="First point set. Click second point to complete access segment.")
                                    self._draw_access_segments()
                                    return
                                # Otherwise toggle the clicked segment's label
                                seg['label'] = 'no_access' if seg.get('label') == 'access_allowed' else 'access_allowed'
                                self._reset_submit_button()
                                self._draw_access_segments()
                                self._update_display()
                                return
                            elif event.button == 3:
                                del self.boundary_access_segments[i]
                                self._reset_submit_button()
                                self._draw_access_segments()
                                self._update_display()
                                return
            # Otherwise treat as adding/completing an access segment
            if event.button in (1, 3):
                self._add_boundary_access_click(event.xdata, event.ydata, event.button)
            return

        # Manual mode - add point to line OR select segment to highlight
        if self.manual_mode and self.boundary_approved:
            h, w = self.segments.shape
            # Clamp coordinates to valid image bounds
            x_int = int(np.clip(event.xdata, 0, w - 1))
            y_int = int(np.clip(event.ydata, 0, h - 1))
            
            clicked_seg = self.segments[y_int, x_int] if (0 <= y_int < h and 0 <= x_int < w) else 0
            
            # If no segment selected yet, first click highlights the segment they want to split
            if self.splitting_segment_id is None:
                if clicked_seg > 0:
                    # Highlight this segment
                    self.manual_status.config(text=f"Segment {clicked_seg} selected - now draw line to split it")
                    # Store which segment we're splitting
                    self.splitting_segment_id = clicked_seg
                    # Redraw with highlight
                    self._update_display_with_highlight(clicked_seg)
                    return
            else:
                # Segment already selected - check if clicking far outside it (>30px from segment boundary)
                if clicked_seg != self.splitting_segment_id:
                    # Calculate distance from click to current segment
                    current_seg_mask = self.segments == self.splitting_segment_id
                    # Get segment boundary pixels
                    seg_pad = np.pad(current_seg_mask, 1, mode='constant', constant_values=False)
                    edge_h = (seg_pad[:-2, 1:-1] != seg_pad[2:, 1:-1])
                    edge_v = (seg_pad[1:-1, :-2] != seg_pad[1:-1, 2:])
                    boundary_mask = edge_h | edge_v
                    boundary_coords = np.argwhere(boundary_mask)  # Returns [y, x] pairs
                    
                    if len(boundary_coords) > 0:
                        # Calculate distance from click to nearest boundary pixel
                        distances = np.sqrt((boundary_coords[:, 1] - x_int)**2 + (boundary_coords[:, 0] - y_int)**2)
                        min_distance = distances.min()
                        
                        # If clicking >30 pixels from segment boundary
                        if min_distance > 30:
                            self._clear_manual_line()
                            if clicked_seg > 0:
                                # Switch to new segment
                                print(f"Click is {min_distance:.1f}px from segment {self.splitting_segment_id} - switching to segment {clicked_seg}")
                                self.splitting_segment_id = clicked_seg
                                self.manual_status.config(text=f"Segment {clicked_seg} selected - now draw line to split it")
                                self._update_display_with_highlight(clicked_seg)
                            else:
                                # Clicked outside boundary - deselect
                                print(f"Click is {min_distance:.1f}px from segment {self.splitting_segment_id} and outside boundary - deselecting")
                                self.splitting_segment_id = None
                                self.manual_status.config(text="Segment deselected - click a segment to select it")
                                self._update_display()
                            return
            
            # Add point to split line (segment already selected, clicking within or near it)
            if self.splitting_segment_id is not None:
                self.manual_line_points.append((x_int, y_int))
                
                # Update status
                seg_id = self.splitting_segment_id
                num_lines = len(self.manual_polylines)
                line_num = num_lines + 1
                self.manual_status.config(text=f"Segment {seg_id} Line {line_num}: {len(self.manual_line_points)} points (Spacebar=new line, Enter=apply)")
                
                # Redraw with all polylines
                self._update_display_with_highlight(self.splitting_segment_id)
            return
        
        # Label segment on click (after boundary is approved)
        if self.segments is not None and self.boundary_approved and not self.manual_mode:
            h, w = self.segments.shape
            x_int = int(np.clip(event.xdata, 0, w - 1))
            y_int = int(np.clip(event.ydata, 0, h - 1))
            
            if 0 <= y_int < h and 0 <= x_int < w:
                segment_id = self.segments[y_int, x_int]

                # In 'Segments' mode, support both labeling and splitting
                if self.mode == 'segments' and segment_id > 0:
                    seg_id_int = int(segment_id)
                    if self.manual_mode:
                        # Select for splitting (existing behavior)
                        self.splitting_segment_id = seg_id_int
                        try:
                            self.manual_btn.config(text="Split Mode (On)", bg='#90EE90')
                        except Exception:
                            pass
                        self.finalize_btn.config(state=tk.NORMAL)
                        self.manual_status.config(text=f"Segment {seg_id_int} selected - draw lines to split")
                        self._update_display_with_highlight(seg_id_int)
                        return
                    else:
                        # Treat like labeling: left-click to label, right-click to remove
                        if event.button == 3:
                            if seg_id_int in self.segment_labels:
                                del self.segment_labels[seg_id_int]
                                self._reset_submit_button()
                                self._update_display()
                                self._update_progress()
                        elif event.button == 1:
                            selected_class = self.class_var.get()
                            self.segment_labels[seg_id_int] = selected_class
                            self._reset_submit_button()
                            self._update_display()
                            self._update_progress()
                            # Visual feedback
                            self.last_clicked_segment = seg_id_int
                            self.root.after(300, lambda: setattr(self, 'last_clicked_segment', None))
                        return

                if segment_id > 0:
                    # Right-click to remove label (undo)
                    if event.button == 3:
                        seg_id_int = int(segment_id)
                        if seg_id_int in self.segment_labels:
                            print(f"Removing label from segment {seg_id_int}")
                            del self.segment_labels[seg_id_int]
                            self._reset_submit_button()
                            self._update_display()
                            self._update_progress()
                        else:
                            print(f"Segment {seg_id_int} has no label to remove")
                    # Left-click to add label
                    elif event.button == 1 and self.mode == 'label':
                        selected_class = self.class_var.get()
                        seg_id_int = int(segment_id)

                        # Visual feedback: briefly highlight the segment being labeled
                        self.last_clicked_segment = seg_id_int

                        print(f"DEBUG: Clicked segment {seg_id_int}, selected class: '{selected_class}'")
                        print(f"DEBUG: Before - segment_labels = {self.segment_labels}")
                        self.segment_labels[seg_id_int] = selected_class
                        print(f"DEBUG: After - segment_labels = {self.segment_labels}")
                        print(f"DEBUG: Total labeled segments: {len(self.segment_labels)}")
                        self._reset_submit_button()
                        self._update_display()
                        self._update_progress()

                        # Clear last clicked after a moment (for visual feedback)
                        self.root.after(300, lambda: setattr(self, 'last_clicked_segment', None))
                else:
                    # Clicked background/boundary - provide feedback
                    print(f"DEBUG: Clicked background (segment ID = 0), ignoring")
                    self.ax.set_title("LABEL MODE: Click a segment to label (not background)")
                    self.canvas.draw()
                    self.root.after(2000, lambda: self.ax.set_title("LABEL MODE: Click segment to label (right-click to remove)") or self.canvas.draw())
        else:
            # Debug output when labeling doesn't work - provide detailed feedback
            print(f"DEBUG: Labeling blocked - segments={self.segments is not None}, boundary_approved={self.boundary_approved}, manual_mode={self.manual_mode}")
            if not self.boundary_approved:
                print(f"DEBUG: Cannot label - boundary not approved. Click '✓ Good' button first!")
                self.ax.set_title("Click '✓ Good' button to approve boundary and generate segments")
                self.canvas.draw()
                self.root.after(3000, lambda: self._update_title_for_mode())
            elif self.manual_mode:
                print(f"DEBUG: Cannot label - still in split mode. Switch to label mode first!")
                self.ax.set_title("Click 'Click to switch to Label mode' button first")
                self.canvas.draw()
                self.root.after(3000, lambda: self._update_title_for_mode())
            elif self.segments is None:
                print(f"DEBUG: Cannot label - segments is None. Generate segments first!")
                self.ax.set_title("Click '✓ Good' button to generate segments")
                self.canvas.draw()
                self.root.after(3000, lambda: self._update_title_for_mode())
    
    def _on_motion(self, event):
        """Handle mouse motion for whole-boundary dragging (translate/rotate)."""
        if event.xdata is None or event.ydata is None:
            return
        # Painting while moving (continuous stroke support)
        if getattr(self, 'painting_road', False) and self.road_mode_active:
            if self.clean_image is None:
                return
            x_int = int(np.clip(event.xdata, 0, self.clean_image.shape[1] - 1))
            y_int = int(np.clip(event.ydata, 0, self.clean_image.shape[0] - 1))
            self._paint_road_at(x_int, y_int, erase=getattr(self, 'paint_erase_mode', False))
            return
        

        
        # Handle whole-boundary dragging
        if not self.dragging_boundary or self.drag_start_pos is None or self.drag_start_boundary is None:
            return
        
        print(f"Dragging: current pos ({event.xdata:.1f}, {event.ydata:.1f})")  # Debug
        
        # Calculate drag offset with reduced sensitivity (6.25% of actual drag distance)
        DRAG_SENSITIVITY = 0.0625
        dx = (event.xdata - self.drag_start_pos[0]) * DRAG_SENSITIVITY
        dy = (event.ydata - self.drag_start_pos[1]) * DRAG_SENSITIVITY
        
        # Check for Ctrl key for rotation
        if event.key == 'control':
            # Rotate based on horizontal drag distance (reduced sensitivity)
            theta_deg = self.boundary_theta + dx * 0.5 * DRAG_SENSITIVITY  # 0.5 degrees per pixel
            if self._profile_drag:
                t0 = time.time()
            self._apply_boundary_transform(self.boundary_dx, self.boundary_dy, theta_deg)
            if self._profile_drag:
                t1 = time.time(); print(f"PROFILE: apply_transform {t1-t0:.4f}s")
        else:
            # Translate
            total_dx = self.boundary_dx + dx
            total_dy = self.boundary_dy + dy
            if self._profile_drag:
                t0 = time.time()
            self._apply_boundary_transform(total_dx, total_dy, self.boundary_theta)
            if self._profile_drag:
                t1 = time.time(); print(f"PROFILE: apply_transform {t1-t0:.4f}s")
        
        # Throttle redraws to avoid flooding redraw pipeline during fast drags
        now = time.time()
        if self._profile_drag or (now - self._last_drag_render_time >= self._drag_render_interval):
            self._last_drag_render_time = now
            try:
                if self._profile_drag:
                    t0 = time.time()
                self._update_boundary_artists()
                if self._profile_drag:
                    t1 = time.time(); print(f"PROFILE: update_artists {t1-t0:.4f}s")
            except Exception:
                # Fallback to full redraw if artist update fails
                if self.enhanced_image is not None:
                    rgb = self.enhanced_image
                else:
                    rgb = cv2.cvtColor(self.clean_image, cv2.COLOR_BGR2RGB)
                self.ax.clear()
                self.ax.imshow(rgb)
                self._draw_editable_boundary()
                self.ax.set_title("Adjust boundary (arrows=move, </>=rotate) and press Enter to proceed")
                self.ax.axis('off')
                self.canvas.draw()
    
    def _on_release(self, event):
        """Handle mouse release to finish whole-boundary drag (translate/rotate)."""

        
        # Reset whole-boundary dragging
        if self.dragging_boundary:
            self.dragging_boundary = False
            # When drag finishes, record final offsets
            if self.drag_start_pos is not None and event.xdata is not None and event.ydata is not None:
                dx = event.xdata - self.drag_start_pos[0]
                dy = event.ydata - self.drag_start_pos[1]
                
                if event.key == 'control':
                    # Rotation mode
                    self.boundary_theta += dx * 0.5
                else:
                    # Translation mode
                    self.boundary_dx += dx
                    self.boundary_dy += dy
            
            self.drag_start_pos = None
            self.drag_start_boundary = None
            # Ensure overlays (access segments) are redrawn after a drag completes
            if self.current_boundary is not None:
                # Force a redraw including access overlays
                if self.enhanced_image is not None:
                    rgb = self.enhanced_image
                else:
                    rgb = cv2.cvtColor(self.clean_image, cv2.COLOR_BGR2RGB)
                self.ax.clear()
                self.ax.imshow(rgb)
                # draw boundary and access overlays explicitly
                self._draw_editable_boundary()
                self._draw_access_segments()
                self.ax.set_title("Adjust boundary (arrows=move, </>=rotate) and press Enter to proceed")
                self.ax.axis('off')
                self.canvas.draw()
        # End road painting if active
        if getattr(self, 'painting_road', False):
            self.painting_road = False
            self.paint_erase_mode = False
    
    def _on_key_press(self, event):
        """Handle key presses for boundary adjustment and manual mode."""
        # Manual mode - Enter to apply, Spacebar to finish current line, Escape to cancel
        if self.manual_mode:
            # Allow 'm' to toggle out of manual mode quickly
            if event.key == 'm':
                self._toggle_manual_mode()
                return
            if event.key == 'enter':
                self._finalize_manual_split()
            elif event.key == ' ':  # Spacebar - finish current line and start new one
                if len(self.manual_line_points) >= 2:
                    self.manual_polylines.append(self.manual_line_points.copy())
                    self.manual_line_points.clear()
                    num_lines = len(self.manual_polylines)
                    self.manual_status.config(text=f"Line {num_lines} complete. Drawing line {num_lines + 1}...")
                    self._update_display_with_highlight(self.splitting_segment_id)
                else:
                    self.manual_status.config(text="Need at least 2 points to finish line!")
            elif event.key == 'escape':
                was_selected = self.splitting_segment_id is not None
                self._clear_manual_line()
                if was_selected:
                    self.manual_status.config(text="Segment deselected - click a segment to select it")
                else:
                    self.manual_status.config(text="Cancelled")
                self._update_display()
            return

        # --- Global shortcuts (work when boundary exists) ---
        key = event.key.lower() if event.key else None
        if key in ('m','b','r','n','p','s','escape'):
            # Toggle split/manual mode
            if key == 'm' and self.boundary_approved:
                self._toggle_manual_mode()
                return
            # Toggle boundary access
            if key == 'b' and self.boundary_approved:
                self._toggle_boundary_access_mode()
                return
            # Toggle road mode
            if key == 'r' and self.boundary_approved:
                self._toggle_road_mode()
                return
            # Next / previous images
            if key == 'n':
                self._next_image()
                return
            if key == 'p':
                self._prev_image()
                return
            # Submit / save
            if key == 's':
                self._submit_annotation()
                return
            # Toggle drag profiling (for debugging)
            if key == 't':
                self._profile_drag = not self._profile_drag
                self.manual_status.config(text=f"Profiling during drag: {self._profile_drag}")
                print(f"Profiling during drag: {self._profile_drag}")
                return
            # Escape: cancel in-progress boundary access or road painting
            if key == 'escape':
                if getattr(self, 'access_click_start', None) is not None:
                    self.access_click_start = None
                    self.manual_status.config(text='Access point canceled')
                    self._draw_access_segments()
                    return
                if getattr(self, 'painting_road', False):
                    self.painting_road = False
                    self.paint_erase_mode = False
                    self.manual_status.config(text='Road painting canceled')
                    return
        
        if self.current_boundary is None:
            return
        
        # Enter to approve boundary (like clicking Good button)
        if event.key == 'enter':
            # Approve boundary if not yet approved, then switch to Segments mode
            if not self.boundary_approved:
                self._approve_boundary()
            try:
                self._set_mode('segments')
            except Exception:
                pass
            return
        
        if self.boundary_approved:
            return
        
        # Rotation keys (< and > or , and .)
        if event.key in ['<', ',']:  # Rotate counter-clockwise
            self.boundary_theta -= 0.5  # 0.5 degrees
            self._apply_boundary_transform(self.boundary_dx, self.boundary_dy, self.boundary_theta)
            self._redraw_boundary()
        elif event.key in ['>', '.']:  # Rotate clockwise
            self.boundary_theta += 0.5  # 0.5 degrees
            self._apply_boundary_transform(self.boundary_dx, self.boundary_dy, self.boundary_theta)
            self._redraw_boundary()
        # Arrow key movement (1 pixel at a time)
        elif event.key == 'up':
            self.boundary_dy -= 1
            self._apply_boundary_transform(self.boundary_dx, self.boundary_dy, self.boundary_theta)
            self._redraw_boundary()
        elif event.key == 'down':
            self.boundary_dy += 1
            self._apply_boundary_transform(self.boundary_dx, self.boundary_dy, self.boundary_theta)
            self._redraw_boundary()
        elif event.key == 'left':
            self.boundary_dx -= 1
            self._apply_boundary_transform(self.boundary_dx, self.boundary_dy, self.boundary_theta)
            self._redraw_boundary()
        elif event.key == 'right':
            self.boundary_dx += 1
            self._apply_boundary_transform(self.boundary_dx, self.boundary_dy, self.boundary_theta)
            self._redraw_boundary()
    
    def _redraw_boundary(self):
        """Redraw boundary with enhanced contrast."""
        if self.clean_image is None:
            return
        
        # Create enhanced version if not cached
        if self.enhanced_image is None:
            lab = cv2.cvtColor(self.clean_image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l)
            lab_enhanced = cv2.merge([l_enhanced, a, b])
            self.enhanced_image = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
        rgb = self.enhanced_image
        
        self.ax.clear()
        self.ax.imshow(rgb)
        self._draw_editable_boundary()
        self.ax.set_title("Adjust boundary (arrows=move, </>=rotate) and press Enter to proceed")
        self.ax.axis('off')
        self.canvas.draw()
    
    def _update_display(self):
        """Update the display with enhanced contrast for better visibility."""
        if self.segments is None:
            return
        
        # Create enhanced version if not cached
        if self.enhanced_image is None:
            lab = cv2.cvtColor(self.clean_image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l)
            lab_enhanced = cv2.merge([l_enhanced, a, b])
            self.enhanced_image = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
        rgb = self.enhanced_image
        
        h, w = rgb.shape[:2]
        
        # Start with original image (keep colors vivid!)
        display = rgb.copy().astype(float) / 255
        
        # Fill labeled segments with semi-transparent class color
        alpha = 0.4  # Blend factor
        print(f"DEBUG _update_display: segment_labels = {self.segment_labels}")
        for seg_id, class_name in self.segment_labels.items():
            print(f"DEBUG: Rendering segment {seg_id} as {class_name}")
            seg_mask = self.segments == seg_id
            pixels_in_seg = seg_mask.sum()
            print(f"DEBUG: Segment {seg_id} has {pixels_in_seg} pixels")
            if pixels_in_seg < 50:  # Skip tiny segments
                print(f"DEBUG: Skipping segment {seg_id} - too small")
                continue
            color = CLASS_COLORS.get(class_name, [1, 1, 1])
            print(f"DEBUG: Applying color {color} to segment {seg_id}")
            display[seg_mask] = display[seg_mask] * (1 - alpha) + np.array(color) * alpha
        
        # Find all segment boundaries
        seg_pad = np.pad(self.segments, 1, mode='edge')
        edge_h = (seg_pad[:-2, 1:-1] != seg_pad[2:, 1:-1])
        edge_v = (seg_pad[1:-1, :-2] != seg_pad[1:-1, 2:])
        all_edges = edge_h | edge_v
        
        # Draw segment boundaries in white for clarity; keep property boundary as highlighted polygon
        display[all_edges] = [1, 1, 1]  # White



        self.ax.clear()
        self.ax.imshow(display, extent=[0, w, h, 0], aspect='equal')
        
        # Add segment ID numbers to help with manual splitting
        for seg_id in range(1, min(self.n_segments + 1, 200)):  # Limit to first 200 segments to avoid clutter
            seg_mask = self.segments == seg_id
            if seg_mask.sum() >= getattr(self, 'min_segment_px', 1000):  # Show segments larger than min_segment_px
                coords = np.argwhere(seg_mask)
                if len(coords) > 0:
                    centroid_y, centroid_x = coords.mean(axis=0)
                    self.ax.text(centroid_x, centroid_y, str(seg_id), 
                               color='white', fontsize=7, ha='center', va='center',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5, edgecolor='none'))
        
        # Add boundary
        if self.current_boundary is not None:
            poly = mpatches.Polygon(self.current_boundary, fill=False,
                                   edgecolor='yellow', linewidth=1, joinstyle='round')
            self.ax.add_patch(poly)
        
        # Re-draw manual lines if in manual mode
        if self.manual_mode:
            # Draw completed polylines in green
            for polyline in self.manual_polylines:
                points = np.array(polyline)
                artist, = self.ax.plot(points[:, 0], points[:, 1], 'g-', linewidth=2, marker='o', markersize=4)
                self.manual_line_artists.append(artist)
            
            # Draw current polyline being drawn in red
            if len(self.manual_line_points) > 0:
                points = np.array(self.manual_line_points)
                if len(points) == 1:
                    artist, = self.ax.plot(points[:, 0], points[:, 1], 'ro', markersize=5)
                else:
                    artist, = self.ax.plot(points[:, 0], points[:, 1], 'r-', linewidth=2, marker='o', markersize=4)
                self.manual_line_artists.append(artist)
        
        # Set title based on current mode
        if self.boundary_access_mode or self.mode == 'access':
            self.ax.set_title("BOUNDARY ACCESS MODE: Click two points on boundary to add access segment")
        elif self.manual_mode:
            self.ax.set_title("SPLIT MODE: Click segment, draw lines (Esc=reselect, Space=new line, Enter=apply)")
        else:
            self.ax.set_title("LABEL MODE: Click segment to label (right-click to remove)")
        self.ax.axis('off')
        # Draw boundary access overlays (if any)
        self._draw_access_segments()
        self.canvas.draw()
    
    def _update_display_with_highlight(self, highlight_seg_id):
        """Update display with a specific segment highlighted."""
        if self.segments is None:
            return
        
        # Create enhanced version if not cached
        if self.enhanced_image is None:
            lab = cv2.cvtColor(self.clean_image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l)
            lab_enhanced = cv2.merge([l_enhanced, a, b])
            self.enhanced_image = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
        rgb = self.enhanced_image
        
        h, w = rgb.shape[:2]
        
        # Start with dimmed image
        display = rgb.copy().astype(float) / 255 * 0.5  # Dim everything
        
        # Highlight selected segment at full brightness
        highlight_mask = self.segments == highlight_seg_id
        display[highlight_mask] = (rgb.astype(float) / 255)[highlight_mask]  # Full brightness
        
        # Add bright border around highlighted segment
        seg_pad = np.pad(self.segments, 1, mode='constant', constant_values=0)
        edge_h = (seg_pad[:-2, 1:-1] == highlight_seg_id) != (seg_pad[2:, 1:-1] == highlight_seg_id)
        edge_v = (seg_pad[1:-1, :-2] == highlight_seg_id) != (seg_pad[1:-1, 2:] == highlight_seg_id)
        edges = edge_h | edge_v
        display[edges] = [1, 1, 0]  # Yellow border
        
        self.ax.clear()
        self.ax.imshow(display, extent=[0, w, h, 0], aspect='equal')
        
        # Add boundary
        if self.current_boundary is not None:
            poly = mpatches.Polygon(self.current_boundary, fill=False,
                                   edgecolor='yellow', linewidth=1.5, joinstyle='round')
            self.ax.add_patch(poly)
        
        # Draw all manual polylines
        # Draw completed polylines in green
        for polyline in self.manual_polylines:
            points = np.array(polyline)
            artist, = self.ax.plot(points[:, 0], points[:, 1], 'g-', linewidth=2, marker='o', markersize=4)
            self.manual_line_artists.append(artist)
        
        # Draw current polyline being drawn in red
        if len(self.manual_line_points) > 0:
            points = np.array(self.manual_line_points)
            if len(points) == 1:
                artist, = self.ax.plot(points[:, 0], points[:, 1], 'ro', markersize=5)
            else:
                artist, = self.ax.plot(points[:, 0], points[:, 1], 'r-', linewidth=2, marker='o', markersize=4)
            self.manual_line_artists.append(artist)
        
        seg_id_text = f"Line {len(self.manual_polylines) + 1}" if len(self.manual_line_points) > 0 or len(self.manual_polylines) > 0 else ""
        # Respect access mode when highlighting a segment
        if self.boundary_access_mode or self.mode == 'access':
            self.ax.set_title(f"BOUNDARY ACCESS MODE: Segment {highlight_seg_id} highlighted - click boundary to add access")
        else:
            self.ax.set_title(f"SPLIT MODE: Segment {highlight_seg_id} highlighted - draw lines to split {seg_id_text}")
        self.ax.axis('off')
        self.canvas.draw()
        
    def _update_progress(self):
        """Update progress label."""
        if self.segments is None:
            return
        
        total_pixels = 0
        labeled_pixels = 0
        small_segments = 0
        
        for seg_id in range(1, self.n_segments + 1):
            area = np.sum(self.segments == seg_id)
            if area >= getattr(self, 'min_segment_px', 1000):
                total_pixels += area
                if seg_id in self.segment_labels:
                    labeled_pixels += area
            else:
                small_segments += 1
        
        if total_pixels > 0:
            percent = 100.0 * labeled_pixels / total_pixels
            self.labeling_progress.config(
                text=f"Labeled: {percent:.1f}% of pixels"
            )
        else:
            self.labeling_progress.config(
                text=f"Labeled: 0% of pixels"
            )
        
    def _auto_fill_unlabeled_segments(self):
        """Auto-fill unlabeled segments using nearest neighbor approach."""
        if self.segments is None:
            return
        
        # Find unlabeled segments
        unlabeled_ids = []
        for seg_id in range(1, self.n_segments + 1):
            area = np.sum(self.segments == seg_id)
            if area >= getattr(self, 'min_segment_px', 100) and seg_id not in self.segment_labels:
                unlabeled_ids.append(seg_id)
        
        if not unlabeled_ids:
            return  # All labeled!
        
        # For each unlabeled segment, find nearest labeled neighbor
        for seg_id in unlabeled_ids:
            seg_mask = self.segments == seg_id
            seg_coords = np.argwhere(seg_mask)
            
            if len(seg_coords) == 0:
                continue
            
            # Get centroid of this segment
            centroid = seg_coords.mean(axis=0)
            
            # Find nearest labeled segment
            min_dist = float('inf')
            nearest_class = "parking_stalls"  # Fallback
            
            for labeled_id, class_name in self.segment_labels.items():
                labeled_mask = self.segments == labeled_id
                labeled_coords = np.argwhere(labeled_mask)
                
                if len(labeled_coords) > 0:
                    labeled_centroid = labeled_coords.mean(axis=0)
                    dist = np.linalg.norm(centroid - labeled_centroid)
                    
                    if dist < min_dist:
                        min_dist = dist
                        nearest_class = class_name
            
            # Assign nearest neighbor's class (but don't add to segment_labels to track it's auto-filled)
            # We'll add it during save with a flag
            pass  # We'll handle this in submit
        
        return unlabeled_ids
    
    def _save_draft(self):
        """Save a draft of the current work (non-destructive). Runs in a separate thread to avoid blocking the UI."""
        def _worker():
            try:
                try:
                    print("DEBUG _save_draft worker start")
                except Exception:
                    pass
                drafts_dir = (Path.cwd() / 'data' / 'drafts').resolve()
                drafts_dir.mkdir(parents=True, exist_ok=True)
                # Choose image name if available
                if getattr(self, 'image_files', None) and len(self.image_files) > 0:
                    image_name = self.image_files[self.current_idx].stem
                else:
                    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                    image_name = f'draft_{ts}'
                base = drafts_dir / image_name
                # Ensure parent exists
                base.parent.mkdir(parents=True, exist_ok=True)
                try:
                    # Diagnostics for unexpected write failures
                    print(f"DEBUG draft dir: {base.parent.resolve()}, exists={base.parent.exists()}")
                except Exception:
                    pass
                # Save segments
                if self.segments is not None:
                    seg_path = base.with_name(base.name + '_segments.npy')
                    # Ensure parent exists (defensive)
                    try:
                        seg_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(seg_path, 'wb') as _f:
                            pass
                    except Exception:
                        pass
                    try:
                        # Diagnostics
                        try:
                            print(f"DEBUG seg_path: {seg_path}, parent_exists={seg_path.parent.exists()}")
                        except Exception:
                            pass
                        # Write to a temp file and atomically replace to avoid file locking races on Windows
                        # Ensure parent directory exists (retry a few times to mitigate transient OS races on Windows)
                        for _i in range(3):
                            try:
                                seg_path.parent.mkdir(parents=True, exist_ok=True)
                                break
                            except Exception:
                                time.sleep(0.01)
                        if not seg_path.parent.exists():
                            # If, for some reason, the direct parent doesn't exist (transient), fallback to drafts_dir
                            try:
                                print(f"WARNING: seg_path.parent missing ({seg_path.parent}) - falling back to drafts_dir {drafts_dir}")
                            except Exception:
                                pass
                            tmp_dir = drafts_dir
                        else:
                            tmp_dir = seg_path.parent
                        # Create an atomic temporary filename under the determined tmp_dir
                        import uuid
                        tmp_name = tmp_dir / (seg_path.name + f'.tmp.{uuid.uuid4().hex}.npy')
                        np.save(str(tmp_name), self.segments)
                        try:
                            os.replace(str(tmp_name), str(seg_path))
                            try:
                                print(f"DEBUG saved segments -> {seg_path}")
                            except Exception:
                                pass
                        except Exception:
                            # fallback to copy
                            try:
                                shutil.copy2(str(tmp_name), str(seg_path))
                                try:
                                    print(f"DEBUG copied segments -> {seg_path}")
                                except Exception:
                                    pass
                            finally:
                                try:
                                    os.unlink(str(tmp_name))
                                except Exception:
                                    pass
                    except Exception as _e:
                        import traceback as _tb
                        print('Failed saving segments to:', repr(str(seg_path)))
                        print(_tb.format_exc())
                        raise
                    seg_file = seg_path
                else:
                    seg_file = None
                # Build metadata
                data = {
                    'image_name': image_name,
                    'timestamp': datetime.now().isoformat(),
                    'boundary_polygon_px': [[float(x), float(y)] for x, y in self.current_boundary] if getattr(self, 'current_boundary', None) is not None else None,
                    'boundary_transform': {'dx': float(getattr(self, 'boundary_dx', 0.0)), 'dy': float(getattr(self, 'boundary_dy', 0.0)), 'theta_deg': float(getattr(self, 'boundary_theta', 0.0))},
                    'segment_labels': {str(k): v for k, v in getattr(self, 'segment_labels', {}).items()},
                    'manual_polylines': getattr(self, 'manual_polylines', []),
                    'splitting_segment_id': int(getattr(self, 'splitting_segment_id', -1)) if getattr(self, 'splitting_segment_id', None) is not None else None,
                    'min_segment_px': int(getattr(self, 'min_segment_px', 1000)),
                    'target_segments': int(getattr(self, 'target_segments', 50)),
                    'segments_file': seg_file.name if seg_file is not None else None,
                    'note': 'draft',
                }
                json_path = base.with_name(base.name + '_draft.json')
                try:
                    # Write JSON atomically avoiding NamedTemporaryFile to reduce Windows race issues
                    import uuid
                    tmp_json = json_path.parent / (json_path.name + f'.tmp.{uuid.uuid4().hex}.json')
                    with open(tmp_json, 'w', encoding='utf-8') as _tf:
                        _tf.write(json.dumps(data, indent=2))
                        try:
                            _tf.flush()
                            os.fsync(_tf.fileno())
                        except Exception:
                            pass
                    try:
                        os.replace(str(tmp_json), str(json_path))
                        try:
                            print(f"DEBUG saved json -> {json_path}")
                        except Exception:
                            pass
                    except Exception:
                        try:
                            shutil.copy2(str(tmp_json), str(json_path))
                            try:
                                print(f"DEBUG copied json -> {json_path}")
                            except Exception:
                                pass
                        finally:
                            try:
                                os.unlink(str(tmp_json))
                            except Exception:
                                pass
                except Exception as e:
                    import traceback as _tb
                    print('Failed writing json_path:', repr(str(json_path)))
                    print(_tb.format_exc())
                    raise
                # Optionally copy to output folder as well
                if getattr(self, 'output_folder', None) is not None:
                    try:
                        dest = self.output_folder / json_path.name
                        shutil.copy2(str(json_path), dest)
                        if seg_file is not None:
                            shutil.copy2(str(seg_file), str(self.output_folder / Path(seg_file).name))
                    except Exception:
                        pass
                # Notify user on main thread
                try:
                    self.root.after(0, lambda: self.manual_status.config(text=f"Draft saved: {json_path.name}"))
                    self.root.after(0, lambda: messagebox.showinfo('Draft saved', f"Draft saved: {json_path}"))
                except Exception:
                    pass
                try:
                    print("DEBUG drafts dir listing:", [p.name for p in drafts_dir.iterdir()])
                except Exception:
                    pass
                try:
                    print("DEBUG _save_draft worker done")
                except Exception:
                    pass
            except Exception as e:
                print('Error saving draft:', e)
                try:
                    self.root.after(0, lambda: messagebox.showerror('Draft save failed', str(e)))
                except Exception:
                    pass
        # Kick off thread in GUI mode; run synchronously in headless/test mode
        try:
            if getattr(self, '_tk_available', False):
                t = threading.Thread(target=_worker, daemon=True)
                t.start()
            else:
                # Headless - run synchronously so tests can observe results
                _worker()
        except Exception:
            _worker()

    def _submit_annotation(self):
        """Submit and save."""
        if self.output_folder is None:
            messagebox.showwarning("Warning", "Select output folder first")
            return
        
        if self.segments is None:
            messagebox.showwarning("Warning", "Generate segments first")
            return
        
        # Count labeled vs unlabeled segments
        total = sum(1 for i in range(1, self.n_segments + 1) 
                   if np.sum(self.segments == i) >= getattr(self, 'min_segment_px', 1000))
        labeled = sum(1 for i in range(1, self.n_segments + 1)
                     if i in self.segment_labels and np.sum(self.segments == i) >= getattr(self, 'min_segment_px', 1000))
        unlabeled = total - labeled
        
        # Don't auto-fill - leave unlabeled regions as 0 in mask
        # During training, use a mask to ignore unlabeled pixels in loss calculation
        auto_filled = {}
        
        # Build segments dict
        segments_dict = {}
        user_labeled = 0
        
        for i in range(1, self.n_segments + 1):
            seg_id = int(i)  # Ensure native Python int
            area = int(np.sum(self.segments == i))  # Convert numpy int to Python int
            if area >= getattr(self, 'min_segment_px', 100):
                if seg_id in self.segment_labels:
                    # User labeled
                    class_name = self.segment_labels[seg_id]
                    # Normalize legacy names to canonical CLASSES
                    if class_name == 'parking_surface':
                        class_name = 'parking_stalls'
                    if class_name == 'building_roof':
                        class_name = 'building'
                    if class_name not in CLASSES:
                        # conservative fallback
                        class_name = 'parking_stalls'
                    segments_dict[str(seg_id)] = {
                        "class": class_name,
                        "area_px": area,
                        "labeled_by": "user"
                    }
                    user_labeled += 1
                # Unlabeled segments are left out - will be 0 in mask
        
        image_name = self.image_files[self.current_idx].stem
        output_file = self.output_folder / f"{image_name}_labels.json"
        
        # Save segmented image visualization
        output_image = self.output_folder / f"{image_name}_segmented.png"
        if self.segments is not None:
            # Create visualization with color-coded segments
            h, w = self.clean_image.shape[:2]
            vis_img = cv2.cvtColor(self.clean_image, cv2.COLOR_BGR2RGB).copy().astype(float) / 255
            alpha = 0.5
            
            for seg_id in range(1, self.n_segments + 1):
                seg_mask = self.segments == seg_id
                if np.sum(seg_mask) < getattr(self, 'min_segment_px', 1000):
                    continue
                
                # Only visualize user-labeled segments
                if seg_id in self.segment_labels:
                    class_name = self.segment_labels[seg_id]
                # normalize known legacy class names
                if class_name == 'parking_surface':
                    class_name = 'parking_stalls'
                if class_name == 'building_roof':
                    class_name = 'building'
            vis_img[all_edges] = [1, 1, 1]
            
            # Draw boundary
            if self.current_boundary is not None:
                boundary_int = self.current_boundary.astype(np.int32)
                cv2.polylines(vis_img, [boundary_int], isClosed=True, color=(1, 1, 0), thickness=2)
            
            # Save
            vis_img_bgr = (vis_img * 255).astype(np.uint8)
            vis_img_bgr = cv2.cvtColor(vis_img_bgr, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_image), vis_img_bgr)
        
        # Save segment boundaries as numpy array
        segments_file = self.output_folder / f"{image_name}_segments.npy"
        np.save(str(segments_file), self.segments)
        
        # Save class mask for ML training (each pixel = class ID)
        mask_file = self.output_folder / f"{image_name}_mask.png"
        h, w = self.segments.shape
        class_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Map each class to a unique ID
        class_to_id = {cls: idx for idx, cls in enumerate(CLASSES)}
        
        for seg_id in range(1, self.n_segments + 1):
            seg_mask = self.segments == seg_id
            if np.sum(seg_mask) < getattr(self, 'min_segment_px', 1000):
                continue
            
            # Only include user-labeled segments
            if seg_id in self.segment_labels:
                class_name = self.segment_labels[seg_id]
                class_id = class_to_id.get(class_name, 0)
                class_mask[seg_mask] = class_id
            # Unlabeled segments remain 0 (background/ignore class)
        
        cv2.imwrite(str(mask_file), class_mask)
        
        data = {
            "image_id": image_name,
            "timestamp": datetime.now().isoformat(),
            "boundary_status": "approved",
            "boundary_transform": {
                "dx": float(self.boundary_dx),
                "dy": float(self.boundary_dy),
                "theta_deg": float(self.boundary_theta)
            },
            "boundary_polygon_px": [[float(x), float(y)] for x, y in self.current_boundary] if self.current_boundary is not None else None,
            "slic_params": {
                "n_segments": int(self.n_segments),
                "compactness": 12,
                "sigma": 1
            },
            "segments": segments_dict,
            "road_access": {
                "lines": self.road_access_lines,
                "none": True
            },
            "metadata": {
                "labeler": "labeling_tool",
                "final_segment_count": len(segments_dict),
                "user_labeled_count": user_labeled,
                "unlabeled_count": unlabeled,
                "total_segments": total
            }
        }

        # Convert fractional boundary access segments to pixel endpoints and add to export
        if self.boundary_access_segments:
            def _frac_to_pt_local(frac):
                pts = self.current_boundary
                seg_starts = pts
                seg_ends = np.vstack([pts[1:], pts[0]])
                seg_vecs = seg_ends - seg_starts
                seg_lens = np.linalg.norm(seg_vecs, axis=1)
                cum = np.concatenate([[0], np.cumsum(seg_lens)])
                total = cum[-1]
                f = (frac % 1.0) * total
                idx = np.searchsorted(cum, f, side='right') - 1
                idx = max(0, min(idx, len(seg_lens)-1))
                local_f = (f - cum[idx]) / (seg_lens[idx] if seg_lens[idx]>0 else 1e-6)
                pt = seg_starts[idx] + seg_vecs[idx] * local_f
                return [float(pt[0]), float(pt[1])]
            segs_out = []
            for seg in self.boundary_access_segments:
                s_pt = _frac_to_pt_local(seg['start_frac'])
                e_pt = _frac_to_pt_local(seg['end_frac'])
                seg_copy = seg.copy()
                seg_copy['start_px'] = s_pt
                seg_copy['end_px'] = e_pt
                segs_out.append(seg_copy)
            data['boundary_access'] = {'segments': segs_out}
        else:
            data['boundary_access'] = {'segments': []}

        # Save road mask if present (values 0=unlabeled/ignore,1=public_road,2=non_road)
        if self.road_mask is not None:
            road_mask_file = self.output_folder / f"{image_name}_road_mask.png"
            # Save raw values (0,1,2) directly
            cv2.imwrite(str(road_mask_file), (self.road_mask).astype(np.uint8))
            data['road_mask_file'] = road_mask_file.name
            data['road_mask_info'] = {
                'buffer_mode': self.buffer_mode,
                'buffer_px': self.road_buffer_px,
                'buffer_pct': self.road_buffer_pct,
                'brush_px': self.road_brush_size,
                'legend': {'0':'ignore','1':'public_road','2':'non_road'}
            }

        # Export interior simplified mask combining classes for training
        if self.current_boundary is not None:
            h, w = self.segments.shape
            roi_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(roi_mask, [self.current_boundary.astype(np.int32)], 1)
            interior_mask = np.zeros((h, w), dtype=np.uint8)
            # Build pixel-wise class using user-labeled segments
            for seg_id_str, seg_data in segments_dict.items():
                seg_id = int(seg_id_str)
                class_name = seg_data['class']
                export_id = self.interior_export_map.get(class_name, 0)
                if export_id > 0:
                    interior_mask[self.segments == seg_id] = export_id
            interior_mask_file = self.output_folder / f"{image_name}_interior_mask.png"
            cv2.imwrite(str(interior_mask_file), interior_mask.astype(np.uint8))
            data['interior_mask_file'] = interior_mask_file.name
            data['interior_mask_info'] = {
                'legend': {0:'ignore',1:'parking_or_road',2:'building_immobile',3:'vegetation'}
            }

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Clear unsaved changes flag
        self.has_unsaved_changes = False
        
        # Update submit button to show saved state
        self.submit_btn.config(text="✓ Saved!", bg='#90EE90')
        
        # Update status to show saved state
        self.boundary_status.config(text=f"✓ Saved! ({user_labeled}/{total} labeled, {unlabeled} unlabeled)")
        
        # Move to next image automatically
        if self.current_idx < len(self.image_files) - 1:
            self._next_image()
        else:
            messagebox.showinfo("Complete", "All images done!")
            
    def run(self):
        """Start the application."""
        self.root.mainloop()
    
    def _save_folder_preferences(self):
        """Save last used folders and current image index to config file."""
        try:
            config = {}
            if self.images_folder:
                config['input_folder'] = str(self.images_folder)
            if self.output_folder:
                config['output_folder'] = str(self.output_folder)
            # Save current image index
            config['current_image_index'] = self.current_idx
            
            self.CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(self.CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save folder preferences: {e}")
    
    def _load_folder_preferences(self):
        """Load last used folders from config file and auto-load if valid."""
        try:
            print(f"Loading preferences from: {self.CONFIG_FILE}")
            if not self.CONFIG_FILE.exists():
                print("No config file found - starting fresh")
                return
            
            with open(self.CONFIG_FILE, 'r') as f:
                config = json.load(f)
            
            input_folder = config.get('input_folder')
            output_folder = config.get('output_folder')
            
            # Auto-load input folder if it exists
            if input_folder and Path(input_folder).exists():
                self.images_folder = Path(input_folder)
                self.last_input_folder = Path(input_folder)
                self.image_files = sorted(list(self.images_folder.glob("*.png")) + 
                                         list(self.images_folder.glob("*.jpg")))
                if self.image_files:
                    self.input_label.config(text=f"{len(self.image_files)} images found")
                    print(f"Auto-loaded input folder: {self.images_folder}")
            
            # Auto-load output folder if it exists
            if output_folder and Path(output_folder).exists():
                self.output_folder = Path(output_folder)
                self.last_output_folder = Path(output_folder)
                self.output_label.config(text=f"Output: {self.output_folder.name}")
                print(f"Auto-loaded output folder: {self.output_folder}")
            
            # Enable buttons and load last viewed image if both folders are valid
            if self.images_folder and self.output_folder and self.image_files:
                self.approve_btn.config(state=tk.NORMAL)
                self.reject_btn.config(state=tk.NORMAL)
                
                # Restore last viewed image index
                last_idx = config.get('current_image_index', 0)
                # Ensure index is within bounds
                last_idx = max(0, min(last_idx, len(self.image_files) - 1))
                print(f"Attempting to load image at index {last_idx}")
                self._load_image(last_idx)
                print(f"Restored to image {last_idx + 1}/{len(self.image_files)}")
                
        except Exception as e:
            print(f"ERROR: Could not load folder preferences: {e}")
            import traceback
            traceback.print_exc()

    # --------------------- Collapsible section helper ---------------------
    def _make_section(self, parent, title, default_open=True):
        """Create a titled, collapsible content frame and return the content frame.
        The header includes a small toggle button to collapse/expand.

        Previously the header and content were siblings of the parent, which caused
        the content to be packed at the end of the parent when shown (so expanded
        content could appear far from its header). We now create a single
        container frame for the whole section and pack both header and content
        inside it so toggling preserves the section's position.
        """
        # Outer container keeps header and content together
        section = tk.Frame(parent, bg=parent['bg'])
        section.pack(pady=1, fill='x')

        header = tk.Frame(section, bg=parent['bg'])
        header.pack(fill='x')
        lbl = tk.Label(header, text=title, font=("Arial", 10, "bold"), bg=parent['bg'])
        lbl.pack(side=tk.LEFT)
        btn = tk.Button(header, text='▾' if default_open else '▸', width=3,
                        command=lambda t=title: self._toggle_section(t))
        btn.pack(side=tk.RIGHT)

        # Content lives inside the section container so pack order is preserved
        content = tk.Frame(section, bg=parent['bg'])
        if default_open:
            content.pack(pady=1, fill='x')
        self.section_frames[title] = (content, btn)
        return content

    def _toggle_section(self, title):
        """Toggle visibility of a previously created section."""
        content, btn = self.section_frames.get(title, (None, None))
        if content is None:
            return
        if content.winfo_viewable():
            content.pack_forget()
            btn.config(text='▸')
        else:
            content.pack(pady=1, fill='x')
            btn.config(text='▾')

    def _set_mode(self, mode: str):
        """Set the current tool mode and show/hide UI elements accordingly.

        Modes:
          - 'boundary' : Boundary adjust tools
          - 'segments' : Segmentation tools
          - 'label'    : Labeling tools + submit
          - 'access'   : Access & Road tools
        """
        # Hide all sections and grouped frames first
        # Hide every collapsible section content
        for title, (content, btn) in self.section_frames.items():
            try:
                content.pack_forget()
                btn.config(text='▸')
            except Exception:
                pass
        # Hide grouped frames
        try:
            self.boundary_tools_frame.pack_forget()
        except Exception:
            pass
        # Keep submit_frame visible so Save is always accessible
        # (users reported it being hidden unexpectedly)
        try:
            # Re-pack into footer explicitly to ensure it stays at the bottom
            self.submit_frame.pack_forget()
            self.submit_frame.pack(in_=self.footer_frame, pady=1, fill='x')
        except Exception:
            pass

        # Reset button styles and ensure all mode buttons are enabled so user can switch anytime
        for k, b in self.mode_buttons.items():
            try:
                b.config(bg=None, state=tk.NORMAL)
            except Exception:
                pass

        # Ensure access/road/manual modes are disabled when switching away
        if mode != 'access':
            if getattr(self, 'boundary_access_mode', False):
                self.boundary_access_mode = False
                try:
                    self.boundary_access_btn.config(bg='#FFE4B5')
                except Exception:
                    pass
            if getattr(self, 'road_mode_active', False):
                self.road_mode_active = False
                try:
                    self.road_mode_btn.config(bg='#E0FFFF')
                except Exception:
                    pass
        if getattr(self, 'manual_mode', False) and mode != 'segments':
            # Turn off manual split mode when leaving segments
            try:
                self._toggle_manual_mode()
            except Exception:
                self.manual_mode = False

        # Show controls for the selected mode
        self.mode = mode
        if mode == 'boundary':
            # Show boundary controls
            try:
                self.boundary_tools_frame.pack(pady=1, fill='x')
            except Exception:
                pass
            self.mode_buttons['boundary'].config(bg='#90EE90')
            # Set title hint
            self.ax.set_title("Adjust boundary (arrows=move, </>=rotate) and press Enter to proceed")
            self.canvas.draw()
        elif mode == 'segments':
            # Show segments section
            content, btn = self.section_frames.get('4. Segments', (None, None))
            if content is not None:
                content.pack(pady=1, fill='x')
                btn.config(text='▾')
            self.mode_buttons['segments'].config(bg='#90EE90')
            # Keep Label button enabled so user can always switch modes
            try:
                self.mode_buttons['label'].config(state=tk.NORMAL)
            except Exception:
                pass
            # Also show class selection and ensure submit stays in footer (visible at bottom)
            content_cls, btn_cls = self.section_frames.get('5. Class (click to label)', (None, None))
            if content_cls is not None:
                content_cls.pack(pady=1, fill='x')
                try:
                    btn_cls.config(text='▾')
                except Exception:
                    pass
            try:
                self.submit_frame.pack_forget()
                self.submit_frame.pack(in_=self.footer_frame, pady=1, fill='x')
            except Exception:
                pass
            self.ax.set_title("SEGMENT MODE: Label, split, refine segments")
            self.canvas.draw()
        elif mode == 'label':
            # Show class selection and submit controls
            content, btn = self.section_frames.get('5. Class (click to label)', (None, None))
            if content is not None:
                content.pack(pady=1, fill='x')
                btn.config(text='▾')
            try:
                self.submit_frame.pack(pady=1, fill='x')
            except Exception:
                pass
            # Ensure label button is enabled when entering Label mode
            try:
                self.mode_buttons['label'].config(state=tk.NORMAL, bg='#90EE90')
            except Exception:
                pass
            # Auto-approve boundary and generate segments if a boundary exists but is not yet approved
            if self.current_boundary is not None and not self.boundary_approved:
                try:
                    self._approve_boundary()
                except Exception:
                    pass
            # If we are currently in manual/split mode, exit it so labeling is available
            if getattr(self, 'manual_mode', False):
                try:
                    self._toggle_manual_mode()
                except Exception:
                    self.manual_mode = False
                    self._clear_manual_line()
            # If boundary is approved but segments haven't been generated (e.g., saved boundary without segments), generate them now
            if self.boundary_approved and self.segments is None:
                try:
                    self._generate_segments()
                except Exception:
                    pass
                # If segments still not generated (e.g., no clean_image available), create a safe fallback single-segment mask covering ROI+buffer
                if self.segments is None and self.current_boundary is not None:
                    # If clean_image exists, use buffer mask; otherwise compute a minimal canvas based on boundary extents
                    if self.clean_image is not None:
                        self._compute_buffer_mask()
                        h, w = (self.clean_image.shape[0], self.clean_image.shape[1])
                        roi_mask = np.zeros((h, w), dtype=np.uint8)
                        cv2.fillPoly(roi_mask, [self.current_boundary.astype(np.int32)], 255)
                        buffer_mask = self.buffer_mask.astype(np.uint8) if getattr(self, 'buffer_mask', None) is not None else np.zeros_like(roi_mask)
                        expanded = (roi_mask > 0) | (buffer_mask > 0)
                    else:
                        # Build a minimal canvas that's slightly larger than the boundary bbox
                        min_xy = np.floor(self.current_boundary.min(axis=0)).astype(int)
                        max_xy = np.ceil(self.current_boundary.max(axis=0)).astype(int)
                        pad = 10
                        h = max(100, max_xy[1] + pad)
                        w = max(100, max_xy[0] + pad)
                        roi_mask = np.zeros((h, w), dtype=np.uint8)
                        shifted = self.current_boundary.copy()
                        # ensure coordinates are within the fallback canvas
                        shifted[:,0] = np.clip(shifted[:,0], 0, w-1)
                        shifted[:,1] = np.clip(shifted[:,1], 0, h-1)
                        cv2.fillPoly(roi_mask, [shifted.astype(np.int32)], 255)
                        expanded = (roi_mask > 0)
                    if expanded.any():
                        self.segments = np.zeros((h, w), dtype=np.int32)
                        self.segments[expanded] = 1
                        self.n_segments = 1
                        try:
                            self.seg_status.config(text=f"✓ {self.n_segments} segments (fallback)")
                        except Exception:
                            pass
                        try:
                            self._update_display()
                        except Exception:
                            pass
            self.ax.set_title("LABEL MODE: Click segment to label (right-click to remove)")
            self.canvas.draw()
        elif mode == 'access':
            # Show access & roads
            content, btn = self.section_frames.get('7. Access & Roads', (None, None))
            if content is not None:
                content.pack(pady=1, fill='x')
                btn.config(text='▾')
            # Default to NOT being in painting/access editing modes — keep user in label mode unless they press the buttons
            try:
                self.boundary_access_mode = False
                self.boundary_access_btn.config(bg='#FFE4B5')
            except Exception:
                pass
            self.mode_buttons['access'].config(bg='#90EE90')
            self.ax.set_title("ACCESS MODE: Boundary access and road painting (use buttons to enable)")
            self.canvas.draw()
        else:
            # Unknown mode - default to boundary
            self._set_mode('boundary')

        # Always update the title to reflect the current mode in a single place
        try:
            self._update_title_for_mode()
        except Exception:
            pass



def main():

    """Main entry point."""
    tool = LabelingTool()
    tool.run()


if __name__ == "__main__":
    main()
