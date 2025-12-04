import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
import numpy as np
import requests
import math
import numpy as np

def dist(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

# Colors for groups (cycling)
COLORS = [
    ("red", "blue"),       # points_color, intersection_color
    ("green", "dark green"),
    ("orange", "dark orange"),
    ("yellow", "olive"),
]

class ImageAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Analysis Tool")
        self.root.geometry("1000x700")

        # Data structures
        self.image_path = None
        self.original_image = None
        self.tk_image = None
        self.current_points = []  # List of (x, y) in ORIGINAL coordinates
        self.all_groups = []      # List of lists of points in ORIGINAL coordinates
        self.group_results = []   # List of API results for each group
        
        # Canvas items mapping: item_id -> type/group_info
        self.group_items = [] 
        self.current_point_items = [] 
        self.temp_items = [] 

        self.selected_point_index = None 
        self.selected_item_id = None
        
        # State
        self.marking_mode = False
        self.scale_ratio = 1.0
        self.pixels_per_unit = None # Calibration scale (pixels per cm)

        self._setup_ui()

    def _setup_ui(self):
        # Left Toolbar
        self.toolbar = tk.Frame(self.root, width=200, bg="#f0f0f0", padx=10, pady=10)
        self.toolbar.pack(side=tk.LEFT, fill=tk.Y)
        self.toolbar.pack_propagate(False)

        # Buttons
        btn_opts = {'fill': tk.X, 'pady': 5}
        
        tk.Button(self.toolbar, text="Upload Image", command=self.upload_image, bg="white").pack(**btn_opts)
        
        tk.Button(self.toolbar, text="Calibrate", command=self.calibrate, bg="#e6e6fa").pack(**btn_opts)

        self.btn_point = tk.Button(self.toolbar, text="Point", command=self.toggle_point_mode, bg="#e0e0e0")
        self.btn_point.pack(**btn_opts)
        
        tk.Button(self.toolbar, text="Compute", command=self.compute, bg="#dddddd").pack(**btn_opts)
        tk.Button(self.toolbar, text="Save Image", command=self.save_image, bg="#ccffcc").pack(**btn_opts)
        tk.Button(self.toolbar, text="Clear", command=self.clear, bg="#ffcccc").pack(**btn_opts)
        tk.Button(self.toolbar, text="Delete All", command=self.delete_all, bg="#ff9999").pack(**btn_opts)

        # Instructions Label
        self.info_label = tk.Label(self.toolbar, text="Upload an image\nto start.", bg="#f0f0f0", justify=tk.LEFT, wraplength=180)
        self.info_label.pack(side=tk.BOTTOM, pady=20)

        # Center Canvas
        self.canvas_frame = tk.Frame(self.root, bg="gray")
        self.canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.canvas_frame, bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Bind events
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Configure>", self.on_resize)
        
        # Dragging bindings
        self.canvas.tag_bind("movable", "<Button-1>", self.on_label_press)
        self.canvas.tag_bind("movable", "<B1-Motion>", self.on_label_drag)
        self.canvas.tag_bind("movable", "<ButtonRelease-1>", self.on_label_release)

    def toggle_point_mode(self):
        self.marking_mode = not self.marking_mode
        if self.marking_mode:
            self.btn_point.config(relief=tk.SUNKEN, bg="#bbbbbb")
            self.info_label.config(text="Point Mode ON.\nClick to mark points.")
        else:
            self.btn_point.config(relief=tk.RAISED, bg="#e0e0e0")
            self.info_label.config(text="Point Mode OFF.")

    def calibrate(self):
        if not self.original_image:
            messagebox.showwarning("Warning", "Upload an image first.")
            return
            
        # Open Toplevel window
        cal_win = tk.Toplevel(self.root)
        cal_win.title("Calibrate - Select 2 Points")
        cal_win.geometry("800x600")
        
        # Canvas for calibration
        cal_canvas = tk.Canvas(cal_win, bg="gray")
        cal_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Display image (resized for this window)
        # We need a separate copy/resize logic for this window or reuse existing if simple
        # Let's just fit to window size 800x500 approx
        img_w, img_h = self.original_image.size
        cal_w, cal_h = 800, 500
        ratio = min(cal_w / img_w, cal_h / img_h)
        # Display image (scaled to fit this window)
        # We need a separate copy of the image for this window
        img_w, img_h = self.original_image.size
        # Initial fit
        win_w, win_h = 800, 600
        ratio = min(win_w / img_w, win_h / img_h)
        new_w = int(img_w * ratio)
        new_h = int(img_h * ratio)
        
        cal_img_tk = ImageTk.PhotoImage(self.original_image.resize((new_w, new_h), Image.Resampling.LANCZOS))
        cal_canvas.create_image(0, 0, image=cal_img_tk, anchor=tk.NW)
        
        # Keep reference
        cal_win.cal_img_tk = cal_img_tk 
        
        cal_points = []
        
        def on_cal_click(event):
            if len(cal_points) >= 2:
                return
            
            # Convert to original coords
            ox = event.x / ratio
            oy = event.y / ratio
            cal_points.append((ox, oy))
            
            # Draw marker
            r = 5
            cal_canvas.create_oval(event.x-r, event.y-r, event.x+r, event.y+r, fill="blue", outline="white")
            
            if len(cal_points) == 2:
                confirm_btn.config(state=tk.NORMAL)
                
        cal_canvas.bind("<Button-1>", on_cal_click)
        
        def confirm_calibration():
            if len(cal_points) != 2:
                return
            
            p1 = cal_points[0]
            p2 = cal_points[1]
            dist_px = dist(p1, p2)
            
            if dist_px == 0:
                messagebox.showerror("Error", "Points cannot be the same.")
                return
                
            # 1 cm reference
            self.pixels_per_cm = dist_px / 1.0
            
            messagebox.showinfo("Success", f"Calibration set: {self.pixels_per_cm:.2f} px/cm")
            cal_win.destroy()
            self.redraw_all() # Update labels
            
        confirm_btn = tk.Button(cal_win, text="Confirm", command=confirm_calibration, state=tk.DISABLED)
        confirm_btn.pack(side=tk.BOTTOM, pady=10)

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if not file_path:
            return

        # Reset everything before loading new image
        self.delete_all()
        self.image_path = file_path
        
        try:
            self.original_image = Image.open(self.image_path)
            self.info_label.config(text="Image uploaded.\nClick 'Point' to start marking.")
            self.redraw_all()
            
        except Exception as e:
            print(f"Error loading image: {e}")
            messagebox.showerror("Error", f"Failed to load image: {e}")
            self.image_path = None 

    def on_resize(self, event):
        if self.original_image:
            self.redraw_all()

    def redraw_all(self):
        self.canvas.delete("all")
        self.group_items = [] # Rebuild these
        self.current_point_items = []
        
        if not self.original_image:
            return

        # Calculate new size
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            return

        img_w, img_h = self.original_image.size
        ratio = min(canvas_width / img_w, canvas_height / img_h)
        new_w = int(img_w * ratio)
        new_h = int(img_h * ratio)
        
        self.scale_ratio = ratio
        
        # Resize and display
        resized_image = self.original_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(resized_image)
        self.canvas.create_image(0, 0, image=self.tk_image, anchor=tk.NW, tags="image")

        # Redraw all groups
        # We need to rebuild self.group_items to track them correctly
        # Since we cleared canvas, we just iterate all_groups and draw them
        
        # Reset group_items to match all_groups structure (list of lists)
        self.group_items = [] 
        for i, res in enumerate(self.group_results):
            if res is None: # Was deleted
                self.group_items.append([])
                continue
            self.draw_group(res, i)

        # Redraw current points
        for i, pt in enumerate(self.current_points):
            self._draw_single_point(pt, i + 1)

    def _draw_single_point(self, pt_orig, label_num):
        x, y = self.to_display_coords(pt_orig[0], pt_orig[1])
        r = 5
        item_id = self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="black", outline="white", tags=("point", "current"))
        text_id = self.canvas.create_text(x+10, y-10, text=f"P{label_num}", fill="black", tags=("text", "current"))
        self.current_point_items.append(item_id)
        self.current_point_items.append(text_id)

    def to_display_coords(self, x, y):
        return x * self.scale_ratio, y * self.scale_ratio

    def to_original_coords(self, x, y):
        return x / self.scale_ratio, y / self.scale_ratio

    def on_canvas_click(self, event):
        if not self.original_image:
            return

        # If NOT in marking mode, try to select items (groups or current points)
        if not self.marking_mode:
            clicked_item = self.canvas.find_closest(event.x, event.y, halo=5)
            if clicked_item:
                tags = self.canvas.gettags(clicked_item[0])
                # Check for group items or current points
                # We look for "group_" or "current" tags
                if any(t.startswith("group_") for t in tags) or "current" in tags:
                    self.select_point(clicked_item[0])
                    return

        if self.marking_mode:
            # Convert to original coords for storage
            ox, oy = self.to_original_coords(event.x, event.y)
            self.add_point(ox, oy)

    def add_point(self, x, y):
        # x, y are in ORIGINAL coords
        self.current_points.append((x, y))
        self._draw_single_point((x, y), len(self.current_points))
        self.info_label.config(text=f"Points: {len(self.current_points)}")

    def select_point(self, item_id):
        # Removed visual highlight as requested
        self.selected_item_id = item_id
        self.info_label.config(text="Item selected. Press Clear to remove.")

    def compute(self):
        n_points = len(self.current_points)
        if n_points == 0:
            messagebox.showwarning("Warning", "No points to compute.")
            return
            
        if n_points % 3 != 0:
            messagebox.showwarning("Warning", f"Need multiples of 3 points. You have {n_points}.")
            return

        # Auto-disable point mode
        if self.marking_mode:
            self.toggle_point_mode()

        points_copy = self.current_points[:] 
        
        # Clear temporary markers
        for item in self.current_point_items:
            self.canvas.delete(item)
        self.current_point_items = []
        self.current_points = [] 

        for i in range(0, n_points, 3):
            group_pts = points_copy[i:i+3]
            
            # Call API
            try:
                payload = {"points": group_pts, "pixels_per_cm": self.pixels_per_cm}
                response = requests.post("http://127.0.0.1:5000/compute", json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    group_idx = len(self.all_groups)
                    self.all_groups.append(group_pts)
                    self.group_results.append(result)
                    self.draw_group(result, group_idx)
                else:
                    messagebox.showerror("Error", f"Server error: {response.text}")
                    # Restore points if failed? For now just stop.
                    self.current_points = points_copy # Restore
                    self.redraw_all() # Restore view
                    return
                    
            except Exception as e:
                messagebox.showerror("Error", f"Connection error: {e}")
                self.current_points = points_copy
                self.redraw_all()
                return
            
        self.info_label.config(text=f"Computed {n_points // 3} group(s).")

    def draw_group(self, result, group_idx):
        # result contains "construction" and "measurements"
        const = result["construction"]
        meas = result["measurements"]
        
        point_color, inter_color = COLORS[group_idx % len(COLORS)]
        group_items = []

        # Unpack construction points
        top = const["top"]
        next_top = const["next_top"]
        lowest = const["lowest"]
        # (ix, iy) is the PERPENDICULAR PROJECTION of 'top' onto the 'next_top'-'lowest' line
        ix, iy = const["intersection"] 
        proj = const["projection"]
        hyp_a = const["hyp_a"]
        hyp_b = const["hyp_b"]
        third = const["third"]
        
        # 2. Draw (Convert to DISPLAY coords)
        
        # Helper for conversion
        def d(pt):
            return self.to_display_coords(pt[0], pt[1])
            
        # Axes
        d_top = d(top)
        d_ix, d_iy = self.to_display_coords(ix, iy)
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        
        # Horizontal Axis (Slanted: next_top -> lowest line, extended across canvas)
        d_nt = d(next_top)
        d_low = d(lowest)
        
        dx = d_low[0] - d_nt[0]
        dy = d_low[1] - d_nt[1]
        
        # Calculate start/end points for the extended horizontal line
        if dx == 0:
            m = 99999999 # effectively infinite slope
            l2 = self.canvas.create_line(d_nt[0], 0, d_nt[0], h, fill=inter_color, width=2, tags=f"group_{group_idx}")
        else:
            m = dy / dx
            y_left = d_nt[1] + m * (0 - d_nt[0])
            y_right = d_nt[1] + m * (w - d_nt[0])
            l2 = self.canvas.create_line(0, y_left, w, y_right, fill=inter_color, width=2, tags=f"group_{group_idx}")

        # --- MODIFIED VERTICAL AXIS (Extended perpendicular line) ---
        # The perpendicular line passes through (d_ix, d_iy) and has slope -1/m
        if dx == 0: # If horizontal line is vertical (m is infinity)
            # The vertical axis is horizontal, passing through (d_ix, d_iy)
            l1 = self.canvas.create_line(0, d_ix, w, d_ix, fill=inter_color, width=2, tags=f"group_{group_idx}") 
        elif dy == 0: # If horizontal line is truly horizontal (m is zero)
            # The vertical axis is truly vertical, passing through (d_ix, d_iy)
            l1 = self.canvas.create_line(d_ix, 0, d_ix, h, fill=inter_color, width=2, tags=f"group_{group_idx}")
        else:
            m_perp = -1.0 / m
            # y = m_perp * (x - d_ix) + d_iy
            
            # Calculate start/end points for the extended vertical line
            y_left_perp = d_iy + m_perp * (0 - d_ix)
            y_right_perp = d_iy + m_perp * (w - d_ix)
            
            l1 = self.canvas.create_line(0, y_left_perp, w, y_right_perp, fill=inter_color, width=2, tags=f"group_{group_idx}")

        group_items.extend([l1, l2])
        
        # Triangle
        d_hyp_a = d(hyp_a)
        d_hyp_b = d(hyp_b)
        d_third = d(third)
        d_next_top = d(next_top)
        d_proj = d(proj)

        l3 = self.canvas.create_line(d_hyp_a[0], d_hyp_a[1], d_hyp_b[0], d_hyp_b[1], fill="#000000", width=2, tags=f"group_{group_idx}")
        l4 = self.canvas.create_line(d_third[0], d_third[1], d_next_top[0], d_next_top[1], fill="#000000", width=2, tags=f"group_{group_idx}")
        l5 = self.canvas.create_line(d_hyp_a[0], d_hyp_a[1], d_third[0], d_third[1], fill="#000000", width=2, tags=f"group_{group_idx}")
        l6 = self.canvas.create_line(d_hyp_b[0], d_hyp_b[1], d_third[0], d_third[1], fill="#000000", width=2, tags=f"group_{group_idx}")
        
        group_items.extend([l3, l4, l5, l6])

        # Projection (This is the original line from third point to its projection point)
        l7 = self.canvas.create_line(d_third[0], d_third[1], d_proj[0], d_proj[1], fill="black", width=2, dash=(4, 2), tags=f"group_{group_idx}")
        group_items.append(l7)

        # 3. Add Movable Labels
        cx = (d_hyp_a[0] + d_hyp_b[0] + d_third[0]) / 3
        cy = (d_hyp_a[1] + d_hyp_b[1] + d_third[1]) / 3
        centroid = (cx, cy)

        def get_offset_pos(p1, p2, cent, offset=30):
            mx = (p1[0] + p2[0]) / 2
            my = (p1[1] + p2[1]) / 2
            vx = mx - cent[0]
            vy = my - cent[1]
            mag = np.sqrt(vx**2 + vy**2)
            if mag == 0: return (mx, my)
            vx /= mag
            vy /= mag
            return (mx + vx * offset, my + vy * offset)

        # Actual Throat
        pos_at = get_offset_pos(d_hyp_a, d_hyp_b, centroid, offset=40)
        lbl_at = self.create_movable_label(pos_at[0], pos_at[1], meas["actual_throat"]["label"], group_idx, "actual_throat")
        
        # Leg1 (Origin -> hyp_a) 
        pos_l1 = get_offset_pos((d_ix, d_iy), d_hyp_a, centroid, offset=40)
        lbl_l1 = self.create_movable_label(pos_l1[0], pos_l1[1], meas["leg1"]["label"], group_idx, "leg1")
        
        # Leg2 (Origin -> hyp_b) 
        pos_l2 = get_offset_pos((d_ix, d_iy), d_hyp_b, centroid, offset=40)
        lbl_l2 = self.create_movable_label(pos_l2[0], pos_l2[1], meas["leg2"]["label"], group_idx, "leg2")
        
        # Root Penetration
        pos_rp = get_offset_pos(d_third, (d_ix, d_iy), centroid, offset=20)
        lbl_rp = self.create_movable_label(pos_rp[0], pos_rp[1], meas["root_penetration"]["label"], group_idx, "root_penetration")

        # Effective Throat
        mx_et = (d_third[0] + d_proj[0]) / 2
        my_et = (d_third[1] + d_proj[1]) / 2
        lbl_et = self.create_movable_label(mx_et + 20, my_et + 20, meas["effective_throat"]["label"], group_idx, "effective_throat")

        group_items.extend([lbl_at, lbl_l1, lbl_l2, lbl_rp, lbl_et])
        self.group_items.append(group_items)

    def create_movable_label(self, x, y, text, group_idx, label_type):
        # Create text with a background rectangle for better visibility? 
        # For now just text.
        tag = f"label_group_{group_idx}"
        unique_tag = f"label_{label_type}_{group_idx}"
        item_id = self.canvas.create_text(x, y, text=text, fill="white", font=("Arial", 12, "bold"), tags=("movable", f"group_{group_idx}", tag, unique_tag))
        return item_id

    def on_label_press(self, event):
        # Find closest movable item
        item = self.canvas.find_closest(event.x, event.y, halo=5)
        if item:
            tags = self.canvas.gettags(item[0])
            if "movable" in tags:
                self._drag_data = {"item": item[0], "x": event.x, "y": event.y}

    def on_label_drag(self, event):
        if hasattr(self, '_drag_data') and self._drag_data["item"]:
            dx = event.x - self._drag_data["x"]
            dy = event.y - self._drag_data["y"]
            self.canvas.move(self._drag_data["item"], dx, dy)
            self._drag_data["x"] = event.x
            self._drag_data["y"] = event.y

    def on_label_release(self, event):
        self._drag_data = {"item": None, "x": 0, "y": 0}


    def save_image(self):
        if not self.original_image:
            messagebox.showwarning("Warning", "No image to save.")
            return

        # Create a copy of the original image to draw on
        img_to_save = self.original_image.copy()
        draw = ImageDraw.Draw(img_to_save)
        
        # Dynamic font size based on image height (approx 1.5%)
        img_w, img_h = img_to_save.size
        font_size = max(12, int(img_h * 0.015))
        try:
            # Try to use a standard font, fallback to default if not found
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
            font_bold = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
        except IOError:
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
                font_bold = ImageFont.truetype("arialbd.ttf", font_size)
            except IOError:
                font = ImageFont.load_default()
                font_bold = ImageFont.load_default()

        # Collect status blocks
        weld_blocks = [] # List of lists of strings

        # Re-draw all groups on the PIL image
        for i, result in enumerate(self.group_results):
            if result is None:
                continue

            const = result["construction"]
            meas = result["measurements"]
            
            # Unpack points
            top = const["top"]
            ix, iy = const["intersection"] # Now the projection point
            hyp_a = const["hyp_a"]
            hyp_b = const["hyp_b"]
            third = const["third"]
            next_top = const["next_top"]
            proj = const["projection"]
            
            # Draw lines (Black, width scaled)
            line_width = max(2, int(img_h * 0.003))
            
            # Triangle
            draw.line([hyp_a, hyp_b], fill="black", width=line_width)
            draw.line([third, next_top], fill="black", width=line_width)
            draw.line([hyp_a, third], fill="black", width=line_width)
            draw.line([hyp_b, third], fill="black", width=line_width)
            
            # Projection
            draw.line([third, proj], fill="black", width=line_width)

            # Axes (White lines inside the triangle)
            origin = (ix, iy) # The new projection origin
            # Leg Lines (Origin to Hypotenuse Endpoints)
            draw.line([origin, hyp_a], fill="white", width=line_width)
            draw.line([origin, hyp_b], fill="white", width=line_width)

            # --- Labels ---
            # Helper to get position from unique tag
            def get_pos(label_type, default_pos):
                unique_tag = f"label_{label_type}_{i}"
                # Find item with this tag
                items = self.canvas.find_withtag(unique_tag)
                if items:
                    item_id = items[0]
                    coords = self.canvas.coords(item_id)
                    # coords is [x, y] in display coordinates
                    ox, oy = self.to_original_coords(coords[0], coords[1])
                    return (ox, oy)
                return default_pos

            offset_val = max(30, int(img_h * 0.04)) # Dynamic offset

            # Draw Labels
            # Actual Throat
            cx = (hyp_a[0] + hyp_b[0] + third[0]) / 3
            cy = (hyp_a[1] + hyp_b[1] + third[1]) / 3
            centroid = (cx, cy)

            def get_default_offset_pos(p1, p2, cent, offset):
                mx = (p1[0] + p2[0]) / 2
                my = (p1[1] + p2[1]) / 2
                vx = mx - cent[0]
                vy = my - cent[1]
                mag = np.sqrt(vx**2 + vy**2)
                if mag == 0: return (mx, my)
                vx /= mag
                vy /= mag
                return (mx + vx * offset, my + vy * offset)

            # Actual Throat
            def_at = get_default_offset_pos(hyp_a, hyp_b, centroid, offset_val)
            pos_at = get_pos("actual_throat", def_at)
            draw.text(pos_at, meas["actual_throat"]["label"], fill="white", font=font_bold, anchor="mm")
            
            # Leg1
            origin = (ix, iy)
            def_l1 = get_default_offset_pos(origin, hyp_a, centroid, offset_val)
            pos_l1 = get_pos("leg1", def_l1)
            draw.text(pos_l1, meas["leg1"]["label"], fill="white", font=font_bold, anchor="mm")
            
            # Leg2
            def_l2 = get_default_offset_pos(origin, hyp_b, centroid, offset_val)
            pos_l2 = get_pos("leg2", def_l2)
            draw.text(pos_l2, meas["leg2"]["label"], fill="white", font=font_bold, anchor="mm")
            
            # Root Penetration
            def_rp = get_default_offset_pos(third, origin, centroid, offset_val/2)
            pos_rp = get_pos("root_penetration", def_rp)
            draw.text(pos_rp, meas["root_penetration"]["label"], fill="white", font=font_bold, anchor="mm")
            
            # Effective Throat
            mx_et = (third[0] + proj[0]) / 2
            my_et = (third[1] + proj[1]) / 2
            def_et = (mx_et + offset_val/2, my_et + offset_val/2)
            pos_et = get_pos("effective_throat", def_et)
            draw.text(pos_et, meas["effective_throat"]["label"], fill="white", font=font_bold, anchor="mm")

            # Build Status Block for this weld
            block_lines = []
            block_lines.append(f"Weld {i+1}:")
            block_lines.append(f"  Leg1: {meas['leg1']['value']:.2f} cm")
            block_lines.append(f"  Leg2: {meas['leg2']['value']:.2f} cm")
            block_lines.append(f"  Actual Throat: {meas['actual_throat']['value']:.2f} cm")
            block_lines.append(f"  Effective Throat: {meas['effective_throat']['value']:.2f} cm")
            block_lines.append(f"  Penetration: {meas['root_penetration']['value']:.2f} cm")
            weld_blocks.append(block_lines)

        # Draw Status Display (Bottom Right, Horizontal)
        if weld_blocks:
            # Calculate dimensions
            padding = 10
            block_padding = 20 # Space between weld blocks
            
            # Estimate line height
            try:
                line_height = font.getbbox("A")[3] + 5 
            except AttributeError:
                 line_height = font.getsize("A")[1] + 5

            # Calculate width/height for each block
            block_dims = []
            max_block_height = 0
            
            for block in weld_blocks:
                b_w = 0
                b_h = 0
                for line in block:
                    try:
                        w = font.getbbox(line)[2]
                    except AttributeError:
                        w = font.getsize(line)[0]
                    b_w = max(b_w, w)
                    b_h += line_height
                block_dims.append((b_w, b_h))
                max_block_height = max(max_block_height, b_h)

            # Header "Measurements:"
            header_text = "Measurements:"
            try:
                header_w = font_bold.getbbox(header_text)[2]
            except AttributeError:
                header_w = font_bold.getsize(header_text)[0]
            header_h = line_height

            # Total Box Dimensions
            # Width = max(header_w, sum of block widths + padding)
            total_blocks_w = sum(d[0] for d in block_dims) + (len(block_dims) - 1) * block_padding
            box_content_w = max(header_w, total_blocks_w)
            
            box_w = box_content_w + 2 * padding
            box_h = header_h + max_block_height + 2 * padding
            
            # Bottom Right Position
            box_x = img_w - box_w - 20
            box_y = img_h - box_h - 20
            
            # Draw Background
            draw.rectangle([box_x, box_y, box_x + box_w, box_y + box_h], fill="white", outline="black", width=2)
            
            # Draw Header
            draw.text((box_x + padding, box_y + padding), header_text, fill="black", font=font_bold)
            
            # Draw Blocks
            current_x = box_x + padding
            start_y = box_y + padding + header_h
            
            for i, block in enumerate(weld_blocks):
                curr_y = start_y
                for line in block:
                    draw.text((current_x, curr_y), line, fill="black", font=font)
                    curr_y += line_height
                
                # Move x for next block
                current_x += block_dims[i][0] + block_padding

        # Save Dialog
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")])
        if file_path:
            try:
                img_to_save.save(file_path)
                messagebox.showinfo("Success", f"Image saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image: {e}")

    def clear(self):
        if self.selected_item_id:
            tags = self.canvas.gettags(self.selected_item_id)
            
            # Case 1: Selected a Group
            group_tag = next((t for t in tags if t.startswith("group_")), None)
            if group_tag:
                group_idx = int(group_tag.split("_")[1])
                self.canvas.delete(f"group_{group_idx}")
                self.all_groups[group_idx] = None 
                self.group_results[group_idx] = None
                self.info_label.config(text="Selected group cleared.")
            
            # Case 2: Selected a Current Point
            elif "current" in tags:
                # Find index in current_point_items
                # current_point_items = [oval, text, oval, text, ...]
                try:
                    idx = self.current_point_items.index(self.selected_item_id)
                    point_idx = idx // 2
                    
                    # Remove from data
                    if point_idx < len(self.current_points):
                        self.current_points.pop(point_idx)
                    
                    # Redraw remaining current points to fix numbering/IDs
                    # First clear all current point items from canvas
                    for item in self.current_point_items:
                        self.canvas.delete(item)
                    self.current_point_items = []
                    
                    # Redraw
                    for i, pt in enumerate(self.current_points):
                        self._draw_single_point(pt, i + 1)
                        
                    self.info_label.config(text="Selected point cleared.")
                except ValueError:
                    pass # Item not found in list (shouldn't happen if logic is correct)

            self.canvas.delete("highlight")
            self.selected_item_id = None
        else:
            self.canvas.delete("all")
            if self.original_image:
                # Redraw just the image
                self.redraw_all()
                # But we want to clear markings, so reset data
                self.current_points = []
                self.current_point_items = []
                self.all_groups = []
                self.group_items = []
                self.group_results = []
            else:
                self.current_points = []
                self.all_groups = []
                self.group_results = []
            
            self.info_label.config(text="All markings cleared.")

    def delete_all(self):
        self.canvas.delete("all")
        self.image_path = None
        self.original_image = None
        self.tk_image = None
        self.current_points = []
        self.current_point_items = []
        self.all_groups = []
        self.group_results = []
        self.group_items = []
        self.selected_item_id = None
        self.scale_ratio = 1.0
        self.pixels_per_cm = None
        self.info_label.config(text="Upload an image\nto start.")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageAnalysisApp(root)
    root.mainloop()
    