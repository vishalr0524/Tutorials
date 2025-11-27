from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsRectItem, QGraphicsSimpleTextItem, QSizePolicy, QGraphicsItem
from PyQt6.QtCore import Qt, pyqtSignal, QPointF, QRectF
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QBrush
from enum import Enum
from utils.geometry import get_line_intersection, distance, midpoint
from ui.dialogs import ColorSelectionDialog, ConfirmDialog

class DrawingMode(Enum):
    NONE = 0
    LINE = 1
    POINT = 2
    LABEL = 3

class ImageViewer(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setBackgroundBrush(Qt.GlobalColor.lightGray)
        self.setFrameShape(QGraphicsView.Shape.NoFrame)
        
        self.setAcceptDrops(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.current_mode = DrawingMode.NONE
        self.current_item = None
        self.start_point = None
        self.is_sticky = False
        self.calibration_factor = None # pixels per unit (cm)
        self.current_line_color = QColor(Qt.GlobalColor.red)

        self.point_counters = {} 
        
        self.calculation_results = [] 
        self.region_count = 0 

    def display_image(self, path):
        pixmap = QPixmap(path)
        if not pixmap.isNull():
            self.pixmap_item.setPixmap(pixmap)
            self.scene.setSceneRect(self.pixmap_item.boundingRect())
            self.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        else:
            self.clear_image()

    def clear_image(self):
        self.pixmap_item.setPixmap(QPixmap())
        self.scene.setSceneRect(0, 0, 0, 0)
        # Clear all items when image is deleted
        for item in self.scene.items():
            if item != self.pixmap_item:
                self.scene.removeItem(item)
        self.point_counters = {}
        self.calculation_results = []
        self.region_count = 0
        self.calibration_factor = None

    def clear_markings(self):
        # Smart Clear: Remove selected items if any, otherwise remove all markings
        selected_items = self.scene.selectedItems()
        items_to_remove = set()

        if selected_items:
            for item in selected_items:
                if item != self.pixmap_item:
                    items_to_remove.add(item)
        else:
            # Clear all drawn items, keep the image
            for item in self.scene.items():
                if item != self.pixmap_item:
                    items_to_remove.add(item)
            for item in self.scene.items():
                if item != self.pixmap_item:
                    items_to_remove.add(item)
            self.point_counters = {}
            self.calculation_results = []
            self.region_count = 0
            self.region_count = 0

        # Check for dependent intersections
        # If we are removing a line, we must also remove any intersections that depend on it
        # Iterate through all items to find intersections
        all_items = self.scene.items()
        for item in all_items:
            if item.data(0) == "intersection":
                line1 = item.data(3)
                line2 = item.data(4)
                if line1 in items_to_remove or line2 in items_to_remove:
                    items_to_remove.add(item)
                    # Also remove the label associated with the intersection?
                    # The label is a separate item, but we didn't link it explicitly.
                    # Ideally we should group them or link them.
                    # For now, let's just remove the intersection dot. 
                    # If the text label is separate, it might linger.
                    # Let's try to find the text label nearby or link it in check_intersections.
                    # IMPROVEMENT: Link text label to intersection item.
                    label_item = item.data(5)
                    if label_item:
                        items_to_remove.add(label_item)
        
        # Also check for linked labels on points (data(5))
        for item in list(items_to_remove): # Use list to avoid runtime error if we modify set? No, iterating copy is safer
             label_item = item.data(5)
             if label_item:
                 items_to_remove.add(label_item)

        for item in items_to_remove:
            self.scene.removeItem(item)
    
    def zoom_in(self):
        self.scale(1.2, 1.2)

    def zoom_out(self):
        self.scale(1 / 1.2, 1 / 1.2)

    def set_mode(self, mode, sticky=False):
        self.current_mode = mode
        self.is_sticky = sticky
        self.current_item = None
        self.start_point = None
        if mode == DrawingMode.NONE:
            self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        else:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)

    def mousePressEvent(self, event):
        if self.current_mode == DrawingMode.NONE:
            super().mousePressEvent(event)
            return

        # Constraint: Check if image is loaded
        if self.pixmap_item.pixmap().isNull():
            return

        sp = self.mapToScene(event.pos())
        
        if self.current_mode == DrawingMode.LINE:
            self.start_point = sp
            self.current_item = None
        
        elif self.current_mode == DrawingMode.LABEL:
            self.start_point = sp
            self.current_item = QGraphicsRectItem(sp.x(), sp.y(), 0, 0)
            self.current_item.setPen(QPen(Qt.GlobalColor.yellow, 2, Qt.PenStyle.DashLine))
            self.scene.addItem(self.current_item)

        elif self.current_mode == DrawingMode.POINT:
            # Color Selection
            dialog = ColorSelectionDialog(self)
            if dialog.exec() == 1:
                color = dialog.get_color()
            else:
                color = QColor(Qt.GlobalColor.red) # Default
            
            # Naming Logic
            color_key = color.name()
            if color_key not in self.point_counters:
                self.point_counters[color_key] = 1
            
            count = self.point_counters[color_key]
            name = f"P{count}"
            
            # Draw Point
            r = 5
            self.current_item = QGraphicsEllipseItem(sp.x() - r, sp.y() - r, r*2, r*2)
            self.current_item.setBrush(QBrush(color))
            self.current_item.setPen(QPen(color))
            self.current_item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
            self.current_item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
            self.current_item.setData(0, "point")
            self.current_item.setData(1, name)
            self.current_item.setData(2, color)
            self.scene.addItem(self.current_item)
            
            # Draw Label
            text = QGraphicsSimpleTextItem(name)
            text.setPos(sp.x() + r, sp.y() - r*2)
            text.setBrush(QBrush(color))
            self.scene.addItem(text)
            
            self.current_item.setData(5, text) # Link label to point

            # Cycle counter: 1->2->3->1
            self.point_counters[color_key] = (count % 3) + 1
            
            if not self.is_sticky:
                self.set_mode(DrawingMode.NONE)

    def mouseMoveEvent(self, event):
        if self.current_mode == DrawingMode.NONE:
            super().mouseMoveEvent(event)
            return

        if self.current_mode == DrawingMode.LINE and self.start_point:
            ep = self.mapToScene(event.pos())
            
            if self.current_item is None:
                self.current_item = QGraphicsLineItem(self.start_point.x(), self.start_point.y(), ep.x(), ep.y())
                self.current_item.setPen(QPen(Qt.GlobalColor.red, 3))
                self.current_item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
                self.current_item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
                self.current_item.setData(0, "line")
                self.scene.addItem(self.current_item)
            else:
                line = self.current_item.line()
                line.setP2(ep)
                self.current_item.setLine(line)
        
        elif self.current_mode == DrawingMode.LABEL and self.current_item and self.start_point:
            ep = self.mapToScene(event.pos())
            rect = QRectF(self.start_point, ep).normalized()
            self.current_item.setRect(rect)

    def mouseReleaseEvent(self, event):
        if self.current_mode == DrawingMode.NONE:
            super().mouseReleaseEvent(event)
            return

        if self.current_mode == DrawingMode.LINE:
            if self.current_item:
                # Check intersections
                self.check_intersections(self.current_item)
            
            self.current_item = None
            self.start_point = None
            if not self.is_sticky:
                self.set_mode(DrawingMode.NONE)
        
        elif self.current_mode == DrawingMode.LABEL:
            if self.current_item:
                rect = self.current_item.rect()
                self.scene.removeItem(self.current_item) # Remove selection box
                self.current_item = None
                self.start_point = None
                
                dialog = ConfirmDialog("Confirm Label", "Calculate measurements for this region?", self)
                if dialog.exec() == 1:
                    self.perform_calculations(rect)
                
                self.set_mode(DrawingMode.NONE)

    def check_intersections(self, new_line_item):
        new_line = new_line_item.line()
        intersection_count = 0
        
        # Count existing intersections to name I1, I2...
        for item in self.scene.items():
            if item.data(0) == "intersection":
                intersection_count += 1
        
        for item in self.scene.items():
            if item != new_line_item and item.data(0) == "line":
                other_line = item.line()
                pt = get_line_intersection(new_line, other_line)
                if pt:
                    intersection_count += 1
                    name = f"I{intersection_count}"
                    
                    # Mark Intersection
                    r = 4
                    i_item = QGraphicsEllipseItem(pt.x() - r, pt.y() - r, r*2, r*2)
                    i_item.setBrush(QBrush(Qt.GlobalColor.black))
                    i_item.setPen(QPen(Qt.GlobalColor.black))
                    i_item.setData(0, "intersection")
                    i_item.setData(1, name)
                    i_item.setData(3, new_line_item) # Dependency 1
                    i_item.setData(4, item)          # Dependency 2 (other_line item)
                    self.scene.addItem(i_item)
                    
                    # Label
                    text = QGraphicsSimpleTextItem(name)
                    text.setPos(pt.x() + r, pt.y() - r*2)
                    text.setBrush(QBrush(Qt.GlobalColor.black))
                    self.scene.addItem(text)
                    
                    i_item.setData(5, text) # Link label to intersection

    def perform_calculations(self, rect):
        # Find items in rect
        items_in_rect = self.scene.items(rect)
        
        points = []
        intersections = []
        
        for item in items_in_rect:
            if item.data(0) == "point":
                points.append(item)
            elif item.data(0) == "intersection":
                intersections.append(item)
        
        # Need 1 Intersection and 3 Points of same color
        if len(intersections) < 1:
            return # Not enough intersections
        
        intersection = intersections[0] # Take first one found
        i_pt = intersection.rect().center()
        
        # Group points by color
        points_by_color = {}
        for p in points:
            color = p.data(2)
            key = color.name()
            if key not in points_by_color:
                points_by_color[key] = []
            points_by_color[key].append(p)
        
        target_points = None
        for key, pts in points_by_color.items():
            if len(pts) == 3:
                # Sort by name P1, P2, P3
                pts.sort(key=lambda x: x.data(1))
                if pts[0].data(1) == "P1" and pts[1].data(1) == "P2" and pts[2].data(1) == "P3":
                    target_points = pts
                    break
        
        if not target_points:
            return # No valid P1, P2, P3 set found
        
        p1 = target_points[0].rect().center()
        p2 = target_points[1].rect().center()
        p3 = target_points[2].rect().center()
        
        # Draw P1-P2 Line (Black)
        p1_p2_line = QGraphicsLineItem(p1.x(), p1.y(), p2.x(), p2.y())
        p1_p2_line.setPen(QPen(Qt.GlobalColor.black, 2))
        self.scene.addItem(p1_p2_line)

        # Increment region count
        self.region_count += 1
        region_title = f"Region {self.region_count}"
        
        current_measurements = []

        # Calculations
        # Leg 1: P1 to I
        val1 = self.draw_measurement(p1, i_pt, "Leg 1")
        current_measurements.append({
            "name": "Leg 1",
            "value": val1,
            "pass_fail": "",
            "remarks": f"{target_points[0].data(1)} - {intersection.data(1)}"
        })
        
        # Leg 2: I to P2
        val2 = self.draw_measurement(i_pt, p2, "Leg 2")
        current_measurements.append({
            "name": "Leg 2",
            "value": val2,
            "pass_fail": "",
            "remarks": f"{intersection.data(1)} - {target_points[1].data(1)}"
        })
        
        # Midpoint of P1-P2
        mid_p1_p2 = midpoint(p1, p2)
        
        # Effective Throat: P3 to Midpoint
        val3 = self.draw_measurement(p3, mid_p1_p2, "Effective Throat")
        current_measurements.append({
            "name": "Effective Throat",
            "value": val3,
            "pass_fail": "",
            "remarks": f"{target_points[2].data(1)} - Mid({target_points[0].data(1)}-{target_points[1].data(1)})"
        })
        
        # Design Throat: I to Midpoint
        val4 = self.draw_measurement(i_pt, mid_p1_p2, "Design Throat")
        current_measurements.append({
            "name": "Design Throat",
            "value": val4,
            "pass_fail": "",
            "remarks": f"{intersection.data(1)} - Mid({target_points[0].data(1)}-{target_points[1].data(1)})"
        })
        
        # Root Penetration: P3 to I
        val5 = self.draw_measurement(p3, i_pt, "Root Penetration")
        current_measurements.append({
            "name": "Root Penetration",
            "value": val5,
            "pass_fail": "",
            "remarks": f"{target_points[2].data(1)} - {intersection.data(1)}"
        })
        
        self.calculation_results.append({
            "title": region_title,
            "measurements": current_measurements
        })

    def draw_measurement(self, p1, p2, label):
        line = QGraphicsLineItem(p1.x(), p1.y(), p2.x(), p2.y())
        line.setPen(QPen(Qt.GlobalColor.blue, 2, Qt.PenStyle.DotLine))
        self.scene.addItem(line)
        
        # Calculate length
        length_px = distance(p1, p2)
        text_content = label
        if self.calibration_factor:
            length_unit = length_px / self.calibration_factor
            text_content += f": {length_unit:.2f} cm"
        
        text = QGraphicsSimpleTextItem(text_content)
        mid = midpoint(p1, p2)
        text.setPos(mid)
        text.setBrush(QBrush(Qt.GlobalColor.blue))
        text.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        text.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.scene.addItem(text)
        
        return f"{length_unit:.2f} cm" if self.calibration_factor else f"{length_px:.2f} px"

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        if files:
            self.display_image(files[0])

    def get_snapshot(self):
        # Create a QPixmap from the scene
        rect = self.scene.sceneRect()
        pixmap = QPixmap(int(rect.width()), int(rect.height()))
        pixmap.fill(Qt.GlobalColor.white)
        
        painter = QPainter(pixmap)
        self.scene.render(painter)
        painter.end()
        return pixmap
