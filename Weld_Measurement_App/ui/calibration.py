from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsEllipseItem
from PyQt6.QtCore import Qt, pyqtSignal, QPointF
from PyQt6.QtGui import QPixmap, QPen, QBrush
from utils.geometry import distance

class CalibrationWidget(QWidget):
    completed = pyqtSignal(float) # Emits pixels per unit
    cancelled = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.points = []
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        
        layout.addWidget(QLabel("Calibration Mode: Click 2 points (C1, C2) to define 1 cm."))
        
        self.view = QGraphicsView()
        self.scene = QGraphicsScene()
        self.view.setScene(self.scene)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        
        # Simple view setup
        self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        layout.addWidget(self.view)

        btn_layout = QHBoxLayout()
        self.confirm_btn = QPushButton("Confirm Calibration")
        self.confirm_btn.setEnabled(False)
        self.confirm_btn.clicked.connect(self.calculate_calibration)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.cancelled.emit)
        
        btn_layout.addWidget(self.confirm_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)
        
        self.setLayout(layout)
        
        # Event filter or subclassing view? Subclassing is cleaner for events.
        # But for speed, let's use event filter or just override view's mouse press here if possible?
        # Actually, let's just monkey-patch or use a simple inner class logic.
        # Better: Assign a custom mouse press handler to the scene or view.
        self.scene.mousePressEvent = self.scene_mouse_press

    def set_image(self, pixmap):
        self.pixmap_item.setPixmap(pixmap)
        self.scene.setSceneRect(self.pixmap_item.boundingRect())
        self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.points = []
        self.confirm_btn.setEnabled(False)
        # Clear previous points
        for item in self.scene.items():
            if item != self.pixmap_item:
                self.scene.removeItem(item)

    def scene_mouse_press(self, event):
        if len(self.points) >= 2:
            return
        
        sp = event.scenePos()
        self.points.append(sp)
        
        # Draw point
        r = 5
        item = QGraphicsEllipseItem(sp.x() - r, sp.y() - r, r*2, r*2)
        item.setBrush(QBrush(Qt.GlobalColor.blue))
        self.scene.addItem(item)
        
        if len(self.points) == 2:
            self.confirm_btn.setEnabled(True)

    def calculate_calibration(self):
        if len(self.points) != 2:
            return
        
        dist_pixels = distance(self.points[0], self.points[1])
        # User requirement: "divide by default 1 cm real pixel value hardcoded"
        # So factor = pixels / 1 cm = pixels per cm
        factor = dist_pixels / 1.0 
        self.completed.emit(factor)
