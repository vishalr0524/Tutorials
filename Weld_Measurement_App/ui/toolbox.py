from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel
from PyQt6.QtCore import pyqtSignal, QTimer

class StickyButton(QPushButton):
    doubleClicked = pyqtSignal()

    def mouseDoubleClickEvent(self, event):
        self.doubleClicked.emit()
        super().mouseDoubleClickEvent(event)

class Toolbox(QWidget):
    upload_requested = pyqtSignal()
    zoom_in_requested = pyqtSignal()
    zoom_out_requested = pyqtSignal()
    line_tool_requested = pyqtSignal()
    line_tool_sticky = pyqtSignal()
    point_tool_requested = pyqtSignal()
    point_tool_sticky = pyqtSignal()
    point_tool_requested = pyqtSignal()
    point_tool_sticky = pyqtSignal()
    compute_tool_requested = pyqtSignal()
    clear_requested = pyqtSignal()
    home_requested = pyqtSignal()
    calibrate_requested = pyqtSignal()
    save_requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        
        title = QLabel("Toolbox")
        layout.addWidget(title)

        # Navigation
        self.home_btn = QPushButton("Home")
        self.home_btn.clicked.connect(self.home_requested.emit)
        layout.addWidget(self.home_btn)

        self.upload_btn = QPushButton("Upload Image")
        self.upload_btn.clicked.connect(self.upload_requested.emit)
        layout.addWidget(self.upload_btn)

        self.zoom_in_btn = QPushButton("Zoom In")
        self.zoom_in_btn.clicked.connect(self.zoom_in_requested.emit)
        layout.addWidget(self.zoom_in_btn)

        self.zoom_out_btn = QPushButton("Zoom Out")
        self.zoom_out_btn.clicked.connect(self.zoom_out_requested.emit)
        layout.addWidget(self.zoom_out_btn)

        # Drawing Tools
        
        # Drawing Tools
        
        self.line_btn = StickyButton("Line")
        self.line_btn.clicked.connect(self.line_tool_requested.emit)
        self.line_btn.doubleClicked.connect(self.line_tool_sticky.emit)
        layout.addWidget(self.line_btn)

        self.point_btn = StickyButton("Point")
        self.point_btn.clicked.connect(self.point_tool_requested.emit)
        self.point_btn.doubleClicked.connect(self.point_tool_sticky.emit)
        layout.addWidget(self.point_btn)

        self.compute_btn = QPushButton("Compute")
        self.compute_btn.clicked.connect(self.compute_tool_requested.emit)
        layout.addWidget(self.compute_btn)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear_requested.emit)
        layout.addWidget(self.clear_btn)
        
        self.calibrate_btn = QPushButton("Calibrate")
        self.calibrate_btn.clicked.connect(self.calibrate_requested.emit)
        layout.addWidget(self.calibrate_btn)

        self.save_btn = QPushButton("Save")
        self.save_btn.clicked.connect(self.save_requested.emit)
        layout.addWidget(self.save_btn)
        
        layout.addStretch()
        
        self.setLayout(layout)
