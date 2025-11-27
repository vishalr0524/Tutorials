from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton
from PyQt6.QtCore import pyqtSignal

class Controls(QWidget):
    delete_requested = pyqtSignal()
    help_requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        
        self.delete_btn = QPushButton("Delete Image")
        self.delete_btn.clicked.connect(self.delete_requested.emit)
        layout.addWidget(self.delete_btn)

        self.help_btn = QPushButton("Help")
        self.help_btn.clicked.connect(self.help_requested.emit)
        layout.addWidget(self.help_btn)

        layout.addStretch()
        self.setLayout(layout)
