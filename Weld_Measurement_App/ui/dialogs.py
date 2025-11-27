from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QRadioButton, QPushButton, QLabel, QButtonGroup, QMessageBox
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor

class ColorSelectionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Point Color")
        self.selected_color = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Select a color for the point:"))

        self.color_group = QButtonGroup(self)
        colors = [
            ("Red", QColor(Qt.GlobalColor.red)),
            ("Green", QColor(Qt.GlobalColor.green)),
            ("Blue", QColor(Qt.GlobalColor.blue)),
            ("Cyan", QColor(Qt.GlobalColor.cyan)),
            ("Magenta", QColor(Qt.GlobalColor.magenta)),
            ("Yellow", QColor(Qt.GlobalColor.yellow)),
            ("Black", QColor(Qt.GlobalColor.black)),
            ("White", QColor(Qt.GlobalColor.white)),
            ("Gray", QColor(Qt.GlobalColor.gray)),
            ("Dark Red", QColor(139, 0, 0))
        ]

        # Grid layout for colors might be better, but VBox is simple for now
        for i, (name, color) in enumerate(colors):
            rb = QRadioButton(name)
            rb.setProperty("color", color)
            self.color_group.addButton(rb, i)
            layout.addWidget(rb)
            if i == 0: rb.setChecked(True)

        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

    def get_color(self):
        if self.result() == QDialog.DialogCode.Accepted:
            button = self.color_group.checkedButton()
            if button:
                return button.property("color")
        return None

class ConfirmDialog(QDialog):
    def __init__(self, title, message, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        layout = QVBoxLayout()
        layout.addWidget(QLabel(message))
        
        btn_layout = QHBoxLayout()
        yes_btn = QPushButton("Yes")
        yes_btn.clicked.connect(self.accept)
        no_btn = QPushButton("No")
        no_btn.clicked.connect(self.reject)
        btn_layout.addWidget(yes_btn)
        btn_layout.addWidget(no_btn)
        layout.addLayout(btn_layout)
        self.setLayout(layout)
