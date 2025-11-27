from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QDateEdit, QFormLayout
from PyQt6.QtCore import QDate

class SaveDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Save Inspection Report")
        self.setModal(True)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Form Layout for inputs
        form_layout = QFormLayout()

        self.part_name_edit = QLineEdit()
        form_layout.addRow("Part Name:", self.part_name_edit)

        self.part_number_edit = QLineEdit()
        form_layout.addRow("Part Number:", self.part_number_edit)

        self.measure_unit_edit = QLineEdit()
        form_layout.addRow("Measure Unit:", self.measure_unit_edit)

        self.date_edit = QDateEdit(QDate.currentDate())
        self.date_edit.setCalendarPopup(True)
        form_layout.addRow("Date:", self.date_edit)

        self.done_by_edit = QLineEdit()
        form_layout.addRow("Done By:", self.done_by_edit)

        layout.addLayout(form_layout)

        # Buttons
        btn_layout = QHBoxLayout()
        self.save_btn = QPushButton("Save")
        self.save_btn.clicked.connect(self.accept)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)

        btn_layout.addWidget(self.save_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

    def get_data(self):
        return {
            "Part Name": self.part_name_edit.text(),
            "Part Number": self.part_number_edit.text(),
            "Measure Unit": self.measure_unit_edit.text(),
            "Date": self.date_edit.date().toString("yyyy-MM-dd"),
            "Done By": self.done_by_edit.text()
        }
