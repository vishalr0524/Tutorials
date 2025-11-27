from PyQt6.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QStackedWidget
from ui.toolbox import Toolbox
from ui.viewer import ImageViewer, DrawingMode
from ui.controls import Controls
from ui.calibration import CalibrationWidget
from ui.save_dialog import SaveDialog
from utils.excel_report import generate_excel_report
from PyQt6.QtWidgets import QMessageBox, QFileDialog

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Weld Inspection App")
        self.resize(1200, 800)

        # Central Widget is now a StackedWidget for navigation
        self.central_stack = QStackedWidget()
        self.setCentralWidget(self.central_stack)

        # Page 0: Home (Inspection)
        self.home_widget = QWidget()
        self.init_home_ui()
        self.central_stack.addWidget(self.home_widget)

        # Page 1: Calibration
        self.calibration_widget = CalibrationWidget()
        self.calibration_widget.completed.connect(self.on_calibration_completed)
        self.calibration_widget.cancelled.connect(self.show_home)
        self.central_stack.addWidget(self.calibration_widget)

        self.current_image_path = None

    def init_home_ui(self):
        layout = QHBoxLayout()
        
        # Left: Toolbox
        self.toolbox = Toolbox()
        self.toolbox.upload_requested.connect(self.open_file_dialog)
        self.toolbox.zoom_in_requested.connect(self.viewer_zoom_in)
        self.toolbox.zoom_out_requested.connect(self.viewer_zoom_out)
        
        # Connect Drawing Signals
        self.toolbox.line_tool_requested.connect(lambda: self.viewer.set_mode(DrawingMode.LINE, sticky=False))
        self.toolbox.line_tool_sticky.connect(lambda: self.viewer.set_mode(DrawingMode.LINE, sticky=True))
        self.toolbox.point_tool_requested.connect(lambda: self.viewer.set_mode(DrawingMode.POINT, sticky=False))
        self.toolbox.point_tool_sticky.connect(lambda: self.viewer.set_mode(DrawingMode.POINT, sticky=True))
        self.toolbox.compute_tool_requested.connect(lambda: self.viewer.set_mode(DrawingMode.LABEL))
        self.toolbox.clear_requested.connect(self.viewer_clear_markings)
        
        # Navigation Signals
        self.toolbox.home_requested.connect(self.show_home)
        self.toolbox.calibrate_requested.connect(self.start_calibration)
        self.toolbox.save_requested.connect(self.save_report)

        layout.addWidget(self.toolbox, 1) # Stretch factor 1

        # Center: Image Viewer
        self.viewer = ImageViewer()
        layout.addWidget(self.viewer, 6) # Stretch factor 6

        # Right: Controls
        self.controls = Controls()
        self.controls.delete_requested.connect(self.viewer.clear_image)
        self.controls.help_requested.connect(self.show_help)
        layout.addWidget(self.controls, 1) # Stretch factor 1
        
        self.home_widget.setLayout(layout)

    def viewer_zoom_in(self):
        self.viewer.zoom_in()

    def viewer_zoom_out(self):
        self.viewer.zoom_out()

    def viewer_clear_markings(self):
        self.viewer.clear_markings()

    def show_home(self):
        self.central_stack.setCurrentIndex(0)

    def start_calibration(self):
        if self.current_image_path:
            # Pass current image to calibration widget
            # We can get pixmap from viewer or reload from path
            # Reloading is safer to ensure clean state
            from PyQt6.QtGui import QPixmap
            self.calibration_widget.set_image(QPixmap(self.current_image_path))
            self.central_stack.setCurrentIndex(1)

    def on_calibration_completed(self, factor):
        self.viewer.calibration_factor = factor
        self.show_home()

    def open_file_dialog(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.xpm *.jpg *.bmp)")
        if file_name:
            self.current_image_path = file_name
            self.viewer.display_image(file_name)

    def save_report(self):
        # 1. Check if image is loaded
        if not self.current_image_path:
            QMessageBox.warning(self, "Warning", "No image loaded.")
            return

        # 2. Check if calculations exist
        if not self.viewer.calculation_results:
            QMessageBox.warning(self, "Warning", "No calculations performed. Please label a region first.")
            return

        # 3. Open Save Details Dialog
        dialog = SaveDialog(self)
        if dialog.exec() == 1:
            metadata = dialog.get_data()
            
            # 4. Get Save Path
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Report", f"{metadata['Part Name']}.xlsx", "Excel Files (*.xlsx)")
            
            if file_path:
                # 5. Generate Excel
                snapshot = self.viewer.get_snapshot()
                success = generate_excel_report(file_path, metadata, snapshot, self.viewer.calculation_results)
                
                if success:
                    QMessageBox.information(self, "Success", "Report saved successfully!")
                else:
                    QMessageBox.critical(self, "Error", "Failed to save report.")

    def show_help(self):
        help_text = """
        <b>How to Use:</b><br>
        1. <b>Upload Image</b>: Click 'Upload Image' to load a weld image.<br>
        2. <b>Calibrate</b>: Click 'Calibrate', mark two points of known distance, and enter the value.<br>
        3. <b>Draw</b>: Use 'Line' and 'Point' tools to mark the weld.<br>
        4. <b>Compute</b>: Click 'Compute', then drag to select a region containing points and intersections.<br>
        5. <b>Save</b>: Click 'Save' to generate an Excel report with calculations.
        """
        QMessageBox.information(self, "Help", help_text)
