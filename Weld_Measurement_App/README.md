# Weld Inspection App

A Python-based desktop application for weld inspection, measurement, and reporting.

## Prerequisites

- Python 3.x
- `pip` (Python package installer)

## Installation

1.  **Clone or Download** the repository to your local machine.
2.  **Install Dependencies**:
    Open a terminal or command prompt in the project folder and run:
    ```bash
    sudo apt update
    sudo apt install libxcb-cursor0
    pip install PyQt6 xlsxwriter
    ```

## How to Run

Run the main script using Python:

```bash
python main.py
```

## Usage Guide

1.  **Load Image**:
    *   Click **"Import Image"** or drag and drop an image file into the main view.
    *   Supported formats: JPG, PNG, BMP.

2.  **Navigation**:
    *   **Zoom**: Use the mouse wheel to zoom in/out.
    *   **Pan**: Hold the middle mouse button (or hold Spacebar + Left Click) and drag to move the image.

3.  **Calibration**:
    *   Click **"Calibration"** in the toolbar.
    *   Draw a line on a known reference object in the image.
    *   Enter the real-world length (e.g., 10 mm) to calibrate pixels to units.

4.  **Measurements**:
    *   Select a tool from the toolbox (e.g., **Line**, **Point**).
    *   **Draw**: Click and drag on the image to measure dimensions.
        *   **Note**: When using points for calculation, the order matters:
            *   **P1**: Top marks
            *   **P2**: Bottom point
            *   **P3**: Penetrate Point
    *   **Select**: Click on an existing shape to select it.
    *   **Delete**: Press `Delete` key to remove selected shapes.

5.  **Export Report**:
    *   Click **"Export Excel"** to generate a report.
    *   The report includes the image with annotations and a table of measurements.
