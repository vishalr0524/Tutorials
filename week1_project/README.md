# Live Shape Detection with OpenCV

This repository contains a simple real-time **Shape Detector** implemented using **OpenCV** and **NumPy**. The program utilizes a webcam feed to detect, classify, and visualize basic geometric shapes (triangles, squares, rectangles, and circles) by analyzing their contours and hierarchical relationships.

-----

## Features

  * **Real-Time Processing:** Processes video frames directly from the webcam.
  * **Shape Classification:** Detects and labels **Triangle**, **Square**, **Rectangle**, and **Circle** shapes.
  * **Contour Hierarchy Visualization:** Draws bounding boxes and shape labels for the detected objects, using different colors to indicate the hierarchical depth (parent-child relationship) of nested shapes.
  * **Core CV Techniques:** Demonstrates the use of color conversion, blurring, Canny edge detection, contour finding, and contour approximation.

-----

## Requirements

  * Python 3.x
  * OpenCV (`cv2`)
  * NumPy

You can install the necessary libraries using pip:

```bash
pip install opencv-python numpy
```

-----

## How to Run

1.  **Ensure a Webcam is Available:** The script is set to use the default camera device (`cv2.VideoCapture(0)`).
2.  **Save the Code:** Save the provided code as a Python file (e.g., `day05_project.py`).
3.  **Execute the Script:**
    ```bash
    python day05_project.py
    ```

A window titled "Shapes" will open, displaying the live video feed with detected shapes annotated with colored bounding boxes and labels.

4.  **Exit:** Press the **'q'** key to close the window and stop the script.

-----

## Core Logic Overview

The script processes each video frame through the following steps:

1.  **Capture & Preprocessing:** Reads a frame (`img`), converts it to **Grayscale** (`gray`), and applies a **Gaussian Blur** (`blurred`) to reduce noise.
2.  **Edge Detection:** Uses the **Canny edge detector** (`edges`) to find sharp intensity changes.
3.  **Contour Finding:** Finds contours (`contours`) from the edges, utilizing the **`cv2.RETR_TREE`** mode to build a full hierarchy of nested contours.
4.  **Hierarchy Analysis:** The hierarchy information is parsed to determine parent-child relationships, which is used for the visualization.
5.  **Shape Detection (`detect_shape` function):**
      * Approximates the contour's polygon using **`cv2.approxPolyDP`**.
      * The number of vertices in the approximation is used for simple classification:
          * **3 vertices:** Triangle
          * **4 vertices:** Checks aspect ratio (`bw / bh`) to distinguish **Square** (aspect ratio close to 1) from **Rectangle**.
      * Calculates **Circularity** based on area and perimeter for **Circle** detection (circularity ratio $> 0.85$).
6.  **Visualization (`draw_hierarchy` function):** Iterates through the top-level (external) contours and recursively draws the bounding box and label for the shape and its children. The color of the bounding box changes based on the nesting **depth** (`color = (0, 255 // (depth + 1), 255 // (depth + 1))`).

-----

## Week 1 Learning Document: Image Processing Fundamentals

This document summarizes the key concepts and techniques learned in the first week of image processing, covering image basics, edge detection, filtering, and contour analysis.

### Image Fundamentals

  * **Images as Arrays:** An image is essentially a **2D or 3D NumPy array** where each element (or a set of elements) represents a **pixel**.
  * **Color Spaces:**
      * **BGR (Blue-Green-Red):** The default color ordering used by OpenCV when reading an image.
      * **RGB (Red-Green-Blue):** A more common convention for displaying or working with image libraries like Matplotlib.
      * **Grayscale:** A single channel image where pixel intensity represents brightness (0=black, 255=white).
      * **HSV (Hue-Saturation-Value):** A color space often preferred for color-based object tracking or segmentation.
  * **Basic Operations:** Learned how to **load, display, and save** images, perform **color space conversions** (BGR to RGB, Grayscale, HSV), and **split/merge channels**.
  * **Image Slicing (ROI):** Manipulating specific parts of an image (Regions of Interest) using **NumPy array slicing** (e.g., `image[y1:y2, x1:x2]`).

### Edge Detection

  * **Concept:** Edges are points in an image where the image **brightness changes sharply**. Edge detection relies on calculating the **image gradient** (derivative) to find these changes.
  * **Sobel & Scharr:** Utilize kernel convolution to approximate the gradient of the image intensity function in the horizontal ($x$) and vertical ($y$) directions. The **magnitude** of these gradients indicates the strength of the edge.
  * **Canny:** A multi-stage optimal edge detection algorithm that involves:
    1.  **Noise Reduction** (Gaussian Blur).
    2.  **Finding Intensity Gradients** (Sobel).
    3.  **Non-Maximum Suppression** (thinning edges).
    4.  **Hysteresis Thresholding** (final edge selection using a high and a low threshold).

### Image Filtering & Noise Removal

  * **Convolution and Kernels:** Applying a small matrix (**kernel**) over the image to perform local operations.
  * **Noise Removal (Blurring):**
      * **Box & Mean Filter (`cv2.blur`):** Simplest smoothing filters, averaging pixel values in a neighborhood. Effective for simple, uniform noise.
      * **Gaussian Filter (`cv2.GaussianBlur`):** Uses a weighted average (Gaussian distribution) to reduce noise, preserving more edge information than mean filtering.
      * **Median Filter (`cv2.medianBlur`):** Replaces the center pixel with the median value of its neighborhood. Highly effective for removing **Salt-and-Pepper noise**.
  * **Sharpening:** Uses kernels that emphasize the difference between a pixel and its neighbors (e.g., a high-pass filter or an unsharp mask approximation) to enhance details and edges.

### Contours & Shape Analysis

  * **Thresholding:** Converting a grayscale image to a binary image (black and white) by setting a threshold value. Used to separate the object (foreground) from the background. Techniques include **Global, Adaptive, and Otsu** thresholding.
  * **Contour Detection (`cv2.findContours`):** Finds the curves joining all continuous points along the boundary of an object in a binary image.
  * **Contour Hierarchy:** Describes the relationship (parent/child) between nested contours. Retrieval modes like **`cv2.RETR_TREE`** are used to fully map this hierarchy.
  * **Shape Properties:** Calculating attributes of a contour for classification:
      * **Area (`cv2.contourArea`)**
      * **Perimeter (`cv2.arcLength`)**
      * **Bounding Box (`cv2.boundingRect`)**
      * **Contour Approximation (`cv2.approxPolyDP`):** Simplifies the contour shape by reducing the number of vertices while preserving the overall shape, which is crucial for identifying geometric shapes like triangles and rectangles.