# Object Localization using SIFT and Homography

This project demonstrates **Object Localization**—finding a known object within a larger scene—using advanced computer vision techniques: **SIFT (Scale-Invariant Feature Transform)** for feature extraction and **Homography** for geometric verification.

-----

## Features

  * **SIFT Feature Extraction:** Detects and computes distinctive, scale-invariant, and rotation-invariant keypoints and descriptors for both the object and the scene image.
  * **FLANN Matching:** Uses the **Fast Library for Approximate Nearest Neighbors (FLANN)**-based matcher, which is highly optimized for high-dimensional descriptors like SIFT.
  * **Lowe's Ratio Test:** Refines the matches by applying a ratio test (e.g., $m.distance < 0.75 \times n.distance$) to filter out ambiguous or poor matches, ensuring only the most reliable keypoint correspondences are kept.
  * **Homography Calculation:** Computes the **Homography matrix** ($\mathbf{H}$) using the RANSAC algorithm to find the perspective transformation that maps the object's points to the scene's points.
  * **Object Localization:** Transforms the corners of the original object image using $\mathbf{H}$ and draws a bounding polygon around the detected object in the scene image.

-----

## Requirements

  * Python 3.x
  * OpenCV (`cv2`)
  * NumPy (`numpy`)
  * Matplotlib (`matplotlib`)

You can install the necessary libraries using pip:

```bash
pip install opencv-python numpy matplotlib
```

-----

## How to Run

1.  **Prepare Images:** Ensure you have two image files:
      * `object.jpg`: The image of the item you want to find (the "query" image).
      * `scene.jpg`: The larger image containing the object (the "training" image or "scene").
2.  **Save the Code:** Save the provided Python code as a file (e.g., `object_detector.py`).
3.  **Execute the Script:**
    ```bash
    python object_detector.py
    ```

The script will display a plot titled "Object detected in Scene" with the original object's boundaries drawn as a green polygon on the scene image.

-----

## Core Logic Overview

The object detection process follows these main steps:

1.  **Feature Detection & Description:** **SIFT** is applied to both the `object.jpg` (`img1`) and `scene.jpg` (`img2`) to get their keypoints (`kp1`, `kp2`) and descriptors (`des1`, `des2`).
2.  **Matching:** The **FLANN** matcher performs $k$-nearest neighbor matching (`knnMatch`), finding the two best matches for each descriptor in the object image.
3.  **Ratio Test:** **Lowe's ratio test** is applied to retain only the most distinctive matches (`good`).
4.  **Homography:** If enough good matches (at least 4) are found, the $\mathbf{H}$ matrix is computed using `cv2.findHomography` with **RANSAC** (RANdom SAmple Consensus) to ensure robustness against outliers.
5.  **Projection:** The four corner coordinates of the object image are transformed using $\mathbf{H}$ and `cv2.perspectiveTransform` to determine their projected location in the scene image.
6.  **Visualization:** A green polygon is drawn using `cv2.polylines` on the scene image based on the transformed corner points, marking the location of the detected object.

-----

-----

## Learning Document: Color, Features, and Matching

This document summarizes the key concepts and techniques learned during the week, focusing on image segmentation, morphological operations, and feature-based computer vision.

### Color & Segmentation

  * **HSV Color Space:** This space, represented by **Hue (H)**, **Saturation (S)**, and **Value (V)**, is preferred over RGB for color-based segmentation because the color (Hue) is decoupled from the lighting conditions (Value).
  * **Color Segmentation:** Objects can be segmented by calculating the **mean HSV value** within their contour and classifying the color based on the Hue range. Low saturation and value ranges are used to identify achromatic colors like **black, white, and gray**.

-----

### Morphological Operations

  * These are simple image processing operations based on shape, applied to **binary images** (masks) using a small matrix called a **structuring element** (or kernel). Types of kernels include **Rectangle, Ellipse, and Cross**.
  * **Erosion:** Shrinks the white (foreground) areas by "eroding" boundaries. Useful for removing small noise spots.
  * **Dilation:** Grows the white (foreground) areas. Useful for connecting broken parts of an object.
  * **Opening:** **Erosion followed by Dilation** ($\text{Open}(I) = \text{Dilate}(\text{Erode}(I))$. Used to remove small objects (noise) while preserving the shape of larger objects.
  * **Closing:** **Dilation followed by Erosion** ($\text{Close}(I) = \text{Erode}(\text{Dilate}(I))$. Used to close small holes within the foreground objects.

-----

### Feature Detection & Matching

  * **Keypoints & Descriptors:** A **Keypoint** is a distinctive, stable point in an image (e.g., a corner or blob). A **Descriptor** is a vector that mathematically describes the local neighborhood around the keypoint, allowing it to be matched to other images.
  * **SIFT (Scale-Invariant Feature Transform):** A robust algorithm for detecting and describing keypoints that are invariant to changes in image scale and rotation.
  * **ORB (Oriented FAST and Rotated BRIEF):** A highly efficient and faster alternative to SIFT/SURF, making it suitable for real-time applications.

-----

### Feature Matching & Object Localization

  * **Brute-Force (BF) Matcher:** Simple matching that checks every descriptor in the first set against every descriptor in the second set. Used for descriptors like ORB that use the **Hamming distance**.
  * **FLANN-Based Matcher:** An optimized approximate matching method, often used with SIFT due to its speed in handling high-dimensional data.
  * **Lowe's Ratio Test:** A quality control technique that keeps a match only if the distance to its best match is significantly smaller than the distance to its second-best match.
  * **Homography:** A $3 \times 3$ matrix ($\mathbf{H}$) representing a perspective transformation between two planes. It is calculated using reliable feature matches (often after RANSAC to remove outliers) and is essential for **object localization** and scene stitching.
  * **Project:** The final project utilized **SIFT** features, **FLANN** matching, and **Homography** to locate a specific object (like a logo) within a complex scene.