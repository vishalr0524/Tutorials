# 20-Day Classical Computer Vision & Basic ML Tutorial

**Duration:** 4 Weeks (5 days per week)
**Level:** Beginner
**Focus:** Hands-on, classical computer vision and basic machine learning (no deep learning)
**Audience:** Freshers and beginners new to computer vision and Python-based image processing

---

## Table of Contents
1. [Overview](#overview)
2. [Contract (Inputs / Outputs)](#contract-inputs--outputs)
3. [Prerequisites](#prerequisites)
4. [Recommended Packages & Setup](#recommended-packages--setup)
5. [Weekly Breakdown](#weekly-breakdown)
    - [Week 1: Image Fundamentals & Edges](#week-1-image-fundamentals--edges)
    - [Week 2: Color, Features, and Matching](#week-2-color-features-and-matching)
    - [Week 3: Classical CV Pipelines + Basic ML](#week-3-classical-cv-pipelines--basic-ml)
    - [Week 4: ML, Deployment & Sockets](#week-4-ml-deployment--sockets)
6. [Final Project Options](#final-project-options)
7. [Folder Structure](#folder-structure)
8. [Tips for Beginners](#tips-for-beginners)

---

## Overview
This 20-day tutorial plan builds intuition in classical computer vision—covering image basics, edge detection, filtering, contours, color segmentation, feature matching (like SIFT), and simple ML algorithms. In the final week, you’ll also learn how to wrap your work in a **Flask API**, **Docker container**, and **Socket-based app**.

By the end, you’ll have:
- Built small working scripts and mini-apps for each topic
- Completed 3 weekly capstone projects
- Learned how to deploy your code

---

## Contract

**Outputs:**
- Daily scripts demonstrating each technique
- Weekly capstone projects
- Final project deliverables (code + README)

**Success Criteria:**
By Day 20, you can:
- Implement multiple classical CV pipelines
- Train and evaluate small ML models
- Containerize and deploy at least one working service

---

## Prerequisites
- Python 3.8+
- Basic programming knowledge (functions, loops, imports)
- Interest in image processing — no prior experience required

---

## Recommended Packages & Setup

Install the following packages:
```bash
pip install opencv-python numpy matplotlib scikit-image scikit-learn
```

Recommended: Create a virtual environment or conda environment
```bash
python -m venv cv_env
source cv_env/bin/activate
pip install --upgrade pip
pip install opencv-python numpy matplotlib scikit-image scikit-learn
```

---

## Weekly Breakdown

### Week 1: Image Fundamentals & Edges (Days 1–5)

**Day 1 — Basics of Images (RGB, Slicing)**
- What is an image? Pixels and arrays
- Color spaces (RGB, BGR, Grayscale, HSV)
- Reading, displaying, and saving images
- Image slicing and regions of interest (ROI)
- **Hands-on:** Load an image, split into RGB channels, crop/modify parts, convert to grayscale/HSV
- **Deliverable:** `day01_basics.py`

**Day 2 — Edge Detection**
- Gradients and edge concept
- Sobel, Scharr, and Canny edge detection
- Threshold tuning and visualization
- **Hands-on:** Compare Sobel vs. Canny, experiment with thresholds
- **Deliverable:** `day02_edges.py`

**Day 3 — Image Filtering & Noise Removal**
- Convolution, kernels
- Blur filters (box, Gaussian, median)
- Sharpening basics
- **Hands-on:** Apply filters to noisy images, compare noise reduction
- **Deliverable:** `day03_filters.py`

**Day 4 — Contours & Shapes**
- Thresholding, contour detection, hierarchy
- Shape properties: area, perimeter, bounding boxes
- **Hands-on:** Find objects using contours, draw contours/bounding boxes
- **Deliverable:** `day04_contours.py`

**Day 5 — Capstone: Simple Shape Detector**
- Project: Detect and label basic geometric shapes (circle, rectangle, triangle)
- Skills: Thresholding, contours, edges
- **Deliverable:** `week1_capstone_shape_detector/`

---

### Week 2: Color, Features, and Matching (Days 6–10)

**Day 6 — Color Spaces & Segmentation**
- HSV color space
- Color-based thresholding
- Histogram visualization
- **Hands-on:** Segment objects by color, plot color histograms
- **Deliverable:** `day06_color.py`

**Day 7 — Morphological Operations**
- Erosion, dilation, opening, closing
- Practical uses: noise removal, object separation
- **Hands-on:** Apply morphological operations on binary masks
- **Deliverable:** `day07_morphology.py`

**Day 8 — Feature Detection (ORB & SIFT Intro)**
- Keypoints and descriptors
- ORB and SIFT overview
- **Hands-on:** Detect and visualize keypoints
- **Deliverable:** `day08_features.py`

**Day 9 — Feature Matching**
- Descriptor matching with BFMatcher & FLANN
- Lowe’s ratio test, homography basics
- **Hands-on:** Match keypoints between two images
- **Deliverable:** `day09_feature_matching.py`

**Day 10 — Capstone: Logo Finder**
- Project: Use feature matching to locate a logo or label in a test image
- Skills: ORB/SIFT, BFMatcher
- **Deliverable:** `week2_capstone_logo_finder/`

---

### Week 3: Classical CV Pipelines + Basic ML (Days 11–15)

**Day 11 — Template Matching**
- Template matching methods (TM_CCOEFF, TM_CCORR_NORMED)
- Handling scale and rotation
- **Hands-on:** Find a small template in a larger image
- **Deliverable:** `day11_template_matching.py`

**Day 12 — Hough Transform**
- Line and circle detection
- Probabilistic Hough Transform
- **Hands-on:** Detect circles and lines in images
- **Deliverable:** `day12_hough.py`

**Day 13 — Segmentation Basics**
- Thresholding (global, adaptive, Otsu)
- Watershed algorithm overview
- **Hands-on:** Segment touching objects
- **Deliverable:** `day13_segmentation.py`

**Day 14 — Simple Detection Pipeline**
- Combining edges, color, and contours
- Designing hybrid detection pipelines
- **Hands-on:** Build a basic object detection pipeline
- **Deliverable:** `day14_pipeline.py`

**Day 15 — Capstone: Object Locator**
- Project: Find a colored object and draw a bounding box in real-time (webcam)
- **Deliverable:** `week3_capstone_object_locator/`

---

### Week 4: ML, Deployment & Sockets (Days 16–20)

**Day 16 — Basic ML: Linear Regression**
- Linear regression and model evaluation
- Feature extraction from images (mean color, area)
- **Hands-on:** Predict a simple value from image features
- **Deliverable:** `day16_linear_regression.py`

**Day 17 — KNN & KMeans**
- Classification with KNN
- Unsupervised clustering with KMeans
- **Hands-on:** Color quantization using KMeans
- **Deliverable:** `day17_knn_kmeans.py`

**Day 18 — Decision Trees & Model Evaluation**
- Decision Trees, overfitting
- Cross-validation, precision, recall
- **Hands-on:** Train/test a simple classifier on features
- **Deliverable:** `day18_trees_eval.py`

**Day 19 — Flask API**
- Flask basics, endpoints, file uploads
- **Hands-on:** Build a Flask API to serve your object detector
- **Deliverable:** `day19_flask_api/`

**Day 20 — Docker & Sockets**
- Docker basics: containerize your Flask API
- Socket basics: client-server for image streaming
- **Hands-on:** Run your CV Flask API in Docker; (Optional) Send image frames via sockets
- **Deliverable:** `day20_docker_sockets/`

---

## Final Project Options
- **Document Scanner + Color Corrector:** Detect, warp, and color-correct t for OCR readiness.
- **Color-based Object Tracker:** Track a colored object using webcam; plot its path.
- **Classical CV + ML Pipeline:** Combine classical features (color histograms, contours) with ML models (KNN or Decision Tree).

---

## Folder Structure
```
cv_tutorial/
│
├── day01_basics.py
├── day02_edges.py
├── day03_filters.py
├── ...
├── week1_capstone_shape_detector/
├── week2_capstone_logo_finder/
├── week3_capstone_object_locator/
└── week4_final_project/
```

---

## Tips for Beginners
- Visualize every step — always use `imshow()` or `plt.imshow()`
- Work with small, clear sample images first
- Document what you learn in a simple README after each week
- Commit your daily progress to GitHub
