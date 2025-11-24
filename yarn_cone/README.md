# Donut Segmentation -- README

This script performs **donut-shaped object segmentation** from an image
using OpenCV. The workflow detects a blue-colored ring, removes noise,
extracts inner and outer contours, and isolates the donut region.

## Features

-   Converts image to HSV for robust color filtering
-   Applies morphological **open + close** operations for noise
    reduction
-   Uses **RETR_LIST** to detect all contours
-   Filters contours by area and identifies **outer** and **inner**
    rings
-   Creates a mask to extract only the donut region
-   Displays intermediate and final results

## Input

Update the image path in the script:

``` python
img = cv2.imread('5455_uv.png')
```

## Processing Steps

1.  HSV conversion
2.  Color thresholding using blue-range mask
3.  Morphological opening → removes small noise
4.  Morphological closing → fills gaps
5.  Contour detection & filtering
6.  Sort contours by area
7.  Largest = outer ring
8.  Second largest = inner ring
9.  Mask outer − inner → donut region
10. Overlay red highlight on detected region

## Windows Displayed

-   Original -- raw input image
-   Mask Ring -- cleaned mask
-   Segmented Donut -- ring-only binary mask
-   Original1 -- mask overlay on the original image

Press any key to close the windows.

## Run

``` bash
python donut_segment.py
```