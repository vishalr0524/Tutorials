import cv2
import numpy as np
image_path = '/home/hp/Documents/Daily_Task/Day_2/Motor_Sparings_Capstone_Project/Assets/Motor_Stampings.png'
original_image = cv2.imread(image_path)
gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
min_area = 1400
max_area = 25000
filtered_contours = []
large_contours = []  # Added: List for contours with area > max_area (outer)
for contour in contours:
    area = cv2.contourArea(contour)
    if min_area < area < max_area:
        filtered_contours.append(contour)
    elif area > max_area:  # Added: Collect large contours for outer circle
        large_contours.append(contour)

result_img = original_image.copy()
cv2.drawContours(result_img, filtered_contours, -1, (0, 255, 0), 2)  # Existing: Draw inner filtered contours in green

if len(filtered_contours) == 1:
    contour = filtered_contours[0]
    (x, y), radius = cv2.minEnclosingCircle(contour)
    center = (int(x), int(y))
    radius = int(radius)
    diameter = 2 * radius
   
    print(f"Inner Circle Center: {center}")
    print(f"Inner Radius: {radius} pixels")
    print(f"Inner Diameter: {diameter} pixels")
   
    # Draw the enclosing circle (blue) and center point (red) on result_img
    # cv2.circle(result_img, center, radius, (255, 0, 0), 2)
    cv2.circle(result_img, center, 1, (0, 0, 255), 3) # Thicker dot for center visibility

if len(large_contours) >= 1:
    # Use the largest large contour for outer circle (in case multiples)
    outer_contour = max(large_contours, key=cv2.contourArea)
    (ox, oy), outer_radius = cv2.minEnclosingCircle(outer_contour)
    outer_center = (int(ox), int(oy))
    outer_radius = int(outer_radius)
    outer_diameter = 2 * outer_radius
   
    print(f"Outer Circle Center: {outer_center}")
    print(f"Outer Radius: {outer_radius} pixels")
    print(f"Outer Diameter: {outer_diameter} pixels")

    cv2.circle(result_img, outer_center, outer_radius, (0, 255, 0), 2)  # Green circle
    cv2.circle(result_img, outer_center, 3, (0, 255, 255), -1)  # Yellow dot for outer center

cv2.imshow('Original Image', thresh)
cv2.imshow('Filtered Contours Result', result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
