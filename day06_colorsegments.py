import cv2
import numpy as np

img_path = '/home/hp/Documents/Daily_Task/Day_2/Assets/shapes_1.jpg'

img = cv2.imread(img_path)
if img is None:
    raise ValueError(f"Could not load image from {img_path}")
h, w = img.shape[:2]
original = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5,5), 0)
edges = cv2.Canny(blurred, 50, 150)
contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
hierarchy = hierarchy[0]
print(f" - Total contours: {len(contours)}")

children = [[] for _ in contours]
for i, h in enumerate(hierarchy):
    parent = h[3]
    if parent != -1:
        children[parent].append(i)

def detect_shape(contour):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, closed=True)
   
    if len(approx) == 3:
        return "triangle"
    if len(approx) == 4:
        x, y, bw, bh = cv2.boundingRect(approx)
        aspect_ratio = bw / float(bh)
        if 0.95 <= aspect_ratio <= 1.05:
            return "square"
        return "rectangle"
   
    area = cv2.contourArea(contour)
    if area == 0:
        return "unknown"
    circularity = 4 * np.pi * area / (peri * peri)
    if circularity > 0.85:
        return "circle"
   
    return f"polygon"

def detect_color(contour, img):
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [contour], 0, 255, thickness=-1)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mean_hsv = cv2.mean(hsv_img, mask=mask)[:3]
    h, s, v = [int(x) for x in mean_hsv]
    
    if s < 30:
        if v < 50:
            return "black"
        elif v > 200:
            return "white"
        else:
            return "gray"
 
    if (h < 10) or (h > 170):
        return "red"
    elif 10 <= h < 25:
        return "orange"
    elif 25 <= h < 40:
        return "yellow"
    elif 40 <= h < 80:
        return "green"
    elif 80 <= h < 110:
        return "cyan"
    elif 110 <= h < 130:
        return "blue"
    elif 130 <= h < 160:
        return "purple"
    elif 160 <= h <= 170:
        return "pink"  
    else:
        return "unknown"

def describe(idx):
    contour = contours[idx]
    area = cv2.contourArea(contour)
    if area < 300:
        return
    shape_name = detect_shape(contour)
    color_name = detect_color(contour, img)  
    x, y, bw, bh = cv2.boundingRect(contour)
    print(f"{shape_name.upper()} {color_name.upper()} (bbox: ({x}, {y}, {bw}, {bh}))")

for i in range(len(contours)):
    if hierarchy[i][3] == -1:
        describe(i)

output = original.copy()

def draw_hierarchy(idx, depth=0):
    if cv2.contourArea(contours[idx]) < 300:
        return
    x, y, bw, bh = cv2.boundingRect(contours[idx])
    color = (0, 255 // (depth + 1), 255 // (depth + 1))
    cv2.rectangle(output, (x, y), (x + bw, y + bh), color, 2)
    shape_label = detect_shape(contours[idx]).upper()
    color_label = detect_color(contours[idx], img).upper() 
    full_label = f"{shape_label} {color_label}"
    cv2.putText(output, full_label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    for child in children[idx]:
        draw_hierarchy(child, depth + 1)

for i in range(len(contours)):
    if hierarchy[i][3] == -1:
        draw_hierarchy(i)

cv2.imwrite('visualized_shapes.jpg', output)
print("Visualization saved as 'visualized_shapes.jpg'")