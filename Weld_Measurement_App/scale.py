import cv2
import numpy as np
import os
import glob

def find_centroid(cnt):
    """
    Calculates the centroid (cx, cy) of a contour.
    Returns None if the contour has zero area.
    """
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)

def validate_gaps(ticks, name):
    """
    Calculates the dominant gap and its frequency between sorted tick positions.
    """
    if len(ticks) < 2:
        print(f"{name}: Not enough ticks\n")
        return 0, None, None

    gaps = np.diff(ticks)

    if len(gaps) == 0:
        return 0, None, None
    
    hist = np.bincount(gaps)
    dominant_gap = np.argmax(hist)
    freq = hist[dominant_gap]

    print(f"{name} tick positions:", ticks)
    print(f"{name} gaps:", gaps)
    print(f"Dominant {name} gap = {dominant_gap}px")

    return freq, dominant_gap, gaps

def smart_distance(p1, p2, tol=3):
    """
    Calculates the distance between two points (p1, p2),
    """
    if p1 is None or p2 is None:
        return 0.0
        
    (x1, y1) = p1
    (x2, y2) = p2

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)

    # If points are mostly aligned horizontally or vertically
    if dy <= tol:
        return dx
    if dx <= tol:
        return dy

    # Otherwise, use Euclidean distance
    return np.sqrt(dx*dx + dy*dy)

def process_image(img_path):
    """
    Processes a single image file to find and measure calibration ticks.
    
    Args:
        img_path (str): The full path to the image file.
    """
    print(f"--- Processing: {os.path.basename(img_path)} ---")

    # Read the image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not read image at {img_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Edge Detection (Sobel)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    sobelx = cv2.convertScaleAbs(sobelx)
    sobely = cv2.convertScaleAbs(sobely)

    # 2. Thresholding
    _, th_x = cv2.threshold(sobelx, 80, 255, cv2.THRESH_BINARY)
    _, th_y = cv2.threshold(sobely, 80, 255, cv2.THRESH_BINARY)

    # 3. Morphological Operations (Closing)
    kernel_vert = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
    kernel_horz = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))

    clean_x = cv2.morphologyEx(th_x, cv2.MORPH_CLOSE, kernel_vert)
    clean_y = cv2.morphologyEx(th_y, cv2.MORPH_CLOSE, kernel_horz)

    # 4. Find Vertical Ticks
    contours_x, _ = cv2.findContours(clean_x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    vertical_ticks = []
    vertical_cnts = []

    for cnt in contours_x:
        x, y, w, h = cv2.boundingRect(cnt)
        if h < 15:  # Height filter
            continue
        if w > 8: # Width filter
            continue

        cx = x + w // 2
        vertical_ticks.append(cx)
        vertical_cnts.append(cnt)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2) # Draw bounding box

    vertical_ticks = np.array(sorted(vertical_ticks))

    # 5. Find Horizontal Ticks
    contours_y, _ = cv2.findContours(clean_y, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    horizontal_ticks = []
    horizontal_cnts = []

    for cnt in contours_y:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 15: # Width filter
            continue
        if h > 8: # Height filter
            continue

        cy = y + h // 2
        horizontal_ticks.append(cy)
        horizontal_cnts.append(cnt)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2) # Draw bounding box

    horizontal_ticks = np.array(sorted(horizontal_ticks))

    # 6. Validate Gaps and Determine Orientation
    v_freq, v_gap, v_gaps = validate_gaps(vertical_ticks, "Vertical")
    h_freq, h_gap, h_gaps = validate_gaps(horizontal_ticks, "Horizontal")

    use_vertical = v_freq > h_freq

    if use_vertical:
        print("\n✔ Using VERTICAL scale\n")
        ticks = vertical_ticks
        gap = v_gap
        cnts = vertical_cnts
    else:
        print("\n✔ Using HORIZONTAL scale\n")
        ticks = horizontal_ticks
        gap = h_gap
        cnts = horizontal_cnts

    if gap is None:
        print("✘ Could not determine a dominant gap. Skipping remaining steps.")
        cv2.imshow(f"Scale Centroid Points - {os.path.basename(img_path)}", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # 7. Find a pair of ticks separated by the dominant gap
    tick_idx = None
    gaps = np.diff(ticks)
    for i, g in enumerate(gaps):
        if g == gap:
            tick_idx = i
            break

    if tick_idx is None:
        print("✘ No matching tick pair found for the dominant gap.")
        cv2.imshow(f"Scale Centroid Points - {os.path.basename(img_path)}", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    tick1 = ticks[tick_idx]
    tick2 = ticks[tick_idx + 1]

    print(f"\nSelected ticks: {tick1} px and {tick2} px")
    print(f"Pixel distance = {abs(tick2 - tick1)} px")

    # 8. Find the corresponding contours (c1, c2)
    c1 = None
    c2 = None

    if use_vertical:
        for cnt in cnts:
            x,y,w,h = cv2.boundingRect(cnt)
            cx = x + w//2
            if cx == tick1:
                c1 = cnt
            if cx == tick2:
                c2 = cnt
    else:
        for cnt in cnts:
            x,y,w,h = cv2.boundingRect(cnt)
            cy = y + h//2
            if cy == tick1:
                c1 = cnt
            if cy == cy:
                c2 = cnt
                
    # 9. Calculate Centroids
    cent1 = find_centroid(c1)
    cent2 = find_centroid(c2)

    print("Centroid 1:", cent1)
    print("Centroid 2:", cent2)
    
    # 10. Calculate Distance and Scale
    distance_px = smart_distance(cent1, cent2)
    print(f"\n Distance = {distance_px:.2f} px")

    # Assuming the physical distance between the two selected ticks is 1.0 cm
    if distance_px > 0:
        px_per_cm = distance_px / 1.0
        print(f"\n Pixels per cm = {px_per_cm:.2f} px/cm\n")
    else:
        print("\n Distance is zero or invalid. Cannot calculate scale.\n")


    # 11. Visualization
    if cent1:
        cv2.circle(img, cent1, 6, (0,255,0), -1)

    if cent2:
        cv2.circle(img, cent2, 6, (0,255,0), -1)

    cv2.imshow(f"Scaled_{os.path.basename(img_path)}", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
# --- Main Execution ---

if __name__ == "__main__":
    
    # SINGLE_IMAGE_PATH = "/home/hp/Documents/Daily_Task/Day_2/Assets/welds/04.jpg" 
    # process_image(SINGLE_IMAGE_PATH)

    IMAGE_FOLDER_PATH = "/home/hp/Documents/Daily_Task/Day_2/Assets/welds" 
    IMAGE_FILE_PATTERN = "*.jpg"
    search_pattern = os.path.join(IMAGE_FOLDER_PATH, IMAGE_FILE_PATTERN)
    image_files = sorted(glob.glob(search_pattern))
    if not image_files:
        print(f"No images found in {IMAGE_FOLDER_PATH} matching pattern {IMAGE_FILE_PATTERN}")
    else:
        print(f"Found {len(image_files)} images to process.")
        for img_file in image_files:
            process_image(img_file)