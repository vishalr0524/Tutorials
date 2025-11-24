import cv2
import numpy as np

#5553 5576 5311 5356 5339

img = cv2.imread('/home/hp/Documents/Daily_Task/Day_2/uv_nov11/5455_uv.png')

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_blue = np.array([103, 129, 32])
upper_blue = np.array([111, 205, 109])
mask = cv2.inRange(hsv, lower_blue, upper_blue)

kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN,kernel1, iterations=1)
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (37, 37))
closed = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel_close, iterations=1)

cnts = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
filtered = []
for c in cnts:
    area = cv2.contourArea(c)
    if area < 300:
        continue
    filtered.append(c)


if len(filtered) < 2:
    print("Not enough contours to form donut")
    exit()


filtered = sorted(filtered, key=cv2.contourArea, reverse=True)
outer = filtered[0]
inner = filtered[1]
ring_mask = np.zeros(mask.shape, dtype=np.uint8)



cv2.drawContours(ring_mask, [outer], -1, 255, -1)
cv2.drawContours(ring_mask, [inner], -1, 0, -1)
segmented = cv2.bitwise_and(img, img, mask=ring_mask)

overlay = img.copy()
overlay[closed == 255] = [0, 0, 255]
cv2.namedWindow("Mask Ring", cv2.WINDOW_NORMAL)
cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
cv2.imshow("Original", img)
cv2.imshow("Mask Ring", opening)
cv2.namedWindow("Segmented Donut", cv2.WINDOW_NORMAL)
cv2.imshow("Segmented Donut", ring_mask)
cv2.namedWindow("Original1", cv2.WINDOW_NORMAL)
cv2.imshow("Original1", overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()