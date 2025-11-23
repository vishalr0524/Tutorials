import cv2
import numpy as np


image = cv2.imread("/home/hp/Documents/Daily_Task/Day_2/uv_nov11/5293_uv.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

k = 5
kern_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
kern_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
kern_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (k, k))



erode_rect = cv2.erode(mask, kern_rect, iterations=5)
erode_ellipse = cv2.erode(mask, kern_ellipse, iterations=2)
erode_cross = cv2.erode(mask, kern_cross, iterations=1)



dilate_rect = cv2.dilate(mask, kern_rect, iterations=3)
dilate_ellipse = cv2.dilate(mask, kern_ellipse, iterations=2)
dilate_cross = cv2.dilate(mask, kern_cross, iterations=1)



open_rect = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern_rect)
open_ellipse = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern_ellipse)
open_cross = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern_cross)



close_rect = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern_rect)
close_ellipse = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern_ellipse)
close_cross = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern_cross)


cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
cv2.namedWindow("Binary", cv2.WINDOW_NORMAL)
cv2.namedWindow("Opening", cv2.WINDOW_NORMAL)
cv2.namedWindow("Close", cv2.WINDOW_NORMAL)
cv2.imshow("Original", image)
cv2.imshow("Binary", mask)
cv2.imshow("Opening", erode_ellipse)
cv2.imshow("Close", close_ellipse)


# cv2.imwrite("mask_otsu.png", mask)
# cv2.imwrite("erode_rect.png", erode_rect)
# cv2.imwrite("open_diamond.png", open_diamond)
# cv2.imwrite("close_ellipse.png", close_ellipse)

cv2.waitKey(0)
cv2.destroyAllWindows()

