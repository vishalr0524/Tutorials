import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

image = cv2.imread("/home/hp/Documents/Daily_Task/Day_2/Assets/national-highway-4.jpg")
img = image.copy()
marker = np.zeros(image.shape[:2],dtype=np.int32)
segments = np.zeros(image.shape, dtype=np.uint8)

def create_rgb(i):
    return tuple(np.array(cm.tab10(i)[:3])*255)

colors = []
for i in range(10):
    colors.append(create_rgb(i))

n_markers = 10
current_marker = 1
marks_updates = False

def mouse_callback(event, x, y, flags, param):
    global marks_updates
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(marker, (x,y), 10, (current_marker),-1)
        cv2.circle(img, (x,y),10,colors[current_marker],-1)
        marks_updates = True

cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('Image',mouse_callback)

while True:
    cv2.imshow("Segments", segments)
    cv2.imshow("Image", img)

    k=cv2.waitKey(1)

    if k == 27:
        break

    elif k == ord('c'):
        image1 = image.copy()
        marker = np.zeros(image.shape[:2], dtype=np.int32)
        segments = np.zeros(image.shape,dtype=np.uint8)

    elif k > 0 and chr(k).isdigit():
        current_marker = int(chr(k))

    if marks_updates:
        marker_img = marker.copy()
        cv2.watershed(image,marker_img)
        segments = np.zeros(image.shape, dtype=np.uint8)

        for i in range(n_markers):
            segments[marker_img==(i)] = colors[i]

