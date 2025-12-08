import cv2
import numpy as np

def draw_length(img, ptA, ptB, label=None):
    x1, y1 = ptA
    x2, y2 = ptB

    dist = int(np.hypot(x2 - x1, y2 - y1))

    if label is None:
        label = f"{dist}px"

    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    mid = ((x1 + x2)//2, (y1 + y2)//2)

    cv2.putText(
        img, label, mid,
        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        (0, 0, 255), 2, cv2.LINE_AA
    )

    return dist


image_path = '/home/hp/Documents/Daily_Task/Day_2/Motor_Sparings_Capstone_Project/Assets/Motor_Stampings2.png'
img = cv2.imread(image_path)
copy_img = img.copy()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

points = {}

for cnt in contours:
    epsilon = 0.005 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    for i, point in enumerate(approx):
        x, y = point.ravel()

        points[f"p{i+1}"] = (x, y)

        # cv2.circle(copy_img, (x, y), 7, (0, 0, 255), -1)
        # cv2.putText(copy_img, f"p{i+1}", (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)


p1  = points["p1"]
p3  = points["p3"]
p7  = points["p7"]
p13 = points["p13"]
p12 = points["p12"]
p14 = points["p14"]
p12 = points["p12"]
p4 = points["p4"]
p8 = points["p8"]
p6 = points["p6"]
p9 = points["p9"]
p2 = points["p2"]
p11 = points["p11"]

draw_length(copy_img, p3, p2)
draw_length(copy_img, p1, p14)
draw_length(copy_img, p13, p12)
draw_length(copy_img, p11, p12)
draw_length(copy_img, p3, p4)
draw_length(copy_img, p9, p8)
draw_length(copy_img, p7, p6)

cv2.imshow("Result", copy_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
