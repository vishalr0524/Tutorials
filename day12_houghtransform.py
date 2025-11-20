import cv2
import numpy as np

src = cv2.imread("/home/hp/Documents/Daily_Task/Day_2/Assets/building.jpg")

color_dst = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

dst = cv2.Canny(src, 50, 200, 3)

lines = cv2.HoughLinesP(
        dst, 
        rho=1, 
        theta=np.pi / 180, 
        threshold=80, 
        minLineLength=30, 
        maxLineGap=10
    )

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0] 
        cv2.line(src, (x1, y1), (x2, y2), (0, 0, 255), 3, cv2.LINE_AA)


cv2.imshow("Source Image (Grayscale)", src)
cv2.imshow("Detected Lines", color_dst)
cv2.waitKey(0)
cv2.destroyAllWindows()