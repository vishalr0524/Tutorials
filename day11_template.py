import cv2
import matplotlib.pyplot as plt
import numpy as np

#Extact image size
face = cv2.imread("/home/hp/Documents/Daily_Task/Day_2/templates/person1.jpg")
image = cv2.imread("/home/hp/Documents/Daily_Task/Day_2/Assets/busiest_airports3.jpg")

full_img = image.copy()

methods = ['cv2.TM_CCOEFF','cv2.TM_CCOEFF_NORMED','cv2.TM_CCORR','cv2.TM_CCORR_NORMED','cv2.TM_SQDIFF','cv2.TM_SQDIFF_NORMED']

for m in methods:
    method = eval(m)
    res = cv2.matchTemplate(full_img, face, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    if method in [cv2.TM_SQDIFF,cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc

    h,w,c = face.shape

    br = (top_left[0]+w, top_left[1]+h)

    cv2.rectangle(full_img,top_left,br,(255,0,0),10)

    plt.subplot(121)
    plt.imshow(res)
    plt.title("HEATMAP")

    plt.subplot(122)
    plt.imshow(full_img)
    plt.title("TEMPLATE BBOX")
    plt.suptitle(m)

    plt.show()

# res = cv2.matchTemplate(full_img, face, cv2.TM_CCOEFF)
# min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
# cv2.imshow("Test",res)
cv2.waitKey(0)


