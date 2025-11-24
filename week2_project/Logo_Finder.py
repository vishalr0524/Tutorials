import numpy as np 
import cv2 as cv
from matplotlib import pyplot as plt
     

def SIFT_with_Homography():
    img1 = cv.imread("/home/hp/Documents/Daily_Task/Day_2/Assets/Feature/object.jpg")
    img2 = cv.imread("/home/hp/Documents/Daily_Task/Day_2/Assets/Feature/scene.jpg")
    img1Gray= cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
    img2Gray= cv.cvtColor(img2,cv.COLOR_BGR2GRAY)

    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1Gray, None)
    kp2, des2 = sift.detectAndCompute(img2Gray, None)


    FLANN_INDEX_KDTREE = 1
    indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    searchParams = dict(checks=50)
    flann = cv.FlannBasedMatcher(indexParams, searchParams)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)

    if len(good) > 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

        H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        h, w = img1Gray.shape
        corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
        transformed = cv.perspectiveTransform(corners, H)
        img2_poly = img2.copy()
        cv.polylines(img2_poly, [np.int32(transformed)], True, (0,255,0), 3)

        plt.imshow(cv.cvtColor(img2_poly, cv.COLOR_BGR2RGB))
        plt.title("Object detected in Scene")
        plt.show()

    else:
        print("Not enough matches found!")

if __name__ == '__main__':
    SIFT_with_Homography()