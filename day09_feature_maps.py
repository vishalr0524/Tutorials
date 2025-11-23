import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def SIFT():
    img1 = cv.imread("/home/hp/Documents/Daily_Task/Day_2/Assets/Feature/object.jpg")
    img2 = cv.imread("/home/hp/Documents/Daily_Task/Day_2/Assets/Feature/scene.jpg")
    img1Gray= cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
    img2Gray= cv.cvtColor(img2,cv.COLOR_BGR2GRAY)

    sift = cv.SIFT_create()
    keypoints, des = sift.detectAndCompute(img1Gray, None)
    keypoints2, des2 = sift.detectAndCompute(img2Gray, None)  
    
    FLANN_INDEX_KDTREE = 1
    nKDtrees = 5
    nLeafChecks = 50
    nNeighbors = 2

    indexParms = dict(algorithm=FLANN_INDEX_KDTREE, trees=nKDtrees)
    searchParams = dict(checks=nLeafChecks)

    flann = cv.FlannBasedMatcher(indexParms, searchParams)
    matches = flann.knnMatch(des, des2, k=nNeighbors)

    # Ratio Test (Lowe's ratio test)
    matchesMask = [[0,0] for i in range(len(matches))]
    testRatio = 0.75

    for i, (m,n) in enumerate(matches):
        if m.distance < testRatio * n.distance:
            matchesMask[i] = [1,0]

    drawParams = dict(matchColor=(0,255,0), 
                    singlePointColor=(255,0,0),
                    matchesMask=matchesMask, 
                    flags=cv.DrawMatchesFlags_DEFAULT)
    imgMatch = cv.drawMatchesKnn(img1, keypoints, img2, keypoints2, matches, None, **drawParams)
    plt.figure()
    plt.imshow(imgMatch)
    plt.show()

def ORB():
    img1 = cv.imread("/home/hp/Documents/Daily_Task/Day_2/Assets/Feature/object.jpg")
    img2 = cv.imread("/home/hp/Documents/Daily_Task/Day_2/Assets/Feature/scene.jpg")
    img1Gray= cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
    img2Gray= cv.cvtColor(img2,cv.COLOR_BGR2GRAY)

    orb = cv.ORB_create()
    keypoints1, descriptor1 = orb.detectAndCompute(img1Gray, None)
    keypoints2, descriptor2 = orb.detectAndCompute(img2Gray, None)
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    

    matches = bf.match(descriptor1, descriptor2)
    matches = sorted(matches, key=lambda x:x.distance)
    
    nMatches = 20
    imgMatch = cv.drawMatches(img1, keypoints1, img2, keypoints2, matches[:nMatches], 
                              None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt.figure()
    plt.imshow(imgMatch)
    plt.show()

def SIFT_with_Homography():
    img1 = cv.imread("/home/hp/Documents/Daily_Task/Day_2/Assets/Feature/object.jpg")
    img2 = cv.imread("/home/hp/Documents/Daily_Task/Day_2/Assets/Feature/scene.jpg")
    img1Gray= cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
    img2Gray= cv.cvtColor(img2,cv.COLOR_BGR2GRAY)

    # SIFT
    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1Gray, None)
    kp2, des2 = sift.detectAndCompute(img2Gray, None)

    # FLANN
    FLANN_INDEX_KDTREE = 1
    indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    searchParams = dict(checks=50)
    flann = cv.FlannBasedMatcher(indexParams, searchParams)
    matches = flann.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)

    # Need at least 4 points to compute Homography
    if len(good) > 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

        H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

        # Get corners of object image
        h, w = img1Gray.shape
        corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)

        # Transform corners into scene image
        transformed = cv.perspectiveTransform(corners, H)

        # Draw boundary on scene image
        img2_poly = img2.copy()
        cv.polylines(img2_poly, [np.int32(transformed)], True, (0,255,0), 3)

        plt.imshow(cv.cvtColor(img2_poly, cv.COLOR_BGR2RGB))
        plt.title("Object detected in Scene")
        plt.show()

    else:
        print("Not enough matches found!")

if __name__ == '__main__':
    SIFT_with_Homography()




