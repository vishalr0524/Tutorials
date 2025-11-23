import cv2 as cv
import matplotlib.pyplot as plt
import numpy


def SIFT():
    img = cv.imread("/home/hp/Documents/Daily_Task/Day_2/Assets/Feature/training_image.jpg")
    imgGray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()
    keypoints = sift.detect(imgGray, None)
    #keypoints, des = sift.detectAndCompute(imgGray, None)  
    imgGray = cv.drawKeypoints(imgGray, keypoints, imgGray, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    print("Total Keypoints:", len(keypoints))
    # print("Descriptor shape:", des.shape)

    plt.figure()
    plt.imshow(imgGray)
    plt.show()

def ORB():
    img = cv.imread("/home/hp/Documents/Daily_Task/Day_2/Assets/Feature/training_image.jpg")
    imgGray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    orb = cv.ORB_create()
    keypoints = orb.detect(imgGray, None)
    keypoints, _ = orb.compute(imgGray, keypoints)
    imgGray = cv.drawKeypoints(imgGray, keypoints, imgGray, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    plt.figure()
    plt.imshow(imgGray)
    plt.show()

if __name__ == '__main__':
    ORB()




