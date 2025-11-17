import numpy as np 
import cv2 
from matplotlib import pyplot as plt
     

query_img = cv2.imread('/home/hp/Documents/Daily_Task/Day_2/Assets/Feature/query_image.jpg') 
train_img = cv2.imread('/home/hp/Documents/Daily_Task/Day_2/Assets/Feature/training_image.jpg') 
 
query_img_gray = cv2.cvtColor(query_img,cv2.COLOR_BGR2GRAY) 
train_img_gray = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY) 
 
orb = cv2.ORB_create() 

query_keypoints, query_descriptors = orb.detectAndCompute(query_img_gray,None) 
train_keypoints, train_descriptors = orb.detectAndCompute(train_img_gray,None) 

matcher = cv2.BFMatcher() 
matches = matcher.match(query_descriptors,train_descriptors) 

output_img = cv2.drawMatches(query_img, query_keypoints, 
train_img, train_keypoints, matches[:20],None) 
 
output_img = cv2.resize(output_img, (1200,650)) 

cv2.imwrite("feature_matching_result.jpg", output_img) 

cv2.waitKey(0)
cv2.destroyAllWindows()