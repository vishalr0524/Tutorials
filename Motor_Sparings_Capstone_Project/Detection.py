# import cv2
# import numpy as np

# def empty(a):
#     pass
# cv2.namedWindow("Parameters")
# cv2.resizeWindow("Parameters", 640, 240)
# cv2.createTrackbar("MIN_AREA", "Parameters", 0, 5000, empty)
# cv2.createTrackbar("MAX_AREA", "Parameters", 0, 50000, empty)


# image_path = '/home/hp/Documents/Daily_Task/Day_2/Motor_Sparings_Capstone_Project/Assets/Motor_Stampings2.png'
# original_image = cv2.imread(image_path)

# gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

# _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# while True:

#     min_area = cv2.getTrackbarPos("MIN_AREA", "Parameters")
#     max_area = cv2.getTrackbarPos("MAX_AREA", "Parameters")
    

#     if min_area >= max_area:
#         min_area = max_area - 1 

#     contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

#     filtered_contours = []
#     for contour in contours:
#         area = cv2.contourArea(contour)
        
#         # Use the dynamic min_area and max_area variables read from the trackbars
#         if min_area < area < max_area:
#             filtered_contours.append(contour)

#     # --- Draw and Display Results ---
#     result_img = original_image.copy()

#     # Draw the filtered contours in Green (0, 255, 0)
#     cv2.drawContours(result_img, filtered_contours, -1, (0, 255, 0), 2)
    
#     cv2.imshow('Original Image', original_image)
#     cv2.imshow('Filtered Contours Result', result_img)
    
#     # Break loop if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cv2.destroyAllWindows()

# import cv2
# import numpy as np

# # Load image
# image = cv2.imread("/home/hp/Documents/Daily_Task/Day_2/Motor_Sparings_Capstone_Project/Assets/Motor_Stampings2.png")
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


# def nothing(x):
#     pass


# # ------------- CREATE WINDOWS + TRACKBARS FOR ALL METHODS -------------
# methods = [
#     "Erode_RECT", "Erode_ELLIPSE", "Erode_CROSS",
#     "Dilate_RECT", "Dilate_ELLIPSE", "Dilate_CROSS",
#     "Open_RECT", "Open_ELLIPSE", "Open_CROSS",
#     "Close_RECT", "Close_ELLIPSE", "Close_CROSS"
# ]

# for name in methods:
#     cv2.namedWindow(name, cv2.WINDOW_NORMAL)
#     cv2.resizeWindow(name, 400, 300)
#     cv2.createTrackbar("Kernel", name, 5, 50, nothing)
#     cv2.createTrackbar("Iter", name, 1, 20, nothing)


# def get_kernel(shape_type, k):
#     if k < 1:
#         k = 1
#     if k % 2 == 0:
#         k += 1

#     if shape_type == "RECT":
#         return cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
#     elif shape_type == "ELLIPSE":
#         return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
#     else:
#         return cv2.getStructuringElement(cv2.MORPH_CROSS, (k, k))


# # ------------------------ LIVE LOOP ------------------------
# while True:

#     for name in methods:

#         # Read trackbars
#         k = cv2.getTrackbarPos("Kernel", name)
#         it = cv2.getTrackbarPos("Iter", name)

#         # Extract method type
#         operation, shape = name.split("_")

#         kernel = get_kernel(shape, k)

#         # Apply morphology operations
#         if operation == "Erode":
#             result = cv2.erode(mask, kernel, iterations=it)

#         elif operation == "Dilate":
#             result = cv2.dilate(mask, kernel, iterations=it)

#         elif operation == "Open":
#             result = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=it)

#         else:  # Close
#             result = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=it)

#         # Display result
#         vis = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
#         cv2.putText(vis, f"{name} | k={k} | it={it}",
#                     (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#         cv2.imshow(name, vis)

#     if cv2.waitKey(1) == 27:  # ESC to exit
#         break

# cv2.destroyAllWindows()



