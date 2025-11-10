import cv2
import numpy as np
from day04_contours import ContourAnalyzer 
from day02_edges import SobelDetector, CannyDetector, Thresholding
from day03_filters import FilterApplier
import matplotlib.pyplot as plt

def main():

    image_path = "/home/hp/Documents/Daily_Task/Day_2/Assets/shapes_1.jpg" 
    image = cv2.imread(image_path)
    cvt = FilterApplier.apply_grayscale(image)
    blur = FilterApplier.apply_gaussian_filter(cvt)
    sobel = SobelDetector(image=blur)
    sx, sy, smag = sobel.detect()

    lower_thresh = np.percentile(smag[smag > 0], 60)  
    upper_thresh = np.percentile(smag[smag > 0], 100) 
    binary_edges = np.zeros_like(smag)
    binary_edges[smag > lower_thresh] = 255

    ctr = ContourAnalyzer(image=image)
    ctr.find_contours(edge_image=binary_edges, include_inner=True)
    shape, contour_data = ctr.analyze_and_draw(show_inner=True, return_contour_data=True)
    ContourAnalyzer.plot_color_histogram(contour_data, image)


    cv2.namedWindow("Shapes", cv2.WINDOW_NORMAL)
    cv2.imshow("Shapes", shape)

    print("Press 'q' to quit...")
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()




if __name__ == "__main__":
    main()