import cv2
import numpy as np
from day04_contours import ContourAnalyzer 
from day02_edges import SobelDetector, CannyDetector, Thresholding
from day03_filters import FilterApplier

def main():

    image_path = "/home/hp/Documents/Daily_Task/Day_2/uv_nov11/5576_uv.png" 
    image = cv2.imread(image_path)
    cvt = FilterApplier.apply_grayscale(image)
    blur = FilterApplier.apply_gaussian_filter(cvt)
    sobel = SobelDetector(image=blur)
    sx, sy, smag = sobel.detect()

    lower_thresh = np.percentile(smag[smag > 0], 40)  
    upper_thresh = np.percentile(smag[smag > 0], 80) 
    binary_edges = np.zeros_like(smag)
    binary_edges[smag > lower_thresh] = 255

    cv2.namedWindow("Shapes", cv2.WINDOW_NORMAL)
    cv2.imshow("Shapes", binary_edges)

    print("Press 'q' to quit...")
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()