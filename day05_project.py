import cv2
import numpy as np
import matplotlib.pyplot as plt
from day02_edges import SobelDetector, CannyDetector, Thresholding
from day03_filters import FilterApplier
from day04_contours import ContourAnalyzer

class ImageLoader:
    def __init__(self, image_path: str):
        self.image_path = image_path
        if self.image_path is None:
            raise FileNotFoundError(f"Image is not found in : {image_path}")
        self.image = None

    def load_image(self) -> np.ndarray:
        self.image = cv2.imread(self.image_path)
        return self.image

def main():
    try:
        loader = ImageLoader("/home/hp/Documents/Daily_Task/Day_2/Assets/shapes_1.jpg")
        img = loader.load_image()
        cvt = FilterApplier.apply_grayscale(img)
        blur = FilterApplier.apply_gaussian_filter(cvt)



        adapt = Thresholding(image=blur)
        cvt_adapt = adapt.otsu_threshold()

        sobel = SobelDetector(image=cvt_adapt)
        sx, sy, smag = sobel.detect()
        # canny = CannyDetector(image=blur)
        # can_edge = canny.detect()

        smag_closed = cv2.morphologyEx(smag, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

        ctr = ContourAnalyzer(image=smag_closed)
        ctr.find_contours(edge_image=smag_closed)
        shape = ctr.analyze_and_draw()

        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Filter", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Edge", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Threshold", cv2.WINDOW_NORMAL)
        cv2.namedWindow("MorphClosed", cv2.WINDOW_NORMAL)
        cv2.namedWindow("contours", cv2.WINDOW_NORMAL)

        cv2.imshow("Image", img)
        cv2.imshow("Filter", blur)
        cv2.imshow("Edge", smag)
        cv2.imshow("Threshold", cvt_adapt)
        cv2.imshow("MorphClosed", smag_closed)
        cv2.imshow("contours", shape)



        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except cv2.error as e:
        print(f"OpenCV Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
