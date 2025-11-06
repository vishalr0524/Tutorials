import cv2
import numpy as np
from typing import Tuple


class EdgeDetection:
    """Base class for image loading and display utilities."""

    def __init__(self, image_path: str = None, image : np.ndarray = None):
        if image is not None:
            self.image = image
        elif image_path is not None:
            self.image = cv2.imread(image_path)
            if self.image is None:
                raise FileNotFoundError(f"Image not found at: {image_path}")
        else:
            raise ValueError("Either image or image path needs to be provided")
        

    def to_gray(self) -> np.ndarray:
        if len(self.image.shape) == 3:
            return cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        return self.image

    def smooth(self, ksize: int = 5) -> np.ndarray:
        return cv2.GaussianBlur(self.to_gray(), (ksize, ksize), 0)


class SobelDetector(EdgeDetection):
    """Applies Sobel edge detection in x, y and combined directions."""

    def detect(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        gray = self.smooth()
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)    #Scharr without kernel size
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        magnitude = np.uint8(np.clip(magnitude, 0, 255))
        return sobel_x, sobel_y, magnitude


class Thresholding(EdgeDetection):
    """Handles global, adaptive and Otsu thresholding."""

    def global_threshold(self, thresh: int = 128) -> np.ndarray:
        gray = self.to_gray()
        _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
        return binary

    def adaptive_threshold(self) -> np.ndarray:
        gray = self.to_gray()
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

    def otsu_threshold(self) -> np.ndarray:
        gray = self.to_gray()
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return binary


class CannyDetector(EdgeDetection):
    """Implements multi-stage Canny edge detection."""

    def detect(self, low_thresh: int = 50, high_thresh: int = 150) -> np.ndarray:
        blurred = self.smooth()
        return cv2.Canny(blurred, low_thresh, high_thresh)


def main():
    image_path = "/home/hp/Documents/Daily_Task/Day_2/Assets/book.jpeg"

    sobel = SobelDetector(image_path)
    sx, sy, smag = sobel.detect()

    thresh = Thresholding(image_path)
    global_th = thresh.global_threshold()
    otsu_th = thresh.otsu_threshold()

    canny = CannyDetector(image_path)
    canny_edges = canny.detect()


    cv2.imshow("Sobel Magnitude", smag)
    cv2.imshow("Otsu Threshold", otsu_th)
    cv2.imshow("Canny Edges", canny_edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
