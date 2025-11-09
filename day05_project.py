#main.py
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


        sobel = SobelDetector(image=blur)
        sx, sy, smag = sobel.detect()
        # canny = CannyDetector(image=blur)
        # can_edge = canny.detect()

        # But apply Canny-like thresholding
        # Lower threshold for weak edges, upper for strong edges
        lower_thresh = np.percentile(smag[smag > 0], 60)  # 20th percentile
        upper_thresh = np.percentile(smag[smag > 0], 100)  # 80th percentile

        # Create binary edge map
        binary_edges = np.zeros_like(smag)
        binary_edges[smag > lower_thresh] = 255

        # Clean up with morphology
        edges_closed = cv2.morphologyEx(binary_edges, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))

        ctr = ContourAnalyzer(image=img)
        ctr.find_contours(edge_image=edges_closed, include_inner=True)
        shape, contour_data = ctr.analyze_and_draw(show_inner=True, return_contour_data=True)

                # Show histograms for each detected shape
        for data in contour_data:
            fig = ctr.visualize_color_confidence(img, data['contour'], data['color'])
            plt.show(block=False)  # Non-blocking

        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Filter", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Edge", cv2.WINDOW_NORMAL)
        cv2.namedWindow("contours", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Threshold", cv2.WINDOW_NORMAL)

        cv2.imshow("Image", img)
        cv2.imshow("Filter", blur)
        cv2.imshow("Edge", smag)
        cv2.imshow("Threshold", binary_edges)
        cv2.imshow("contours", shape)

        print("Press 'q' to quit...")
        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except cv2.error as e:
        print(f"OpenCV Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
