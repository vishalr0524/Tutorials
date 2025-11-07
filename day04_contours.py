import cv2
import numpy as np

class ContourAnalyzer:
    """A class for detecting and classifying shapes in an image."""

    def __init__(self, image_path: str = None, image : np.ndarray = None):
        
        if image is not None:
            self.original_image = image
        elif image_path is not None:
            self.original_image = cv2.imread(image_path)
            if self.original_image is None:
                raise FileNotFoundError(f"Image not found: {image_path}")

        self.image_path = image_path
        self.gray_image = self._convert_to_gray(self.original_image)
        self.contours = []
        self.hierarchy = None

    def _load_image(self) -> np.ndarray:
        """Load an image from a given path."""
        image = cv2.imread(self.image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {self.image_path}")
        return image
        
    def _convert_to_gray(self, image: np.ndarray) -> np.ndarray:
        """Convert to grayscale."""
        if len(image.shape) == 2:  # already grayscale
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def preprocess(self) -> np.ndarray:
        """
        Use edge detection for contour detection.
        This works better for colorful objects on dark backgrounds.
        """
        blurred = cv2.GaussianBlur(self.gray_image, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        return edges

    def find_contours(self, edge_image: np.ndarray) -> None:
        """Find contours in the image."""
        contours, hierarchy = cv2.findContours(
            edge_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        self.contours = contours
        self.hierarchy = hierarchy

    @staticmethod
    def classify_shape(contour: np.ndarray) -> str:
        """Classify shape based on number of vertices."""
        approx = cv2.approxPolyDP(
            contour, 0.04 * cv2.arcLength(contour, True), True
        )
        vertices = len(approx)

        if vertices == 3:
            return "Triangle"
        elif vertices == 4:
            # Distinguish between square and rectangle
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            if 0.95 <= aspect_ratio <= 1.05:
                return "Square"
            return "Rectangle"
        elif vertices == 5:
            return "Pentagon"
        elif vertices > 5:
            return "Circle"
        return "Unknown"

    def analyze_and_draw(self) -> np.ndarray:
        """Draw contours and classify shapes."""
        output = self.original_image.copy()

        for i, contour in enumerate(self.contours):
            area = cv2.contourArea(contour)
            if area < 500:  # ignore small noise
                continue

            perimeter = cv2.arcLength(contour, True)
            x, y, w, h = cv2.boundingRect(contour)

            shape_name = self.classify_shape(contour)

            # Draw contour and bounding box
            cv2.drawContours(output, [contour], -1, (0, 255, 0), 2)
            cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Label shape
            cv2.putText(
                output,
                shape_name,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

            print(
                f"Shape #{i+1}: {shape_name}, "
                f"Area={area:.2f}, Perimeter={perimeter:.2f}, "
                f"BoundingBox=({x}, {y}, {w}, {h})"
            )

        return output

    def run(self) -> None:
        """Full pipeline."""
        edges = self.preprocess()
        self.find_contours(edges)

        print(f"Total contours detected: {len(self.contours)}")

        result = self.analyze_and_draw()
        cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Edges", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Detected Shapes", cv2.WINDOW_NORMAL)


        cv2.imshow("Original", self.original_image)
        cv2.imshow("Edges", edges)
        cv2.imshow("Detected Shapes", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    """Main entry."""
    image_path = "/home/hp/Documents/Daily_Task/Day_2/Assets/shapes_1.jpg"
    analyzer = ContourAnalyzer(image_path)
    analyzer.run()


if __name__ == "__main__":
    main()
