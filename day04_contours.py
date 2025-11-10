import cv2
import numpy as np
import matplotlib.pyplot as plt

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
    
    @staticmethod
    def plot_color_histogram(contour_data, image):
        import matplotlib.pyplot as plt
        import numpy as np
        import cv2

        bucket = {}

        # Collect ring pixels for each detected shape
        for entry in contour_data:
            shape = f"Shape #{entry['index']}: {entry['color']} {entry['shape']}"
            contour = entry["contour"]

            ring_pixels = ContourAnalyzer._get_ring_pixels(image, contour)
            if len(ring_pixels) == 0:
                continue

            if shape not in bucket:
                bucket[shape] = []

            bucket[shape].extend(ring_pixels.tolist())

        if not bucket:
            print("No ring pixel data found.")
            return

        shape_names = []
        h_min = []
        h_max = []
        s_min = []
        s_max = []
        v_min = []
        v_max = []

        # Compute HSV min/max per shape
        for shape, pixels in bucket.items():
            px = np.array(pixels)

            hsv = cv2.cvtColor(
                px.reshape(-1, 1, 3).astype(np.uint8),
                cv2.COLOR_BGR2HSV
            ).reshape(-1, 3)

            h_vals, s_vals, v_vals = hsv[:, 0], hsv[:, 1], hsv[:, 2]

            shape_names.append(shape)
            h_min.append(int(np.min(h_vals)))
            h_max.append(int(np.max(h_vals)))
            s_min.append(int(np.min(s_vals)))
            s_max.append(int(np.max(s_vals)))
            v_min.append(int(np.min(v_vals)))
            v_max.append(int(np.max(v_vals)))

        # -------------------------------------------------------
        # OPTION D: Min–max vertical line plot + labeled HSV values
        # -------------------------------------------------------
        x = np.arange(len(shape_names))
        plt.figure(figsize=(20, 7))

        for i in range(len(shape_names)):
            # H range (red)
            plt.plot([i, i], [h_min[i], h_max[i]], color='red', linewidth=4)

            # S range (green)
            plt.plot([i + 0.12, i + 0.12], [s_min[i], s_max[i]], color='green', linewidth=4)

            # V range (blue)
            plt.plot([i - 0.12, i - 0.12], [v_min[i], v_max[i]], color='blue', linewidth=4)

            # Add MIN label (bottom)
            min_y = min(h_min[i], s_min[i], v_min[i]) - 5
            plt.text(
                i - 0.35, min_y,
                f"({h_min[i]}, {s_min[i]}, {v_min[i]})",
                fontsize=7, ha='left', va='top'
            )

            # Add MAX label (top)
            max_y = max(h_max[i], s_max[i], v_max[i]) + 5
            plt.text(
                i - 0.35, max_y,
                f"({h_max[i]}, {s_max[i]}, {v_max[i]})",
                fontsize=7, ha='left', va='bottom'
            )

        plt.xticks(x, shape_names, rotation=45, ha='right')
        plt.ylabel("HSV Values")
        plt.title("HSV Min–Max Range per Shape with HSV Labels (H=red, S=green, V=blue)")
        plt.tight_layout()
        plt.show()




        
    @staticmethod
    def _get_ring_pixels(image: np.ndarray, contour: np.ndarray) -> np.ndarray:
        """Helper to get ring pixels for use in aggregation."""
        outer_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(outer_mask, [contour], -1, 255, -1)
        area = cv2.contourArea(contour)
        erosion_size = max(2, int(np.sqrt(area) * 0.10))
        kernel = np.ones((erosion_size, erosion_size), np.uint8)
        inner_mask = cv2.erode(outer_mask, kernel, iterations=1)
        ring_mask = cv2.subtract(outer_mask, inner_mask)
        ring_pixels = image[ring_mask > 0]
        
        if len(ring_pixels) < 50:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                sample_radius = max(3, erosion_size)
                y1, y2 = max(0, cy-sample_radius), min(image.shape[0], cy+sample_radius)
                x1, x2 = max(0, cx-sample_radius), min(image.shape[1], cx+sample_radius)
                ring_pixels = image[y1:y2, x1:x2].reshape(-1, 3)
                
        return ring_pixels

    def find_contours(self, edge_image: np.ndarray, include_inner: bool = False) -> None:
        """Find contours in the image."""
        mode = cv2.RETR_TREE if include_inner else cv2.RETR_EXTERNAL
        contours, hierarchy = cv2.findContours(
            edge_image, mode, cv2.CHAIN_APPROX_SIMPLE
        )
        self.contours = contours
        self.hierarchy = hierarchy

    @staticmethod
    def detect_color(image: np.ndarray, contour: np.ndarray) -> str:
        """Detect color from inner ring using frequency analysis."""

        outer_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(outer_mask, [contour], -1, 255, -1)
        
        area = cv2.contourArea(contour)
        erosion_size = max(2, int(np.sqrt(area) * 0.10))  # 5% of shape size
        kernel = np.ones((erosion_size, erosion_size), np.uint8)
        inner_mask = cv2.erode(outer_mask, kernel, iterations=1)
        
        ring_mask = cv2.subtract(outer_mask, inner_mask)
        ring_pixels = image[ring_mask > 0]
        
        if len(ring_pixels) < 50:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                sample_radius = max(3, erosion_size)
                y1, y2 = max(0, cy-sample_radius), min(image.shape[0], cy+sample_radius)
                x1, x2 = max(0, cx-sample_radius), min(image.shape[1], cx+sample_radius)
                ring_pixels = image[y1:y2, x1:x2].reshape(-1, 3)
        
        unique_colors, counts = np.unique(ring_pixels.reshape(-1, 3), axis=0, return_counts=True)
        dominant_bgr = unique_colors[np.argmax(counts)]
        
        hsv = cv2.cvtColor(np.uint8([[dominant_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
        h, s, v = hsv[0], hsv[1], hsv[2]
        
        if s < 30:
            return "Black" if v < 50 else ("White" if v > 200 else "Gray")
        
        if h < 10 or h > 170: return "Red"
        elif h < 25: return "Orange"
        elif h < 40: return "Yellow"
        elif h < 80: return "Green"
        elif h < 100: return "Cyan"
        elif h < 130: return "Blue"
        elif h < 150: return "Purple"
        else: return "Pink"

    @staticmethod
    def classify_shape(contour: np.ndarray) -> str:
        """Classify shape based on number of vertices."""
        approx = cv2.approxPolyDP(
            contour, 0.002 * cv2.arcLength(contour, True), True
        )
        vertices = len(approx)

        if vertices == 3:
            return "Triangle"
        elif vertices == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            if 0.95 <= aspect_ratio <= 1.05:
                return "Square"
            return "Rectangle"
        elif vertices == 5:
            return "Pentagon"
        elif vertices >= 8:  # ← CHANGED: More reliable threshold
            # Additional check: circularity
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity > 0.7:  # Circle-like shapes
                return "Circle"
            return "Polygon"
        elif 6 <= vertices <= 7:
            return "Hexagon/Pentagon"  # Ambiguous cases
        return "Unknown"

    def analyze_and_draw(self, show_inner: bool = False, return_contour_data: bool = False) -> tuple:
        """Draw contours and classify shapes."""
        output = self.original_image.copy()
        contour_data = []  # Store contour info for histogram

        if self.hierarchy is None:
            print("No hierarchy info available.")
            return (output, contour_data) if return_contour_data else output

        for i, contour in enumerate(self.contours):
            area = cv2.contourArea(contour)
            if area < 1000:
                continue

            if not show_inner and self.hierarchy[0][i][3] != -1:
                continue

            if self.hierarchy[0][i][3] != -1:
                parent_idx = self.hierarchy[0][i][3]
                parent_area = cv2.contourArea(self.contours[parent_idx])
                area_ratio = area / parent_area if parent_area > 0 else 0
                if area_ratio > 0.8:
                    continue

            perimeter = cv2.arcLength(contour, True)
            x, y, w, h = cv2.boundingRect(contour)
            shape_name = self.classify_shape(contour)
            color_name = self.detect_color(self.original_image, contour)
            label = f"{color_name} {shape_name}" + (" (inner)" if self.hierarchy[0][i][3] != -1 else "")

            # Store contour data
            if return_contour_data:
                contour_data.append({
                    'contour': contour,
                    'color': color_name,
                    'shape': shape_name,
                    'index': i
                })

            if self.hierarchy[0][i][3] == -1:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)

            cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
            label_y = y - 15 if y > 30 else y + h + 25
            label_x = max(5, x)
            cv2.putText(output, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            print(f"Shape #{i+1}: {color_name} {shape_name}, Area={area:.2f}, Perimeter={perimeter:.2f}, BoundingBox=({x}, {y}, {w}, {h})")

        return (output, contour_data) if return_contour_data else output

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
