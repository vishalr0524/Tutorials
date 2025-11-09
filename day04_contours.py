#contours.py
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
    def visualize_color_confidence(image: np.ndarray, contour: np.ndarray, color_name: str) -> np.ndarray:
        """Create histogram showing color distribution and confidence."""
        # Extract ring pixels (same logic as detect_color)
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
        
        # Create histogram figure
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle(f'Color Analysis: {color_name}', fontsize=14, fontweight='bold')
        
        # BGR histogram
        colors_bgr = ('b', 'g', 'r')
        for i, col in enumerate(colors_bgr):
            hist = cv2.calcHist([ring_pixels], [i], None, [256], [0, 256])
            axes[0, 0].plot(hist, color=col, label=col.upper())
        axes[0, 0].set_title('BGR Distribution')
        axes[0, 0].set_xlabel('Intensity')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        
        # HSV histogram
        hsv_pixels = cv2.cvtColor(ring_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
        axes[0, 1].hist(hsv_pixels[:, 0], bins=180, color='purple', alpha=0.7)
        axes[0, 1].set_title('Hue Distribution')
        axes[0, 1].set_xlabel('Hue (0-180)')
        axes[0, 1].set_ylabel('Frequency')
        
        # Color frequency bar chart
        unique_colors, counts = np.unique(ring_pixels.reshape(-1, 3), axis=0, return_counts=True)
        top_5_idx = np.argsort(counts)[-5:][::-1]
        top_colors = unique_colors[top_5_idx]
        top_counts = counts[top_5_idx]
        
        color_patches = [top_colors[i][::-1]/255.0 for i in range(len(top_colors))]  # BGR to RGB
        axes[1, 0].bar(range(len(top_counts)), top_counts, color=color_patches)
        axes[1, 0].set_title('Top 5 Dominant Colors')
        axes[1, 0].set_xlabel('Color Rank')
        axes[1, 0].set_ylabel('Pixel Count')
        
        # Confidence metrics
        total_pixels = len(ring_pixels)
        dominant_count = counts[np.argmax(counts)]
        confidence = (dominant_count / total_pixels) * 100
        
        metrics_text = f"Total Pixels: {total_pixels}\n"
        metrics_text += f"Dominant Color Pixels: {dominant_count}\n"
        metrics_text += f"Confidence: {confidence:.1f}%\n"
        metrics_text += f"Unique Colors: {len(unique_colors)}\n"
        metrics_text += f"Classification: {color_name}"
        
        axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Confidence Metrics')
        
        plt.tight_layout()
        return fig

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
            contour, 0.04 * cv2.arcLength(contour, True), True
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
            if area < 500:
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
