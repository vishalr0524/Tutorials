import cv2
import numpy as np
from typing import Optional, Tuple, List


class ImageLoader:
    """Utility class to load images from disk."""

    @staticmethod
    def load(path: str, color: bool = True) -> np.ndarray:
        flag = cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE
        img = cv2.imread(path, flag)
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        return img


class LineDetector:
    """Detects lines using Probabilistic Hough Transform."""

    def __init__(self, rho: float = 1, theta: float = np.pi / 180,
                 threshold: int = 80, min_line_length: int = 50,
                 max_line_gap: int = 10) -> None:
        self.rho = rho
        self.theta = theta
        self.threshold = threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap

    def detect(self, img: np.ndarray) -> List[np.ndarray]:
        edges = cv2.Canny(img, 40, 180)
        lines = cv2.HoughLinesP(
            edges,
            rho=self.rho,
            theta=self.theta,
            threshold=self.threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap,
        )
        return [] if lines is None else lines

    @staticmethod
    def draw_lines(img: np.ndarray, lines: List[np.ndarray]) -> np.ndarray:
        out = img.copy()
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
        return out


class CircleDetector:
    """Detects circles using Hough Circle Transform."""

    def __init__(self, dp: float = 1.2, min_dist: float = 30,
                 param1: float = 100, param2: float = 30,
                 min_radius: int = 10, max_radius: int = 60) -> None:
        self.dp = dp
        self.min_dist = min_dist
        self.param1 = param1
        self.param2 = param2
        self.min_radius = min_radius
        self.max_radius = max_radius

    def detect(self, img: np.ndarray) -> Optional[np.ndarray]:
        gray = cv2.medianBlur(img, 5)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=self.dp,
            minDist=self.min_dist,
            param1=self.param1,
            param2=self.param2,
            minRadius=self.min_radius,
            maxRadius=self.max_radius,
        )
        return circles

    @staticmethod
    def draw_circles(img: np.ndarray, circles: Optional[np.ndarray]) -> np.ndarray:
        out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for c in circles[0, :]:
                x, y, r = c
                cv2.circle(out, (x, y), r, (0, 255, 0), 2)
                cv2.circle(out, (x, y), 2, (0, 0, 255), 3)
        return out


class HoughPipeline:
    """Pipeline to run both line and circle detection."""

    def __init__(self, lane_img_path: str, coin_img_path: str) -> None:
        self.lane_img_path = lane_img_path
        self.coin_img_path = coin_img_path

        self.line_detector = LineDetector()
        self.circle_detector = CircleDetector()

    def run(self) -> Tuple[np.ndarray, np.ndarray]:
        # Line detection
        lane_img = ImageLoader.load(self.lane_img_path)
        lines = self.line_detector.detect(lane_img)
        lane_out = self.line_detector.draw_lines(lane_img, lines)

        # Circle detection
        coin_img = ImageLoader.load(self.coin_img_path, color=False)
        circles = self.circle_detector.detect(coin_img)
        coin_out = self.circle_detector.draw_circles(coin_img, circles)

        return lane_out, coin_out


if __name__ == "__main__":
    pipeline = HoughPipeline("/home/hp/Documents/Daily_Task/Day_2/templates/lane.jpg", "/home/hp/Documents/Daily_Task/Day_2/Assets/coin.jpeg")
    lane_result, coin_result = pipeline.run()

    cv2.imshow("Detected Lines", lane_result)
    cv2.imshow("Detected Circles", coin_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
