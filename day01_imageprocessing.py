import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class ImageProcessor:
    image_path: str
    image: np.ndarray | None = None

    def load(self) -> np.ndarray:
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            raise FileNotFoundError(f"Cannot load image: {self.image_path}")
        return self.image

    def to_rgb(self) -> np.ndarray:
        self._ensure_loaded()
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

    def to_gray(self) -> np.ndarray:
        self._ensure_loaded()
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def to_hsv(self) -> np.ndarray:
        self._ensure_loaded()
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

    def split_channels(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._ensure_loaded()
        b, g, r = cv2.split(self.image)
        return r, g, b 

    def crop(self, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
        self._ensure_loaded()
        return self.image[y1:y2, x1:x2]

    def save(self, output_path: str, img: np.ndarray) -> None:
        cv2.imwrite(output_path, img)

    def _ensure_loaded(self):
        if self.image is None:
            raise ValueError("Image not loaded. Call load() first.")


if __name__ == "__main__":
    processor = ImageProcessor("/home/hp/Documents/Daily_Task/Day_2/Assets/balls.jpg")

    img = processor.load()
    rgb = processor.to_rgb()
    gray = processor.to_gray()
    hsv = processor.to_hsv()

    r, g, b = processor.split_channels()


    h, w = img.shape[:2]
    crop_example = processor.crop(0, 0, w // 2, h // 2)


    processor.save("output_rgb.png", rgb)
    processor.save("output_gray.png", gray)
    processor.save("output_hsv.png", hsv)
    processor.save("output_crop.png", crop_example)

    print("Processing complete. Files saved.")
