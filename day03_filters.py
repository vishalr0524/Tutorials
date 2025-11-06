import cv2
import numpy as np
import matplotlib.pyplot as plt


class ImageLoader:
    """Handles image loading and color conversion."""

    def __init__(self, image_path: str):
        self.image_path = image_path
        self.image = None

    def load_image(self) -> np.ndarray:
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            raise FileNotFoundError(f"Image not found at {self.image_path}")
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        return self.image


class NoiseAdder:
    """Adds different types of noise to an image."""

    @staticmethod
    def add_gaussian_noise(image: np.ndarray, mean=0, sigma=25) -> np.ndarray:
        noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)
        noisy_image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return noisy_image

    @staticmethod
    def add_salt_pepper_noise(image: np.ndarray, amount=0.02) -> np.ndarray:
        noisy = np.copy(image)
        num_salt = np.ceil(amount * image.size * 0.5)
        num_pepper = np.ceil(amount * image.size * 0.5)

        # Salt noise (white pixels)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape[:2]]
        noisy[coords[0], coords[1]] = 255

        # Pepper noise (black pixels)
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape[:2]]
        noisy[coords[0], coords[1]] = 0

        return noisy


class FilterApplier:
    """Applies different filters for noise removal and sharpening."""

    @staticmethod
    def apply_box_filter(image: np.ndarray, ksize=5) -> np.ndarray:
        return cv2.boxFilter(image, -1, (ksize, ksize))

    @staticmethod
    def apply_mean_filter(image: np.ndarray, ksize=5) -> np.ndarray:
        return cv2.blur(image, (ksize, ksize))

    @staticmethod
    def apply_gaussian_filter(image: np.ndarray, ksize=5, sigma=1) -> np.ndarray:
        return cv2.GaussianBlur(image, (ksize, ksize), sigma)
    
    @staticmethod
    def apply_grayscale(image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    @staticmethod
    def apply_median_filter(image: np.ndarray, ksize=5) -> np.ndarray:
        return cv2.medianBlur(image, ksize)

    @staticmethod
    def apply_sharpening(image: np.ndarray) -> np.ndarray:
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        return cv2.filter2D(image, -1, kernel)


class Visualizer:
    """Displays original, noisy, and filtered images side-by-side."""

    @staticmethod
    def show_results(images_dict: dict):
        plt.figure(figsize=(15, 10))
        for i, (title, img) in enumerate(images_dict.items(), 1):
            plt.subplot(2, 4, i)
            plt.imshow(img)
            plt.title(title)
            plt.axis("off")
        plt.tight_layout()
        plt.show()


def main():

    loader = ImageLoader("/home/hp/Documents/Daily_Task/Day_2/Assets/Test.png")
    original = loader.load_image()


    noisy_gauss = NoiseAdder.add_gaussian_noise(original)
    noisy_sp = NoiseAdder.add_salt_pepper_noise(original)

    filtered_box = FilterApplier.apply_box_filter(noisy_gauss)
    filtered_mean = FilterApplier.apply_mean_filter(noisy_gauss)
    filtered_gauss = FilterApplier.apply_gaussian_filter(noisy_gauss)
    filtered_median = FilterApplier.apply_median_filter(noisy_sp) 
    filtered_sharp = FilterApplier.apply_sharpening(original)

 
    results = {
        "Original": original,
        "Gaussian Noise": noisy_gauss,
        "Salt & Pepper Noise": noisy_sp,
        "Box Filter": filtered_box,
        "Mean Filter": filtered_mean,
        "Gaussian Filter": filtered_gauss,
        "Median Filter": filtered_median,
        "Sharpened": filtered_sharp,
    }

    Visualizer.show_results(results)


if __name__ == "__main__":
    main()
