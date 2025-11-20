import argparse
import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class ImageLoader:

    path: str
    color: Optional[np.ndarray] = None
    gray: Optional[np.ndarray] = None

    def load(self) -> None:
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Image not found: {self.path}")

        img = cv2.imread(self.path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Unable to read image: {self.path}")

        self.color = img
        self.gray = cv2.cvtColor(self.color, cv2.COLOR_BGR2GRAY)

    def basename(self) -> str:
        return os.path.splitext(os.path.basename(self.path))[0]

class FeatureDetector:

    def detect_and_compute(self, image: np.ndarray) -> Tuple[list, np.ndarray]:
        """Detect keypoints and compute descriptors.

        Returns
        -------
        keypoints : list
            List of cv2.KeyPoint objects.
        descriptors : np.ndarray
            Descriptor array (N x D) or None.
        """
        raise NotImplementedError


class ORBDetector(FeatureDetector):

    def __init__(self, n_features: int = 500) -> None:
        self.n_features = n_features
        # create ORB instance
        self._detector = cv2.ORB_create(nfeatures=self.n_features)

    def detect_and_compute(self, image: np.ndarray) -> Tuple[list, np.ndarray]:
        keypoints, descriptors = self._detector.detectAndCompute(image, None)
        return keypoints, descriptors


class SIFTDetector(FeatureDetector):

    def __init__(self, n_features: int = 0) -> None:
        sift = None
        if hasattr(cv2, "SIFT_create"):
            sift = cv2.SIFT_create(nfeatures=n_features)
        else:
            try:
                sift = cv2.SIFT_create(nfeatures=n_features) 
            except Exception:
                sift = None

        if sift is None:
            raise RuntimeError(
                "SIFT is not available in your OpenCV build. "
                "Install opencv-contrib-python or use ORB instead."
            )

        self._detector = sift

    def detect_and_compute(self, image: np.ndarray) -> Tuple[list, np.ndarray]:
        keypoints, descriptors = self._detector.detectAndCompute(image, None)
        return keypoints, descriptors

class Visualizer:

    @staticmethod
    def draw_keypoints_bgr(image_bgr: np.ndarray, keypoints: list) -> np.ndarray:
        return cv2.drawKeypoints(
            image_bgr, keypoints, outImage=None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

    @staticmethod
    def show_side_by_side(original_bgr: np.ndarray, kp_bgr: np.ndarray, title: str) -> None:
        orig_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
        kp_rgb = cv2.cvtColor(kp_bgr, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(12, 6))
        plt.suptitle(title)

        plt.subplot(1, 2, 1)
        plt.imshow(orig_rgb)
        plt.axis('off')
        plt.title('Original')

        plt.subplot(1, 2, 2)
        plt.imshow(kp_rgb)
        plt.axis('off')
        plt.title('Keypoints')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def save_image(path: str, image_bgr: np.ndarray) -> None:
        cv2.imwrite(path, image_bgr)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Feature detection demo: ORB & SIFT')
    parser.add_argument('--image', '-i', required=True, help='Path to input image')
    parser.add_argument(
        '--detector', '-d', choices=('sift', 'orb', 'both'), default='both',
        help='Which detector to run (default: both)'
    )
    parser.add_argument('--nfeatures', type=int, default=500, help='Max features for ORB (default: 500)')
    return parser.parse_args()


def _save_descriptors(base: str, name: str, descriptors: Optional[np.ndarray]) -> None:
    """Save descriptors as .npy if available, otherwise skip."""
    if descriptors is None:
        print(f'No descriptors for {name}; skipping save.')
        return

    out_path = f"{base}_{name}_desc.npy"
    np.save(out_path, descriptors)
    print(f'Saved descriptors: {out_path} (shape={descriptors.shape})')


def run(image_path: str, detector_choice: str, nfeatures: int) -> None:
    loader = ImageLoader(path=image_path)
    loader.load()

    basename = loader.basename()

    detectors = []

    if detector_choice in ('orb', 'both'):
        detectors.append(('orb', ORBDetector(n_features=nfeatures)))

    if detector_choice in ('sift', 'both'):
        try:
            detectors.append(('sift', SIFTDetector()))
        except RuntimeError as exc:
            print(f'Warning: {exc}')
            if detector_choice == 'sift':
                raise

    if not detectors:
        print('No detectors available. Exiting.')
        return

    for name, detector in detectors:
        print(f'Running detector: {name}')
        keypoints, descriptors = detector.detect_and_compute(loader.gray)
        print(f'Found {len(keypoints)} keypoints using {name}')

        kp_image = Visualizer.draw_keypoints_bgr(loader.color, keypoints)

        # Save visual and descriptors
        save_vis_path = f"{basename}_{name}_kp.png"
        Visualizer.save_image(save_vis_path, kp_image)
        print(f'Saved visualization: {save_vis_path}')

        _save_descriptors(basename, name, descriptors)

        # Show side-by-side for quick inspection
        Visualizer.show_side_by_side(loader.color, kp_image, title=f'{name.upper()} keypoints')


def main() -> None:
    args = parse_args()

    try:
        run(args.image, args.detector, args.nfeatures)
    except Exception as exc:  
        print(f'Error: {exc}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
