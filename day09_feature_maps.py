import argparse
import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import numpy as np


logger = logging.getLogger("feature_matching")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(levelname)s: %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


@dataclass
class MatchResult:
    """Container for the results of a feature matching run."""

    keypoints1: List[cv2.KeyPoint]
    keypoints2: List[cv2.KeyPoint]
    good_matches: List[cv2.DMatch]
    homography: Optional[np.ndarray]
    mask: Optional[np.ndarray]


class FeatureMatcher:

    def __init__(
        self,
        detector: str = "sift",
        matcher: str = "flann",
        ratio: float = 0.75,
        ransac_thresh: float = 5.0,
    ) -> None:
        self.detector_name = detector.lower()
        self.matcher_name = matcher.lower()
        self.ratio = float(ratio)
        self.ransac_thresh = float(ransac_thresh)
        self.detector = self._create_detector(self.detector_name)

    @staticmethod
    def _create_detector(name: str):
        name = name.lower()
        if name == "sift":
            try:
                return cv2.SIFT_create()
            except AttributeError:
                try:
                    return cv2.xfeatures2d.SIFT_create()
                except Exception:
                    logger.warning("SIFT not available — falling back to ORB.")
                    return cv2.ORB_create(nfeatures=2000)
        elif name == "orb":
            return cv2.ORB_create(nfeatures=2000)
        elif name == "akaze":
            return cv2.AKAZE_create()
        elif name == "brisk":
            return cv2.BRISK_create()
        else:
            logger.warning("Unknown detector '%s'; defaulting to ORB.", name)
            return cv2.ORB_create(nfeatures=2000)

    def _create_matcher(self, descriptor_type: str):
        if self.matcher_name in ["bf", "bfmatcher"]:
            if descriptor_type == "float":
                return cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        if descriptor_type == "float":
            index_params = {"algorithm": 1, "trees": 5}
        else:
            index_params = {
                "algorithm": 6,
                "table_number": 6,
                "key_size": 12,
                "multi_probe_level": 1,
            }
        search_params = {"checks": 50}
        return cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)

    @staticmethod
    def _is_float_descriptor(descriptor: np.ndarray) -> bool:
        return descriptor is not None and descriptor.dtype in (np.float32, np.float64)

    def detect_and_compute(self, image: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        if image is None:
            raise ValueError("Input image is None")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        if descriptors is None:
            dtype = np.float32 if self.detector_name == "sift" else np.uint8
            descriptors = np.empty((0, 128 if dtype == np.float32 else 32), dtype=dtype)
        logger.info("Detected %d keypoints using %s", len(keypoints), self.detector_name.upper())
        return keypoints, descriptors

    def match(self, img1: np.ndarray, img2: np.ndarray) -> MatchResult:
        kps1, des1 = self.detect_and_compute(img1)
        kps2, des2 = self.detect_and_compute(img2)
        if len(des1) == 0 or len(des2) == 0:
            logger.warning("No descriptors to match.")
            return MatchResult(kps1, kps2, [], None, None)

        descriptor_type = "float" if self._is_float_descriptor(des1) else "binary"
        matcher = self._create_matcher(descriptor_type)
        matches = matcher.knnMatch(des1, des2, k=2)

        good = [m for m, n in matches if m.distance < self.ratio * n.distance]
        logger.info("%d good matches after Lowe's ratio test (%.2f)", len(good), self.ratio)

        homography, mask = None, None
        if len(good) >= 4:
            src_pts = np.float32([kps1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kps2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self.ransac_thresh)
            if homography is not None:
                logger.info("Homography found with %d inliers", int(mask.sum()))

        return MatchResult(kps1, kps2, good, homography, mask)


def draw_matches(img1, img2, result, max_matches=50):
    if result.mask is not None:
        matches_mask = result.mask.ravel().tolist()
    else:
        matches_mask = None

    sorted_matches = sorted(result.good_matches, key=lambda m: m.distance)[:max_matches]
    draw_params = dict(matchColor=None, singlePointColor=(255, 0, 0), flags=2)
    if matches_mask is not None:
        sel_mask = [matches_mask[i] for i in range(len(sorted_matches)) if i < len(matches_mask)]
        draw_params["matchesMask"] = sel_mask

    return cv2.drawMatches(img1, result.keypoints1, img2, result.keypoints2, sorted_matches, None, **draw_params)


def warp_image(img_src, img_dst, homography):
    if homography is None:
        raise ValueError("Homography is None — cannot warp image.")
    h_dst, w_dst = img_dst.shape[:2]
    return cv2.warpPerspective(img_src, homography, (w_dst, h_dst))


def save_image(path, image):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    cv2.imwrite(path, image)
    logger.info("Saved image: %s", path)


def parse_args():
    parser = argparse.ArgumentParser(description="Feature matching demo (BFMatcher & FLANN)")
    parser.add_argument("--img1", required=True, help="Path to first image (source)")
    parser.add_argument("--img2", required=True, help="Path to second image (target)")
    parser.add_argument(
        "--detector",
        choices=["sift", "orb", "akaze", "brisk"],
        default="sift",
        help="Keypoint detector/descriptor",
    )
    parser.add_argument("--matcher", choices=["flann", "bf"], default="flann", help="Matcher to use")
    parser.add_argument("--ratio", type=float, default=0.75, help="Lowe's ratio")
    parser.add_argument("--ransac", type=float, default=5.0, help="RANSAC reprojection threshold")
    parser.add_argument("--min_matches", type=int, default=10, help="Minimum good matches")
    parser.add_argument("--draw", action="store_true", help="Draw and save match visualization")
    parser.add_argument("--max_draw", type=int, default=50, help="Maximum matches to draw")
    return parser.parse_args()


def _friendly_name(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0].replace(" ", "_")


def main():
    args = parse_args()

    img1 = cv2.imread(args.img1)
    img2 = cv2.imread(args.img2)
    if img1 is None or img2 is None:
        raise SystemExit("Could not read one or both images.")

    fm = FeatureMatcher(args.detector, args.matcher, args.ratio, args.ransac)
    result = fm.match(img1, img2)

    out_prefix = _friendly_name(args.img2)
    matches_fname = f"matches_{args.matcher}_{args.detector}_{out_prefix}.jpg"

    if args.draw:
        img_matches = draw_matches(img1, img2, result, args.max_draw)
        save_image(matches_fname, img_matches)

    if result.homography is not None:
        warped = warp_image(img1, img2, result.homography)
        warped_fname = f"warped_{args.detector}_to_{out_prefix}.jpg"
        save_image(warped_fname, warped)

    logger.info("Feature matching complete.")


if __name__ == "__main__":
    main()
