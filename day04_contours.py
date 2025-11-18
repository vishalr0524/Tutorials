import cv2
import numpy as np
from typing import Tuple, List


class ContourProcessor:
    def __init__(self, image_path: str):
        self.image_path = image_path
        self.original = cv2.imread(image_path)
        if self.original is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        self.gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        self.binary = None

    def apply_threshold(self, thresh_val: int = 127) -> np.ndarray:
        _, self.binary = cv2.threshold(self.gray, thresh_val, 255, cv2.THRESH_BINARY)
        return self.binary

    def get_contours(self, mode: int, method: int = cv2.CHAIN_APPROX_SIMPLE) -> Tuple[List[np.ndarray], np.ndarray]:
        if self.binary is None:
            raise ValueError("Threshold not applied before finding contours.")
        contours, hierarchy = cv2.findContours(self.binary, mode, method)
        return contours, hierarchy

    def analyze_hierarchy(self, contours: List[np.ndarray], hierarchy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        extern = np.zeros(self.original.shape, dtype=np.uint8)
        intern = np.zeros(self.original.shape, dtype=np.uint8)

        for i, cnt in enumerate(contours):
            # External contour
            if hierarchy[0][i][3] == -1:
                cv2.drawContours(extern, contours, i, (0, 255, 0), 2)
            # Internal contour
            else:
                cv2.drawContours(intern, contours, i, (0, 0, 255), 2)

        return extern, intern

    def draw_properties(self, contours: List[np.ndarray]) -> np.ndarray:
        output = self.original.copy()

        for cnt in contours:
            area = cv2.contourArea(cnt)
            peri = cv2.arcLength(cnt, True)
            x, y, w, h = cv2.boundingRect(cnt)

            cv2.rectangle(output, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(output, f"A:{int(area)} P:{int(peri)}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return output


def main():
    img_path = "/home/hp/Documents/Daily_Task/Day_2/Assets/contours.png"  
    processor = ContourProcessor(img_path)

    processor.apply_threshold(120)

    # cv2.namedWindow("Binary", cv2.WINDOW_NORMAL)
    # cv2.imshow("Binary", processor.binary)

    retrieval_modes = {
        "RETR_TREE": cv2.RETR_TREE,
        "RETR_LIST": cv2.RETR_LIST,
        "RETR_EXTERNAL": cv2.RETR_EXTERNAL,
        "RETR_CCOMP": cv2.RETR_CCOMP,
    }

    # mode = cv2.RETR_TREE
    # mode = cv2.RETR_LIST
    # mode = cv2.RETR_EXTERNAL
    # mode = cv2.RETR_CCOMP

    method_name = "RETR_TREE"  
    mode = retrieval_modes[method_name]

    print(f"--- Mode: {method_name} ---")
    contours, hierarchy = processor.get_contours(mode)
    print("Hierarchy:", hierarchy)

    extern, intern = processor.analyze_hierarchy(contours, hierarchy)
    cv2.imshow(f"{method_name} - External Contours", extern)
    cv2.imshow(f"{method_name} - Internal Contours", intern)

    props_img = processor.draw_properties(contours)
    cv2.imshow(f"{method_name} - Properties", props_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()(f"\n--- Mode: {method_name} ---")
    contours, hierarchy = processor.get_contours(mode)
    print("Hierarchy:\n", hierarchy)

    extern, intern = processor.analyze_hierarchy(contours, hierarchy)
    cv2.imshow(f"External Contours", extern)
    cv2.imshow(f"Internal Contours", intern)

    props_img = processor.draw_properties(contours)
    cv2.imshow(f"Properties", props_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("Processing complete. Windows displayed.")


if __name__ == "__main__":
    main()
