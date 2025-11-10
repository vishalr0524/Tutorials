import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
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

def process_image_with_stages(image_path: str):
    """Process single image and show all intermediate stages."""
    try:
        loader = ImageLoader(image_path)
        img = loader.load_image()
        cvt = FilterApplier.apply_grayscale(img)
        blur = FilterApplier.apply_gaussian_filter(cvt)

        sobel = SobelDetector(image=blur)
        sx, sy, smag = sobel.detect()
        # canny = CannyDetector(image=blur)
        # can_edge = canny.detect()

        smag_closed = cv2.morphologyEx(smag, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

        adapt = Thresholding(image=smag_closed)
        cvt_adapt = adapt.otsu_threshold()

        ctr = ContourAnalyzer(image=img)
        ctr.find_contours(edge_image=smag_closed, include_inner=True)
        shape = ctr.analyze_and_draw(show_inner=True)

        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Filter", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Edge", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Threshold", cv2.WINDOW_NORMAL)
        cv2.namedWindow("MorphClosed", cv2.WINDOW_NORMAL)
        cv2.namedWindow("contours", cv2.WINDOW_NORMAL)

        cv2.imshow("Image", img)
        cv2.imshow("Filter", blur)
        cv2.imshow("Edge", smag)
        cv2.imshow("Threshold", cvt_adapt)
        cv2.imshow("MorphClosed", smag_closed)
        cv2.imshow("contours", shape)

        print("Press 'q' to quit...")
        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error: {e}")

def process_webcam():
    """Process webcam feed in real-time."""
    try:
        cap = cv2.VideoCapture(0)  # 0 for default webcam
        
        if not cap.isOpened():
            raise IOError("Cannot open webcam")
        
        cv2.namedWindow("Webcam Feed", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Detected Shapes", cv2.WINDOW_NORMAL)
        
        print("Press 'q' to quit, 's' to save current frame...")
        
        while True:
            ret, img = cap.read()
            
            if not ret:
                print("Failed to grab frame")
                break
            
            # Process frame
            cvt = FilterApplier.apply_grayscale(img)
            blur = FilterApplier.apply_gaussian_filter(cvt)
            
            sobel = SobelDetector(image=blur)
            sx, sy, smag = sobel.detect()
            canny = CannyDetector(image=blur)
            can_edge = canny.detect()
            
            # But apply Canny-like thresholding
            # Lower threshold for weak edges, upper for strong edges
            lower_thresh = np.percentile(smag[smag > 0], 60)  # 20th percentile
            upper_thresh = np.percentile(smag[smag > 0], 100)  # 80th percentile
            
            # Create binary edge map
            binary_edges = np.zeros_like(smag)
            binary_edges[smag > lower_thresh] = 255
            
            # Clean up with morphology
            edges_closed = cv2.morphologyEx(can_edge, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
            
            ctr = ContourAnalyzer(image=img)
            ctr.find_contours(edge_image=edges_closed, include_inner=True)
            shape, contour_data = ctr.analyze_and_draw(show_inner=True, return_contour_data=True)
            
            # # Show histograms for each detected shape
            # for data in contour_data:
            #     fig = ctr.visualize_color_confidence(img, data['contour'], data['color'])
            #     plt.show(block=True)  # Non-blocking
            
            # Display results
            # cv2.imshow("Webcam Feed", img)
            cv2.imshow("Detected Shapes", shape)
            
            # Key controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite("captured_frame.jpg", img)
                cv2.imwrite("detected_shapes.jpg", shape)
                print("Frame saved!")
        
        cap.release()
        cv2.destroyAllWindows()
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except cv2.error as e:
        print(f"OpenCV Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def main():
    parser = argparse.ArgumentParser(description='Shape Detection - Webcam or Image Mode')
    parser.add_argument('--input', type=str, help='Path to input image (if not provided, uses webcam)')
    
    args = parser.parse_args()
    
    if args.input:
        # Single image mode with all stages
        print(f"Processing image: {args.input}")
        process_image_with_stages(args.input)
    else:
        # Webcam mode
        print("Starting webcam mode...")
        process_webcam()

if __name__ == "__main__":
    main()