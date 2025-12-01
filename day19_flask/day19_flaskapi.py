from flask import Flask, request, jsonify
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'gif'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max size


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def detect_shape(contour):
    """Shape detection logic identical to your camera script."""
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, closed=True)

    if len(approx) == 3:
        return "triangle"

    if len(approx) == 4:
        x, y, bw, bh = cv2.boundingRect(approx)
        aspect_ratio = bw / float(bh)
        if 0.95 <= aspect_ratio <= 1.05:
            return "square"
        return "rectangle"

    area = cv2.contourArea(contour)
    if area == 0:
        return "unknown"

    circularity = 4 * np.pi * area / (peri * peri)
    if circularity > 0.85:
        return "circle"

    return "polygon"


def process_image(img):
    """Process the uploaded image EXACTLY like your webcam code."""
    h, w = img.shape[:2]


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, hierarchy = cv2.findContours(
        edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    if hierarchy is None:
        return {
            'total_contours': 0,
            'detected_shapes': 0,
            'shapes': [],
            'image_dimensions': {'width': w, 'height': h}
        }

    hierarchy = hierarchy[0]


    children = [[] for _ in contours]
    for idx, h_data in enumerate(hierarchy):
        parent = h_data[3]
        if parent != -1:
            children[parent].append(idx)

    shapes_data = []

    def process_contour(idx, depth=0):
        contour = contours[idx]
        area = cv2.contourArea(contour)

        if area < 300:  
            return

        shape_name = detect_shape(contour)
        perimeter = cv2.arcLength(contour, True)
        x, y, bw, bh = cv2.boundingRect(contour)

        shapes_data.append({
            "shape": shape_name,
            "area": float(area),
            "perimeter": float(perimeter),
            "bounding_box": {
                "x": int(x), "y": int(y),
                "width": int(bw), "height": int(bh)
            },
            "depth": depth,           
            "children": children[idx] 
        })

        for ch in children[idx]:
            process_contour(ch, depth + 1)


    for i in range(len(contours)):
        if hierarchy[i][3] == -1:
            process_contour(i)

    return {
        "total_contours": len(contours),
        "detected_shapes": len(shapes_data),
        "shapes": shapes_data,
        "image_dimensions": {"width": w, "height": h}
    }

@app.route('/detect-shapes', methods=['POST'])
def detect_shapes():
    if 'image' not in request.files:
        return jsonify({
            "error": "No image file provided",
            "message": "Upload using field 'image'"
        }), 400

    file = request.files['image']

    if file.filename == "":
        return jsonify({
            "error": "No file selected"
        }), 400

    if not allowed_file(file.filename):
        return jsonify({
            "error": "Invalid file type",
            "allowed_types": list(ALLOWED_EXTENSIONS)
        }), 400

    try:

        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({
                "error": "Invalid image file"
            }), 400

        result = process_image(img)

        return jsonify({
            "success": True,
            "data": result
        }), 200

    except Exception as e:
        return jsonify({
            "error": "Processing error",
            "message": str(e)
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "Shape Detection API"
    }), 200


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
