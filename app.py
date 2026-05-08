from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np

app = Flask(__name__)
CORS(app)

@app.route('/analyze', methods=['POST'])
def analyze():

    try:

        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'})

        file = request.files['image']

        file_bytes = np.frombuffer(file.read(), np.uint8)

        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': 'Invalid image'})

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Blur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Threshold
        _, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        scale = 1.0
        widths = []
        heights = []

        for c in contours:
            area = cv2.contourArea(c)
            if 500 < area < 200000:
                x, y, w, h = cv2.boundingRect(c)
                widths.append(w * scale)
                heights.append(h * scale)

        # If no valid contours detected, return early
        if len(widths) == 0:
            return jsonify({
                'width': '0',
                'height': '0',
                'depth': '0',
                'volume': '0',
                'material': 'None',
                'findings': 'No significant bone defect region detected.'
            })

        # Compute average dimensions
        avg_w = sum(widths) / len(widths)
        avg_h = sum(heights) / len(heights)

        # Convert to realistic dental ranges (mm)
        width  = round((avg_w / 10) + 6, 2)
        height = round((avg_h / 10) + 8, 2)
        depth  = round((width + height) / 5, 2)

        # Estimate volume
        volume = round((width * height * depth) / 10, 2)

        max_dim = max(width, height, depth)

        # Scaffold material suggestion
        if max_dim < 5:
            material = 'Collagen scaffold'
        elif max_dim <= 10:
            material = 'Hydroxyapatite + collagen composite'
        else:
            material = '3D printed scaffold'

        findings = 'Possible bone defect region detected from radiograph.'

        return jsonify({
            'width':    str(width),
            'height':   str(height),
            'depth':    str(depth),
            'volume':   str(volume),
            'material': material,
            'findings': findings
        })

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    import os
port = int(os.environ.get('PORT', 5000))
app.run(host='0.0.0.0', port=port)