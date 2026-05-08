from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# ── Calibration ───────────────────────────────────────────────────────────────
# Typical periapical X-ray: ~3–4 px/mm at standard resolution.
# Formula: PIXELS_PER_MM = DPI / 25.4
# e.g. 96 DPI  → 3.78 px/mm
#      150 DPI → 5.91 px/mm
#      300 DPI → 11.8 px/mm
PIXELS_PER_MM = 3.78  # ← adjust to match your scanner/imaging DPI

# ── Anatomically realistic clamp ranges (mm) ──────────────────────────────────
WIDTH_RANGE  = (2.0, 20.0)
HEIGHT_RANGE = (2.0, 18.0)
DEPTH_RANGE  = (2.0, 10.0)

# ── Contour filter thresholds (pixels²) ──────────────────────────────────────
MIN_CONTOUR_AREA = 500
MAX_CONTOUR_AREA = 200_000


def pixel_to_mm(pixels: float) -> float:
    """Convert pixel measurement to millimetres using calibrated scale."""
    return pixels / PIXELS_PER_MM


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(value, hi))


def recommend_scaffold(volume_mm3: float) -> tuple[str, str]:
    """
    Return (scaffoldRange, material) based on defect volume.

    Clinical volume thresholds:
        ≤ 200 mm³  → Small  (single-tooth / minor defect)
        ≤ 800 mm³  → Medium (2-tooth span / moderate defect)
        >  800 mm³ → Large  (multi-tooth / severe defect)
    """
    if volume_mm3 <= 200:
        return (
            'Small Scaffold',
            'Collagen Membrane + Hydroxyapatite (HA)'
        )
    elif volume_mm3 <= 800:
        return (
            'Medium Scaffold',
            'Hydroxyapatite (HA) + β-TCP (Biphasic Calcium Phosphate)'
        )
    else:
        return (
            'Large Scaffold',
            'PCL/PLA + HA Composite (3-D Printed)'
        )


@app.route('/health', methods=['GET'])
def health():
    """Simple health-check endpoint."""
    return jsonify({'status': 'ok'})


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # ── 1. Validate upload ────────────────────────────────────────────────
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400

        # ── 2. Decode image ───────────────────────────────────────────────────
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': 'Could not decode image. Ensure it is a valid JPEG/PNG.'}), 400

        img_h, img_w = img.shape[:2]

        # ── 3. Pre-processing ─────────────────────────────────────────────────
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # CLAHE enhances local contrast — helps with low-contrast X-rays
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Gaussian blur to suppress noise before thresholding
        blur = cv2.GaussianBlur(enhanced, (5, 5), 0)

        # Otsu's method picks the optimal threshold automatically
        _, thresh = cv2.threshold(
            blur, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # ── 4. Morphological clean-up ─────────────────────────────────────────
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel, iterations=1)

        # ── 5. Contour detection ──────────────────────────────────────────────
        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        widths, heights = [], []

        for c in contours:
            area = cv2.contourArea(c)
            if MIN_CONTOUR_AREA < area < MAX_CONTOUR_AREA:
                x, y, w, h = cv2.boundingRect(c)
                widths.append(float(w))
                heights.append(float(h))

        # ── 6. No defect found ────────────────────────────────────────────────
        if not widths:
            return jsonify({
                'width':        '0',
                'height':       '0',
                'depth':        '0',
                'volume':       '0',
                'scaffoldRange':'None',
                'material':     'None',
                'findings':     'No significant bone defect region detected.',
                'imageSize':    f'{img_w} x {img_h} px'
            })

        # ── 7. Pixel → mm conversion ──────────────────────────────────────────
        avg_w_px = sum(widths)  / len(widths)
        avg_h_px = sum(heights) / len(heights)

        raw_w = pixel_to_mm(avg_w_px)
        raw_h = pixel_to_mm(avg_h_px)

        # Depth is not visible in a 2-D X-ray.
        # Estimate: alveolar bone depth ≈ 50 % of the defect width (radiographic norm).
        raw_d = raw_w * 0.5

        # ── 8. Clamp to anatomically plausible range ──────────────────────────
        width  = round(clamp(raw_w, *WIDTH_RANGE),  2)
        height = round(clamp(raw_h, *HEIGHT_RANGE), 2)
        depth  = round(clamp(raw_d, *DEPTH_RANGE),  2)
        volume = round(width * height * depth, 2)

        # ── 9. Scaffold recommendation ────────────────────────────────────────
        scaffold_range, material = recommend_scaffold(volume)

        # ── 10. Severity label ────────────────────────────────────────────────
        if volume <= 200:
            severity = 'Mild'
        elif volume <= 800:
            severity = 'Moderate'
        else:
            severity = 'Severe'

        findings = (
            f'{severity} bone defect region detected from radiograph. '
            f'Estimated dimensions: {width} mm (W) × {height} mm (H) × {depth} mm (D). '
            f'Approximate defect volume: {volume} mm³.'
        )

        return jsonify({
            'width':        str(width),
            'height':       str(height),
            'depth':        str(depth),
            'volume':       str(volume),
            'scaffoldRange': scaffold_range,
            'material':     material,
            'severity':     severity,
            'findings':     findings,
            'contoursFound': len(widths),
            'imageSize':    f'{img_w} x {img_h} px'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
