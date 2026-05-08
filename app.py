from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# ── Calibration ───────────────────────────────────────────────────────────────
# Panoramic X-ray: real jaw width ≈ 150 mm
# Formula: PIXELS_PER_MM = image_width_px / 150.0  (auto-computed per image)
# Periapical X-ray fallback: 3.78 px/mm (96 DPI)
REAL_JAW_WIDTH_MM   = 150.0   # standard adult jaw width used for panoramic scale
PERIAPICAL_PPM      = 3.78    # fallback for small periapical X-rays
PANORAMIC_THRESHOLD = 1200    # images wider than this are treated as panoramic

# Depth is not visible in 2-D X-ray — estimated as % of width
DEPTH_RATIO = 0.5

# ── Anatomical clamp ranges (mm) ─────────────────────────────────────────────
WIDTH_RANGE  = (2.0, 20.0)
HEIGHT_RANGE = (2.0, 18.0)
DEPTH_RANGE  = (1.5,  8.0)

# ── Contour quality filters ───────────────────────────────────────────────────
MIN_AREA         = 500
MAX_AREA         = 120_000
MAX_ASPECT_RATIO = 5.0
MIN_SOLIDITY     = 0.30


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_pixels_per_mm(img_w_px: int) -> float:
    """
    Auto-detect scale based on image width.
    Panoramic X-rays are wide (>1200px) and represent ~150mm jaw width.
    Periapical X-rays are small and typically scanned at 96 DPI.
    """
    if img_w_px >= PANORAMIC_THRESHOLD:
        return img_w_px / REAL_JAW_WIDTH_MM
    return PERIAPICAL_PPM


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(value, hi))


def recommend_scaffold(volume_mm3: float) -> tuple:
    if volume_mm3 <= 200:
        return ('Small Scaffold',  'Collagen Membrane + Hydroxyapatite (HA)')
    elif volume_mm3 <= 800:
        return ('Medium Scaffold', 'Hydroxyapatite (HA) + β-TCP (Biphasic Calcium Phosphate)')
    else:
        return ('Large Scaffold',  'PCL/PLA + HA Composite (3-D Printed)')


def severity_label(volume_mm3: float) -> str:
    if volume_mm3 <= 0:   return 'None'
    elif volume_mm3 <= 200: return 'Mild'
    elif volume_mm3 <= 800: return 'Moderate'
    else:                   return 'Severe'


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
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
            return jsonify({'error': 'Could not decode image. Ensure it is JPEG or PNG.'}), 400

        img_h_px, img_w_px = img.shape[:2]

        # ── 3. Auto-detect scale ──────────────────────────────────────────────
        pixels_per_mm = get_pixels_per_mm(img_w_px)
        is_panoramic  = img_w_px >= PANORAMIC_THRESHOLD

        # ── 4. Pre-processing ─────────────────────────────────────────────────
        gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        blur     = cv2.GaussianBlur(enhanced, (5, 5), 0)

        # Otsu threshold
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # ── 5. Morphological clean-up ─────────────────────────────────────────
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel, iterations=1)

        # ── 6. Contour detection ──────────────────────────────────────────────
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total_contours = len(contours)

        widths, heights = [], []

        for c in contours:
            area = cv2.contourArea(c)

            # Gate 1 — Area
            if not (MIN_AREA < area < MAX_AREA):
                continue

            x, y, w, h = cv2.boundingRect(c)

            # Gate 2 — Aspect ratio
            aspect = max(w, h) / (min(w, h) + 1e-5)
            if aspect > MAX_ASPECT_RATIO:
                continue

            # Gate 3 — Solidity
            hull      = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            solidity  = area / (hull_area + 1e-5)
            if solidity < MIN_SOLIDITY:
                continue

            # Gate 4 — Position (skip top 10% and bottom 10% — background artefacts)
            cy = y + h / 2.0
            if cy < img_h_px * 0.10 or cy > img_h_px * 0.90:
                continue

            # Gate 5 — For panoramic: skip contours wider than 20% of image
            #           (those are whole-jaw structures, not defects)
            if is_panoramic and w > img_w_px * 0.20:
                continue

            widths.append(float(w))
            heights.append(float(h))

        # ── 7. No defect found ────────────────────────────────────────────────
        if not widths:
            return jsonify({
                'width':         '0',
                'height':        '0',
                'depth':         '0',
                'volume':        '0',
                'scaffoldRange': 'None',
                'material':      'None',
                'severity':      'None',
                'findings':      'No significant bone defect region detected.',
                'debug': {
                    'contoursTotal': total_contours,
                    'contoursKept':  0,
                    'imageSize':     f'{img_w_px} x {img_h_px} px',
                    'pixelsPerMM':   round(pixels_per_mm, 2),
                    'isPanoramic':   is_panoramic,
                }
            })

        # ── 8. Pick best contour — largest qualifying one ─────────────────────
        candidates = sorted(zip(widths, heights), key=lambda p: p[0] * p[1], reverse=True)
        best       = candidates[:1]

        avg_w_px = sum(p[0] for p in best) / len(best)
        avg_h_px = sum(p[1] for p in best) / len(best)

        # ── 9. Pixel → mm ─────────────────────────────────────────────────────
        raw_w = avg_w_px / pixels_per_mm
        raw_h = avg_h_px / pixels_per_mm
        raw_d = raw_w * DEPTH_RATIO

        # ── 10. Clamp ─────────────────────────────────────────────────────────
        width  = round(clamp(raw_w, *WIDTH_RANGE),  2)
        height = round(clamp(raw_h, *HEIGHT_RANGE), 2)
        depth  = round(clamp(raw_d, *DEPTH_RANGE),  2)
        volume = round(width * height * depth, 2)

        # ── 11. Recommendations ───────────────────────────────────────────────
        scaffold_range, material = recommend_scaffold(volume)
        severity = severity_label(volume)

        findings = (
            f'{severity} bone defect detected. '
            f'Dimensions: {width} mm (W) × {height} mm (H) × {depth} mm (D). '
            f'Volume: {volume} mm³.'
        )

        return jsonify({
            'width':         str(width),
            'height':        str(height),
            'depth':         str(depth),
            'volume':        str(volume),
            'scaffoldRange': scaffold_range,
            'material':      material,
            'severity':      severity,
            'findings':      findings,
            'debug': {
                'raw_w_px':      round(avg_w_px, 1),
                'raw_h_px':      round(avg_h_px, 1),
                'raw_w_mm':      round(raw_w, 2),
                'raw_h_mm':      round(raw_h, 2),
                'raw_d_mm':      round(raw_d, 2),
                'contoursTotal': total_contours,
                'contoursKept':  len(widths),
                'imageSize':     f'{img_w_px} x {img_h_px} px',
                'pixelsPerMM':   round(pixels_per_mm, 2),
                'isPanoramic':   is_panoramic,
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
