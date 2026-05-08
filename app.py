from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# ── X-ray type detection ──────────────────────────────────────────────────────
PANORAMIC_THRESHOLD = 1200      # px — images wider than this = panoramic
REAL_JAW_WIDTH_MM   = 150.0     # standard adult jaw width in mm
PERIAPICAL_PPM      = 3.78      # px/mm for small periapical films (96 DPI)
DEPTH_RATIO         = 0.5       # depth = width × 0.5 (not visible in 2-D)

# ── Anatomical clamp ranges (mm) ─────────────────────────────────────────────
WIDTH_RANGE  = (2.0, 20.0)
HEIGHT_RANGE = (2.0, 18.0)
DEPTH_RANGE  = (1.5,  8.0)

# ── Contour gates — RELAXED so large contours are not accidentally dropped ────
MIN_AREA         = 400          # px²  — drop only tiny specks
MAX_AREA         = 500_000      # px²  — drop only the full-image blob
MAX_ASPECT_RATIO = 6.0          # allow slightly elongated shapes
MIN_SOLIDITY     = 0.25         # relaxed — bone defects can be irregular


# ─────────────────────────────────────────────────────────────────────────────
def get_ppmm(img_w_px: int) -> float:
    if img_w_px >= PANORAMIC_THRESHOLD:
        return img_w_px / REAL_JAW_WIDTH_MM   # e.g. 1850 / 150 = 12.33
    return PERIAPICAL_PPM


def clamp(v, lo, hi):
    return max(lo, min(v, hi))


def recommend_scaffold(vol: float):
    if vol <= 200:
        return 'Small Scaffold',  'Collagen Membrane + Hydroxyapatite (HA)'
    elif vol <= 800:
        return 'Medium Scaffold', 'Hydroxyapatite (HA) + β-TCP (Biphasic Calcium Phosphate)'
    else:
        return 'Large Scaffold',  'PCL/PLA + HA Composite (3-D Printed)'


def severity_label(vol: float) -> str:
    if vol <= 0:    return 'None'
    if vol <= 200:  return 'Mild'
    if vol <= 800:  return 'Moderate'
    return 'Severe'


# ─────────────────────────────────────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # 1. Validate
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400

        # 2. Decode
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'error': 'Cannot decode image'}), 400

        img_h_px, img_w_px = img.shape[:2]
        is_panoramic  = img_w_px >= PANORAMIC_THRESHOLD
        ppmm          = get_ppmm(img_w_px)

        # 3. Pre-process
        gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        blur     = cv2.GaussianBlur(enhanced, (5, 5), 0)

        _, thresh = cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # 4. Morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel, iterations=1)

        # 5. Find contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        total = len(contours)

        # 6. Filter contours
        candidates = []
        for c in contours:
            area = cv2.contourArea(c)

            # Gate 1 — area bounds
            if not (MIN_AREA < area < MAX_AREA):
                continue

            x, y, w, h = cv2.boundingRect(c)

            # Gate 2 — aspect ratio
            aspect = max(w, h) / (min(w, h) + 1e-5)
            if aspect > MAX_ASPECT_RATIO:
                continue

            # Gate 3 — solidity
            hull_area = cv2.contourArea(cv2.convexHull(c))
            solidity  = area / (hull_area + 1e-5)
            if solidity < MIN_SOLIDITY:
                continue

            # Gate 4 — vertical position (skip extreme top/bottom 8%)
            cy = y + h / 2.0
            if cy < img_h_px * 0.08 or cy > img_h_px * 0.92:
                continue

            # Gate 5 — for panoramic: skip blobs wider than 25% of image
            #           (whole jaw / skull structures)
            if is_panoramic and w > img_w_px * 0.25:
                continue

            candidates.append((float(w), float(h), area))

        # 7. No contours kept
        if not candidates:
            return jsonify({
                'width': '0', 'height': '0', 'depth': '0', 'volume': '0',
                'scaffoldRange': 'None', 'material': 'None',
                'severity': 'None',
                'findings': 'No significant bone defect region detected.',
                'debug': {
                    'contoursTotal': total, 'contoursKept': 0,
                    'imageSize': f'{img_w_px} x {img_h_px} px',
                    'pixelsPerMM': round(ppmm, 2),
                    'isPanoramic': is_panoramic,
                }
            })

        # 8. Pick largest qualifying contour by bounding-box area
        candidates.sort(key=lambda c: c[0] * c[1], reverse=True)
        best_w_px, best_h_px, best_area = candidates[0]

        # 9. Convert px → mm
        raw_w = best_w_px / ppmm
        raw_h = best_h_px / ppmm
        raw_d = raw_w * DEPTH_RATIO

        # 10. Clamp to anatomy
        width  = round(clamp(raw_w, *WIDTH_RANGE),  2)
        height = round(clamp(raw_h, *HEIGHT_RANGE), 2)
        depth  = round(clamp(raw_d, *DEPTH_RANGE),  2)
        volume = round(width * height * depth, 2)

        # 11. Recommend
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
                'best_w_px':     round(best_w_px, 1),
                'best_h_px':     round(best_h_px, 1),
                'best_area_px2': round(best_area, 0),
                'raw_w_mm':      round(raw_w, 2),
                'raw_h_mm':      round(raw_h, 2),
                'raw_d_mm':      round(raw_d, 2),
                'contoursTotal': total,
                'contoursKept':  len(candidates),
                'imageSize':     f'{img_w_px} x {img_h_px} px',
                'pixelsPerMM':   round(ppmm, 2),
                'isPanoramic':   is_panoramic,
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
