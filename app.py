from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# ── Calibration ───────────────────────────────────────────────────────────────
# Formula: PIXELS_PER_MM = DPI / 25.4
# Common values:
#   72 DPI  → 2.83 px/mm
#   96 DPI  → 3.78 px/mm  (default screen)
#  150 DPI  → 5.91 px/mm
#  300 DPI  → 11.81 px/mm (high-res scan)
PIXELS_PER_MM = 3.78

# Depth is not visible in 2-D X-ray — estimated as % of width
DEPTH_RATIO = 0.5   # alveolar bone depth ≈ 50 % of defect width

# ── Anatomical clamp ranges (mm) ─────────────────────────────────────────────
WIDTH_RANGE  = (2.0, 15.0)
HEIGHT_RANGE = (2.0, 12.0)
DEPTH_RANGE  = (1.5,  6.0)

# ── Contour quality filters ───────────────────────────────────────────────────
MIN_AREA         = 800      # px² — ignore tiny noise specks
MAX_AREA         = 50_000   # px² — ignore huge background blobs / full jaw
MAX_ASPECT_RATIO = 4.0      # ignore razor-thin edges (not bone defects)
MIN_SOLIDITY     = 0.35     # ignore scattered / wispy contours


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def pixel_to_mm(pixels: float) -> float:
    return pixels / PIXELS_PER_MM


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(value, hi))


def recommend_scaffold(volume_mm3: float) -> tuple:
    """
    Clinical volume thresholds (matches patients.tsx exactly):
        ≤ 200 mm³  → Small  scaffold
        ≤ 800 mm³  → Medium scaffold
        >  800 mm³ → Large  scaffold
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


def severity_label(volume_mm3: float) -> str:
    if volume_mm3 <= 0:
        return 'None'
    elif volume_mm3 <= 200:
        return 'Mild'
    elif volume_mm3 <= 800:
        return 'Moderate'
    else:
        return 'Severe'


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

        # ── 3. Pre-processing ─────────────────────────────────────────────────

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # CLAHE — improves contrast on low-contrast dental X-rays
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Gaussian blur — suppresses high-frequency noise before thresholding
        blur = cv2.GaussianBlur(enhanced, (5, 5), 0)

        # Otsu's thresholding — auto-picks optimal threshold per image
        _, thresh = cv2.threshold(
            blur, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # ── 4. Morphological clean-up ─────────────────────────────────────────
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # Close: fills small holes inside detected regions
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        # Open: removes small isolated noise blobs
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel, iterations=1)

        # ── 5. Contour detection ──────────────────────────────────────────────
        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        total_contours = len(contours)
        widths, heights = [], []

        for c in contours:
            area = cv2.contourArea(c)

            # Gate 1 — Area: ignore tiny speckles and huge jaw/background blobs
            if not (MIN_AREA < area < MAX_AREA):
                continue

            x, y, w, h = cv2.boundingRect(c)

            # Gate 2 — Aspect ratio: real defects are roughly boxy
            aspect = max(w, h) / (min(w, h) + 1e-5)
            if aspect > MAX_ASPECT_RATIO:
                continue

            # Gate 3 — Solidity: defects are filled shapes, not wispy fragments
            hull      = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            solidity  = area / (hull_area + 1e-5)
            if solidity < MIN_SOLIDITY:
                continue

            # Gate 4 — Position: skip contours touching very top/bottom edges
            #           (those are usually background artefacts, not defects)
            cy = y + h / 2.0
            if cy < img_h_px * 0.10 or cy > img_h_px * 0.90:
                continue

            widths.append(float(w))
            heights.append(float(h))

        # ── 6. No defect found ────────────────────────────────────────────────
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
                    'contoursTotal':  total_contours,
                    'contoursKept':   0,
                    'imageSize':      f'{img_w_px} x {img_h_px} px',
                    'pixelsPerMM':    PIXELS_PER_MM,
                }
            })

        # ── 7. Select best contour(s) ─────────────────────────────────────────
        # Sort candidates by bounding-box area descending.
        # Use the single largest qualifying contour — it is almost always
        # the defect. Change [:1] to [:3] if multi-defect averaging is needed.
        candidates = sorted(
            zip(widths, heights),
            key=lambda p: p[0] * p[1],
            reverse=True
        )
        best = candidates[:1]

        avg_w_px = sum(p[0] for p in best) / len(best)
        avg_h_px = sum(p[1] for p in best) / len(best)

        # ── 8. Pixel → mm conversion ──────────────────────────────────────────
        raw_w = pixel_to_mm(avg_w_px)
        raw_h = pixel_to_mm(avg_h_px)
        raw_d = raw_w * DEPTH_RATIO   # depth estimated — not visible in 2-D

        # ── 9. Clamp to anatomically plausible range ──────────────────────────
        width  = round(clamp(raw_w, *WIDTH_RANGE),  2)
        height = round(clamp(raw_h, *HEIGHT_RANGE), 2)
        depth  = round(clamp(raw_d, *DEPTH_RANGE),  2)
        volume = round(width * height * depth, 2)

        # ── 10. Recommendations ───────────────────────────────────────────────
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
            # ── debug block (remove in production) ───────────────────────────
            'debug': {
                'raw_w_px':      round(avg_w_px, 1),
                'raw_h_px':      round(avg_h_px, 1),
                'raw_w_mm':      round(raw_w, 2),
                'raw_h_mm':      round(raw_h, 2),
                'raw_d_mm':      round(raw_d, 2),
                'contoursTotal': total_contours,
                'contoursKept':  len(widths),
                'imageSize':     f'{img_w_px} x {img_h_px} px',
                'pixelsPerMM':   PIXELS_PER_MM,
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
