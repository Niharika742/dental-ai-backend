from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import os
import joblib

app = Flask(__name__)
CORS(app)

# ── X-ray type detection ──────────────────────────────────────────────────────
PANORAMIC_THRESHOLD = 1200
REAL_JAW_WIDTH_MM   = 150.0
PERIAPICAL_PPM      = 3.78
DEPTH_RATIO         = 0.5

# ── Anatomical clamp ranges (mm) ──────────────────────────────────────────────
WIDTH_RANGE  = (2.0, 20.0)
HEIGHT_RANGE = (2.0, 18.0)
DEPTH_RANGE  = (1.5,  8.0)

# ── Contour gates ─────────────────────────────────────────────────────────────
MIN_AREA         = 400
MAX_AREA         = 500000
MAX_ASPECT_RATIO = 6.0
MIN_SOLIDITY     = 0.25

# ── AI Model config ───────────────────────────────────────────────────────────
TOOTH_TYPES = {
    '1':'Third Molar','2':'Second Molar','3':'First Molar',
    '4':'Second Premolar','5':'First Premolar','6':'Canine',
    '7':'Lateral Incisor','8':'Central Incisor','9':'Central Incisor',
    '10':'Lateral Incisor','11':'Canine','12':'First Premolar',
    '13':'Second Premolar','14':'First Molar','15':'Second Molar',
    '16':'Third Molar','17':'Third Molar','18':'Second Molar',
    '19':'First Molar','20':'First Premolar','21':'First Premolar',
    '22':'Canine','23':'Lateral Incisor','24':'Central Incisor',
    '25':'Central Incisor','26':'Lateral Incisor','27':'Canine',
    '28':'First Premolar','29':'Second Premolar','30':'First Molar',
    '31':'Second Molar','32':'Third Molar'
}

# ── Load AI models at startup ─────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_model(name):
    path = os.path.join(BASE_DIR, name)
    if os.path.exists(path):
        return joblib.load(path)
    return None

ai_models = {
    'width':    load_model('width_mm_model.pkl'),
    'height':   load_model('height_mm_model.pkl'),
    'depth':    load_model('depth_mm_model.pkl'),
    'volume':   load_model('volume_mm3_model.pkl'),
    'scaffold': load_model('scaffold_mm3_model.pkl'),
    'material': load_model('scaffold_material_model.pkl'),
    'tooth':    load_model('tooth_classifier_model.pkl'),
}
le_tooth    = load_model('label_encoder_tooth.pkl')
le_material = load_model('label_encoder_material.pkl')

ai_ready = all(v is not None for v in ai_models.values()) and le_tooth and le_material
print(f"AI models loaded: {ai_ready}")


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_ppmm(img_w_px):
    if img_w_px >= PANORAMIC_THRESHOLD:
        return img_w_px / REAL_JAW_WIDTH_MM
    return PERIAPICAL_PPM

def clamp(v, lo, hi):
    return max(lo, min(v, hi))

def recommend_scaffold(vol):
    if vol <= 200:
        return 'Small Scaffold', 'Collagen Membrane + Hydroxyapatite (HA)'
    elif vol <= 800:
        return 'Medium Scaffold', 'Hydroxyapatite (HA) + Beta-TCP (Biphasic Calcium Phosphate)'
    else:
        return 'Large Scaffold', 'PCL/PLA + HA Composite (3-D Printed)'

def severity_label(vol):
    if vol <= 0:   return 'None'
    if vol <= 200: return 'Mild'
    if vol <= 800: return 'Moderate'
    return 'Severe'

def extract_features_for_prediction(img, x, y, w_px, h_px):
    x, y = max(0, x), max(0, y)
    roi = img[y:y+h_px, x:x+w_px]
    if roi.size == 0 or w_px < 3 or h_px < 3:
        return None

    tooth_resized = cv2.resize(roi, (32, 64))
    features = []

    img_h, img_w = img.shape[:2]
    features.append(w_px / img_w)
    features.append(h_px / img_h)
    features.append(w_px / img_w * h_px / img_h)
    features.append(float(w_px) / float(h_px) if h_px > 0 else 0)

    cx = (x + w_px / 2) / img_w
    cy = (y + h_px / 2) / img_h
    features.append(cx)
    features.append(cy)

    pixels = tooth_resized[tooth_resized > 0]
    if len(pixels) == 0:
        return None
    features.append(float(np.mean(pixels)))
    features.append(float(np.std(pixels)))
    features.append(float(np.median(pixels)))
    features.append(float(np.percentile(pixels, 25)))
    features.append(float(np.percentile(pixels, 75)))

    hist = cv2.calcHist([tooth_resized], [0], None, [16], [1, 256])
    hist = hist.flatten() / (hist.sum() + 1e-7)
    features.extend(hist.tolist())

    edges = cv2.Canny(tooth_resized, 30, 100)
    features.append(float(np.sum(edges > 0)) / (32 * 64))
    features.append(0.75)

    return np.array(features).reshape(1, -1)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'ai_ready': ai_ready})


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400

        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'error': 'Cannot decode image'}), 400

        img_h_px, img_w_px = img.shape[:2]
        is_panoramic = img_w_px >= PANORAMIC_THRESHOLD
        ppmm         = get_ppmm(img_w_px)

        gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        blur     = cv2.GaussianBlur(enhanced, (5, 5), 0)

        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel, iterations=1)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total = len(contours)

        candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            if not (MIN_AREA < area < MAX_AREA):
                continue
            x, y, w, h = cv2.boundingRect(c)
            aspect = max(w, h) / (min(w, h) + 1e-5)
            if aspect > MAX_ASPECT_RATIO:
                continue
            hull_area = cv2.contourArea(cv2.convexHull(c))
            solidity  = area / (hull_area + 1e-5)
            if solidity < MIN_SOLIDITY:
                continue
            cy2 = y + h / 2.0
            if cy2 < img_h_px * 0.08 or cy2 > img_h_px * 0.92:
                continue
            if is_panoramic and w > img_w_px * 0.25:
                continue
            candidates.append((float(w), float(h), area))

        if not candidates:
            return jsonify({
                'width': '0', 'height': '0', 'depth': '0', 'volume': '0',
                'scaffoldRange': 'None', 'material': 'None', 'severity': 'None',
                'findings': 'No significant bone defect region detected.',
                'debug': {
                    'contoursTotal': total, 'contoursKept': 0,
                    'imageSize': f'{img_w_px} x {img_h_px} px',
                    'pixelsPerMM': round(ppmm, 2), 'isPanoramic': is_panoramic
                }
            })

        candidates.sort(key=lambda c: c[0] * c[1], reverse=True)
        best_w_px, best_h_px, best_area = candidates[0]

        raw_w = best_w_px / ppmm
        raw_h = best_h_px / ppmm
        raw_d = raw_w * DEPTH_RATIO

        width  = round(clamp(raw_w, *WIDTH_RANGE),  2)
        height = round(clamp(raw_h, *HEIGHT_RANGE), 2)
        depth  = round(clamp(raw_d, *DEPTH_RANGE),  2)
        volume = round(width * height * depth, 2)

        scaffold_range, material = recommend_scaffold(volume)
        severity = severity_label(volume)
        findings = (
            f'{severity} bone defect detected. '
            f'Dimensions: {width} mm (W) x {height} mm (H) x {depth} mm (D). '
            f'Volume: {volume} mm3.'
        )

        return jsonify({
            'width': str(width), 'height': str(height),
            'depth': str(depth), 'volume': str(volume),
            'scaffoldRange': scaffold_range, 'material': material,
            'severity': severity, 'findings': findings,
            'debug': {
                'best_w_px': round(best_w_px, 1), 'best_h_px': round(best_h_px, 1),
                'best_area_px2': round(best_area, 0),
                'raw_w_mm': round(raw_w, 2), 'raw_h_mm': round(raw_h, 2),
                'raw_d_mm': round(raw_d, 2),
                'contoursTotal': total, 'contoursKept': len(candidates),
                'imageSize': f'{img_w_px} x {img_h_px} px',
                'pixelsPerMM': round(ppmm, 2), 'isPanoramic': is_panoramic,
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/ai-analyze', methods=['POST'])
def ai_analyze():
    try:
        if not ai_ready:
            return jsonify({'error': 'AI models not loaded on server'}), 503

        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400

        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'error': 'Cannot decode image'}), 400

        img_h_px, img_w_px = img.shape[:2]
        is_panoramic = img_w_px >= PANORAMIC_THRESHOLD

        gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        blur     = cv2.GaussianBlur(enhanced, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel, iterations=1)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        teeth_results = []

        for c in contours:
            area = cv2.contourArea(c)
            if not (MIN_AREA < area < MAX_AREA):
                continue
            x, y, w, h = cv2.boundingRect(c)
            aspect = max(w, h) / (min(w, h) + 1e-5)
            if aspect > MAX_ASPECT_RATIO:
                continue
            hull_area = cv2.contourArea(cv2.convexHull(c))
            solidity  = area / (hull_area + 1e-5)
            if solidity < MIN_SOLIDITY:
                continue
            cy2 = y + h / 2.0
            if cy2 < img_h_px * 0.08 or cy2 > img_h_px * 0.92:
                continue
            if is_panoramic and w > img_w_px * 0.25:
                continue

            features = extract_features_for_prediction(enhanced, x, y, w, h)
            if features is None:
                continue

            tooth_num = le_tooth.inverse_transform(ai_models['tooth'].predict(features))[0]
            mat_label = le_material.inverse_transform(ai_models['material'].predict(features))[0]
            width_mm   = round(float(ai_models['width'].predict(features)[0]), 2)
            height_mm  = round(float(ai_models['height'].predict(features)[0]), 2)
            depth_mm   = round(float(ai_models['depth'].predict(features)[0]), 2)
            volume_mm3 = round(float(ai_models['volume'].predict(features)[0]), 2)
            scaffold   = round(float(ai_models['scaffold'].predict(features)[0]), 2)

            teeth_results.append({
                'tooth_number':      str(tooth_num),
                'tooth_type':        TOOTH_TYPES.get(str(tooth_num), 'Unknown'),
                'width_mm':          width_mm,
                'height_mm':         height_mm,
                'depth_mm':          depth_mm,
                'volume_mm3':        volume_mm3,
                'scaffold_mm3':      scaffold,
                'scaffold_material': mat_label,
                'position':          {'x': x, 'y': y, 'w': w, 'h': h}
            })

        teeth_results.sort(key=lambda t: int(t['tooth_number']) if t['tooth_number'].isdigit() else 99)

        if not teeth_results:
            return jsonify({'error': 'No teeth detected in image'}), 400

        total_volume   = round(sum(t['volume_mm3'] for t in teeth_results), 2)
        total_scaffold = round(sum(t['scaffold_mm3'] for t in teeth_results), 2)

        return jsonify({
            'teeth':              teeth_results,
            'teeth_count':        len(teeth_results),
            'total_volume_mm3':   total_volume,
            'total_scaffold_mm3': total_scaffold,
            'image_type':         'panoramic' if is_panoramic else 'periapical',
            'ai_powered':         True
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)