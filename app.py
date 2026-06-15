from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import os
import urllib.request

app = Flask(__name__)
CORS(app)

# ── Configuration ─────────────────────────────────────────────
IMG_SIZE = 224

MATERIAL_CLASSES = [
    'Collagen Membrane',
    'Hydroxyapatite (HA)',
    'Hydroxyapatite (HA) + Beta-TCP',
    'PCL/PLA + HA Composite'
]

WIDTH_MIN,  WIDTH_MAX  = 2.0,  20.0
HEIGHT_MIN, HEIGHT_MAX = 2.0,  18.0
DEPTH_MIN,  DEPTH_MAX  = 1.5,   8.0
VOLUME_MIN, VOLUME_MAX = 6.0, 2880.0

# ── Download Model from Google Drive ──────────────────────────
TFLITE_PATH = 'dental_scaffold_model_v1.tflite'
DRIVE_FILE_ID = '1fLIUamewwS4w8SHllD6GOde1jZAFKDrJ'

def download_model():
    if not os.path.exists(TFLITE_PATH):
        print('📥 Downloading model from Google Drive...')
        try:
            url = f'https://drive.google.com/uc?export=download&id={DRIVE_FILE_ID}&confirm=t'
            urllib.request.urlretrieve(url, TFLITE_PATH)
            size = os.path.getsize(TFLITE_PATH) / (1024*1024)
            print(f'✅ Model downloaded! Size: {size:.1f} MB')
        except Exception as e:
            print(f'⚠️  Download failed: {e}')
    else:
        print('✅ Model already exists locally')

download_model()

# ── Load TFLite Model ─────────────────────────────────────────
interpreter = None
MODEL_TYPE = 'opencv'

def load_tflite():
    global interpreter, MODEL_TYPE
    if os.path.exists(TFLITE_PATH):
        try:
            import tflite_runtime.interpreter as tflite
interpreter = tflite.Interpreter(model_path=TFLITE_PATH)
            interpreter.allocate_tensors()
            inp = interpreter.get_input_details()
            out = interpreter.get_output_details()
            print(f'✅ TFLite model loaded!')
            print(f'   Input : {inp[0]["shape"]}')
            print(f'   Outputs: {[o["name"] for o in out]}')
            MODEL_TYPE = 'tflite'
        except Exception as e:
            print(f'⚠️  TFLite load failed: {e}')
            MODEL_TYPE = 'opencv'
    else:
        print('⚠️  No model file found — using OpenCV fallback')

load_tflite()

# ── Helper Functions ──────────────────────────────────────────
def clamp(v, lo, hi):
    return max(lo, min(float(v), hi))

def denorm(val, lo, hi):
    return round(float(val) * (hi - lo) + lo, 2)

def get_severity(volume):
    if volume <= 0:    return 'None'
    if volume <= 200:  return 'Mild'
    if volume <= 800:  return 'Moderate'
    return 'Severe'

def get_material_from_volume(volume):
    if volume <= 200:  return MATERIAL_CLASSES[0]
    if volume <= 500:  return MATERIAL_CLASSES[1]
    if volume <= 800:  return MATERIAL_CLASSES[2]
    return MATERIAL_CLASSES[3]

def get_scaffold_range(volume):
    if volume <= 200:  return 'Small Scaffold'
    if volume <= 800:  return 'Medium Scaffold'
    return 'Large Scaffold'

# ── Preprocess Image ──────────────────────────────────────────
def preprocess(img_bytes):
    nparr = np.frombuffer(img_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    img_rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_res  = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    img_norm = img_res.astype(np.float32) / 255.0
    return np.expand_dims(img_norm, axis=0)

# ── TFLite Prediction ─────────────────────────────────────────
def predict_tflite(img_array):
    try:
        inp_det = interpreter.get_input_details()
        out_det = interpreter.get_output_details()

        interpreter.set_tensor(inp_det[0]['index'], img_array)
        interpreter.invoke()

        # Get all outputs
        outputs = {}
        for o in out_det:
            outputs[o['name']] = interpreter.get_tensor(o['index'])

        print(f'TFLite outputs: {list(outputs.keys())}')

        # Find measurement head (shape: N,4) and material head (shape: N,4)
        meas_tensor = None
        mat_tensor  = None

        for name, tensor in outputs.items():
            shape = tensor.shape
            print(f'  {name}: shape={shape}')
            if len(shape) >= 2:
                if shape[-1] == 4 and 'meas' in name.lower():
                    meas_tensor = tensor[0]
                elif shape[-1] == len(MATERIAL_CLASSES) and ('mat' in name.lower() or 'softmax' in name.lower()):
                    mat_tensor = tensor[0]

        # Fallback by shape if name matching fails
        if meas_tensor is None or mat_tensor is None:
            tensors_by_shape = {}
            for name, tensor in outputs.items():
                if len(tensor.shape) >= 2:
                    size = tensor.shape[-1]
                    tensors_by_shape[size] = tensor[0]

            if 4 in tensors_by_shape:
                # Two tensors of size 4 — one is measurements, one is material
                # measurements uses sigmoid (values 0-1), material uses softmax (sums to 1)
                vals = [v for k, v in outputs.items() if v.shape[-1] == 4]
                if len(vals) >= 2:
                    # sigmoid output — values spread out
                    # softmax output — sums to ~1
                    if abs(sum(vals[0][0]) - 1.0) < 0.1:
                        mat_tensor  = vals[0][0]
                        meas_tensor = vals[1][0]
                    else:
                        meas_tensor = vals[0][0]
                        mat_tensor  = vals[1][0]
                elif len(vals) == 1:
                    meas_tensor = vals[0][0]

        if meas_tensor is None:
            print('⚠️  Could not find measurement tensor — using OpenCV')
            return None

        # Denormalize to real mm values
        w = clamp(denorm(meas_tensor[0], WIDTH_MIN,  WIDTH_MAX),  WIDTH_MIN,  WIDTH_MAX)
        h = clamp(denorm(meas_tensor[1], HEIGHT_MIN, HEIGHT_MAX), HEIGHT_MIN, HEIGHT_MAX)
        d = clamp(denorm(meas_tensor[2], DEPTH_MIN,  DEPTH_MAX),  DEPTH_MIN,  DEPTH_MAX)
        v = clamp(denorm(meas_tensor[3], VOLUME_MIN, VOLUME_MAX), VOLUME_MIN, VOLUME_MAX)

        # Material
        if mat_tensor is not None:
            mat_idx  = int(np.argmax(mat_tensor))
            material = MATERIAL_CLASSES[mat_idx]
            mat_conf = float(np.max(mat_tensor)) * 100
        else:
            material = get_material_from_volume(v)
            mat_conf = 75.0

        return {
            'width_mm':  round(w, 2),
            'height_mm': round(h, 2),
            'depth_mm':  round(d, 2),
            'volume_mm': round(v, 2),
            'material':  material,
            'mat_conf':  round(mat_conf, 1)
        }

    except Exception as e:
        print(f'⚠️  TFLite prediction error: {e}')
        return None

# ── OpenCV Fallback ───────────────────────────────────────────
def predict_opencv(img_bytes):
    nparr = np.frombuffer(img_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None

    img_h, img_w = img.shape[:2]
    is_panoramic = img_w >= 1200
    ppmm = img_w / 150.0 if is_panoramic else 3.78

    gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    blur     = cv2.GaussianBlur(enhanced, (5,5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    thresh   = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh   = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates  = []

    for c in contours:
        area = cv2.contourArea(c)
        if not (400 < area < 500_000):
            continue
        x, y, w, h = cv2.boundingRect(c)
        aspect = max(w,h) / (min(w,h) + 1e-5)
        if aspect > 6.0:
            continue
        hull_a   = cv2.contourArea(cv2.convexHull(c))
        solidity = area / (hull_a + 1e-5)
        if solidity < 0.25:
            continue
        cy = y + h / 2.0
        if cy < img_h * 0.08 or cy > img_h * 0.92:
            continue
        if is_panoramic and w > img_w * 0.25:
            continue
        candidates.append((float(w), float(h), area))

    if not candidates:
        return {
            'width_mm':  2.0,
            'height_mm': 2.0,
            'depth_mm':  1.5,
            'volume_mm': 6.0,
            'material':  MATERIAL_CLASSES[0],
            'mat_conf':  70.0
        }

    candidates.sort(key=lambda c: c[0]*c[1], reverse=True)
    bw, bh, _ = candidates[0]

    raw_w = bw / ppmm
    raw_h = bh / ppmm
    raw_d = raw_w * 0.5

    w = round(clamp(raw_w, WIDTH_MIN,  WIDTH_MAX),  2)
    h = round(clamp(raw_h, HEIGHT_MIN, HEIGHT_MAX), 2)
    d = round(clamp(raw_d, DEPTH_MIN,  DEPTH_MAX),  2)
    v = round(w * h * d, 2)

    return {
        'width_mm':  w,
        'height_mm': h,
        'depth_mm':  d,
        'volume_mm': v,
        'material':  get_material_from_volume(v),
        'mat_conf':  72.0
    }

# ── Routes ────────────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status':       'ok',
        'model_type':   MODEL_TYPE,
        'model_loaded': interpreter is not None,
        'tflite_file':  os.path.exists(TFLITE_PATH)
    })

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400

        img_bytes = file.read()
        result    = None

        # Try TFLite model first
        if MODEL_TYPE == 'tflite' and interpreter is not None:
            img_array = preprocess(img_bytes)
            if img_array is not None:
                result = predict_tflite(img_array)
                if result:
                    print(f'✅ TFLite prediction: {result}')

        # Fallback to OpenCV
        if result is None:
            print('⚠️  Using OpenCV fallback')
            result = predict_opencv(img_bytes)

        if result is None:
            return jsonify({'error': 'Could not analyze image'}), 500

        w    = result['width_mm']
        h    = result['height_mm']
        d    = result['depth_mm']
        v    = result['volume_mm']
        mat  = result['material']
        conf = result['mat_conf']
        sev  = get_severity(v)
        sr   = get_scaffold_range(v)

        findings = (
            f'{sev} bone defect detected. '
            f'Dimensions: {w}mm (W) × {h}mm (H) × {d}mm (D). '
            f'Volume: {v}mm³. '
            f'Recommended scaffold: {mat}.'
        )

        return jsonify({
            'width':         str(w),
            'height':        str(h),
            'depth':         str(d),
            'volume':        str(v),
            'scaffoldRange': sr,
            'material':      mat,
            'severity':      sev,
            'findings':      findings,
            'confidence':    f'{conf:.1f}%',
            'model_used':    MODEL_TYPE,
            'debug': {
                'model_type':   MODEL_TYPE,
                'mat_conf_pct': conf,
                'w_mm': w, 'h_mm': h,
                'd_mm': d, 'v_mm3': v
            }
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
