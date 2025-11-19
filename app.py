from flask import Flask, render_template, request, jsonify, send_from_directory
import os, uuid, io, requests, json, logging
from pathlib import Path
from PIL import Image

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(r"C:\INTERNSHIP_VIHARATECH\INTERN_PR2\Selected_images")

UPLOAD_FOLDER = str((BASE_DIR / "static" / "uploads").resolve())
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------- Basic logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---------- Data loading helpers ----------
def normalize_key(value: str) -> str:
    """Lower-case alphanumeric key used for lookups."""
    if not value:
        return ""
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def load_json_file(candidates, default):
    """Return the first successfully loaded JSON file from candidate paths."""
    for path in candidates:
        if path and Path(path).exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as exc:
                logging.warning("Failed to load %s: %s", path, exc)
    return default


def sanitize_nutrition_entries(raw):
    data = []
    lookup = {}
    for entry in raw or []:
        name = entry.get("food_name")
        info = entry.get("nutritional_info") or {}
        if not name:
            continue
        safe_info = {
            "protein": info.get("protein"),
            "fiber": info.get("fiber"),
            "calories": info.get("calories"),
            "carbohydrates": info.get("carbohydrates"),
            "fat": info.get("fat")
        }
        data.append({"food_name": name, "nutritional_info": safe_info})
        lookup[name.lower()] = safe_info
    return data, lookup


def sanitize_class_metrics(entry):
    return {
        "class_name": entry.get("Class_Name"),
        "support": entry.get("Support"),
        "precision": entry.get("Precision(%)"),
        "recall": entry.get("Recall(%)"),
        "f1_score": entry.get("F1(%)"),
        "tp": entry.get("TP"),
        "fp": entry.get("FP"),
        "fn": entry.get("FN"),
        "tn": entry.get("TN")
    }


NUTRITION_DATA_RAW = load_json_file(
    [BASE_DIR / "food_nutrition.json", DATA_DIR / "food_nutrition.json"],
    []
)
MODEL_PERFORMANCE = load_json_file(
    [BASE_DIR / "model_performance.json", DATA_DIR / "model_performance.json"],
    []
)

NUTRITION_DATA, NUTRITION_LOOKUP = sanitize_nutrition_entries(NUTRITION_DATA_RAW)
CLASS_NAME_MAP = {}

for item in NUTRITION_DATA:
    CLASS_NAME_MAP.setdefault(normalize_key(item["food_name"]), item["food_name"])

MODEL_INDEX = {}
for block in MODEL_PERFORMANCE or []:
    model_name = (block.get("Model_Name") or "").strip()
    if not model_name:
        continue
    lower_model = model_name.lower()
    group_classes = []
    for cls_entry in block.get("Classes", []):
        metrics = sanitize_class_metrics(cls_entry)
        class_name = metrics.get("class_name")
        if not class_name:
            continue
        norm = normalize_key(class_name)
        CLASS_NAME_MAP.setdefault(norm, class_name)
        group_classes.append({"normalized": norm, "metrics": metrics})
        if class_name.lower() not in NUTRITION_LOOKUP:
            NUTRITION_LOOKUP[class_name.lower()] = {}
    group_metrics = [item["metrics"] for item in group_classes]
    for item in group_classes:
        MODEL_INDEX.setdefault(lower_model, {})[item["normalized"]] = {
            "model_name": model_name,
            "model_file": block.get("Model_File"),
            "group": block.get("Group_Name"),
            "test_path": block.get("Test_Path"),
            "test_samples": block.get("Test_Samples"),
            "test_accuracy": block.get("Test_Accuracy"),
            "class_metrics": item["metrics"],
            "group_metrics": group_metrics
        }

# ensure nutrition list contains entries for every known class
known_names = {entry["food_name"] for entry in NUTRITION_DATA}
for canonical_name in CLASS_NAME_MAP.values():
    if canonical_name not in known_names:
        NUTRITION_DATA.append({"food_name": canonical_name, "nutritional_info": NUTRITION_LOOKUP.get(canonical_name.lower(), {})})

ALL_CLASSES = sorted(CLASS_NAME_MAP.values(), key=lambda x: x.lower())


def resolve_class_name(label: str):
    if not label:
        return None
    key = normalize_key(label)
    if key in CLASS_NAME_MAP:
        return CLASS_NAME_MAP[key]
    for stored_key, stored_name in CLASS_NAME_MAP.items():
        if key in stored_key or stored_key in key:
            return stored_name
    return None


def resolve_model_name(label: str):
    key = normalize_key(label)
    if not key:
        return "Custom Model"
    if "resnet" in key:
        return "ResNet-50"
    if "vgg" in key:
        return "VGG-16"
    return "Custom Model"


# store last uploaded filename
last_uploaded_image = None

# ROUTES
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/static/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/classes", methods=["GET"])
def classes():
    # Return the list of class names
    return jsonify(ALL_CLASSES)

@app.route("/nutrition", methods=["GET"])
def nutrition():
    return jsonify(NUTRITION_DATA)

@app.route("/models", methods=["GET"])
def models():
    return jsonify(MODEL_PERFORMANCE)

@app.route("/upload", methods=["POST"])
def upload():
    global last_uploaded_image
    if "file" not in request.files:
        return jsonify({"error":"No file"}), 400
    f = request.files["file"]
    # allow jpg/jpeg only
    filename = f.filename.lower()
    if not (filename.endswith(".jpg") or filename.endswith(".jpeg")):
        # still save but warn (you might choose to reject)
        logging.info("Uploaded file not jpg/jpeg: %s", filename)
    ext = "jpg"
    name = f"{uuid.uuid4().hex}.{ext}"
    path = os.path.join(UPLOAD_FOLDER, name)
    f.save(path)
    last_uploaded_image = name
    return jsonify({"url":f"/static/uploads/{name}"})

@app.route("/load_url", methods=["POST"])
def load_url():
    global last_uploaded_image
    data = request.get_json()
    url = data.get("url")
    if not url:
        return jsonify({"error":"No url"}), 400
    try:
        resp = requests.get(url, timeout=8)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        name = f"{uuid.uuid4().hex}.jpg"
        path = os.path.join(UPLOAD_FOLDER, name)
        img.save(path)
        last_uploaded_image = name
        return jsonify({"url":f"/static/uploads/{name}"})
    except Exception as e:
        logging.exception("load_url failed")
        return jsonify({"error":"Invalid image URL"}), 400

@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}
    requested_model = payload.get("model") or "Custom Model"
    requested_class = payload.get("class")

    canonical_class = resolve_class_name(requested_class)
    if not canonical_class:
        return jsonify({"error": "Unable to match the provided class with dataset entries."}), 400

    model_name = resolve_model_name(requested_model)
    bucket = MODEL_INDEX.get(model_name.lower())
    if not bucket:
        return jsonify({"error": f"No performance data available for model '{model_name}'."}), 404

    class_key = normalize_key(canonical_class)
    record = bucket.get(class_key)
    if not record:
        return jsonify({"error": f"{model_name} does not include metrics for '{canonical_class}'."}), 404

    metrics = record.get("class_metrics", {})
    nutrition = NUTRITION_LOOKUP.get(canonical_class.lower(), {})

    report_payload = {}
    for cls_metrics in record.get("group_metrics", []):
        cname = cls_metrics.get("class_name")
        if not cname:
            continue
        report_payload[cname] = {
            "precision": cls_metrics.get("precision"),
            "recall": cls_metrics.get("recall"),
            "f1-score": cls_metrics.get("f1_score"),
            "support": cls_metrics.get("support"),
            "tp": cls_metrics.get("tp"),
            "fp": cls_metrics.get("fp"),
            "fn": cls_metrics.get("fn"),
            "tn": cls_metrics.get("tn")
        }

    analysis_parts = [
        f"Model {model_name} ({record.get('group')})",
    ]
    if record.get("test_accuracy") is not None:
        analysis_parts.append(
            f"Group accuracy {record['test_accuracy']}% on {record.get('test_samples')} test samples."
        )
    if metrics:
        prec = metrics.get("precision")
        rec = metrics.get("recall")
        f1 = metrics.get("f1_score")
        analysis_parts.append(
            f"{canonical_class}: precision {prec if prec is not None else '-'}%, "
            f"recall {rec if rec is not None else '-'}%, "
            f"F1 {f1 if f1 is not None else '-'}%."
        )
    if nutrition:
        analysis_parts.append(
            f"Nutrition → calories {nutrition.get('calories')}, protein {nutrition.get('protein')}, fat {nutrition.get('fat')}."
        )
    if payload.get("filename"):
        analysis_parts.append(f"Detected from file '{payload['filename']}'.")
    if payload.get("image_url"):
        analysis_parts.append(f"Uploaded path: {payload['image_url']}.")

    analysis = " ".join(part for part in analysis_parts if part)

    response = {
        "model": model_name,
        "model_file": record.get("model_file"),
        "group": record.get("group"),
        "test_samples": record.get("test_samples"),
        "test_path": record.get("test_path"),
        "class": canonical_class,
        "nutrition": nutrition,
        "accuracy": record.get("test_accuracy"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "f1_score": metrics.get("f1_score"),
        "support": metrics.get("support"),
        "tp": metrics.get("tp"),
        "fp": metrics.get("fp"),
        "fn": metrics.get("fn"),
        "tn": metrics.get("tn"),
        "report": report_payload,
        "analysis": analysis or f"{model_name} metrics for {canonical_class}."
    }
    return jsonify(response)


if __name__ == "__main__":
    logging.info("Starting Flask app with %d classes", len(ALL_CLASSES))
    app.run(debug=True)
