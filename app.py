"""Smart Crop Disease Detection with local image classification and cached AI guidance."""
import json, os, re, sqlite3, threading, urllib.request, uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote, urlparse
from flask import Flask, render_template, request, redirect, url_for, flash
from PIL import Image, UnidentifiedImageError
from research import research_disease, offline_guidance
from vision_verification import verify_diagnosis_image

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "models"
EXPANDED_MODEL_PATH = MODEL_DIR / "plantdisease_expanded_frozen_old_candidate.keras"
EXPANDED_LABELS_PATH = MODEL_DIR / "labels_plantdisease_expanded_frozen_old_candidate.json"
ALL_MODEL_PATH = MODEL_DIR / "plantdisease_all38_mnv2.keras"
ALL_LABELS_PATH = MODEL_DIR / "labels_plantdisease_all38.json"
TEN_MODEL_PATH = MODEL_DIR / "plantdisease_10_mnv2.keras"
TEN_LABELS_PATH = MODEL_DIR / "labels_plantdisease_10.json"
PV_MODEL_PATH = MODEL_DIR / "best_mnv2_pv_original.keras"
PV_LABELS_PATH = MODEL_DIR / "labels_pv_original.json"
UPLOAD_DIR = BASE / "static" / "uploads"
DB_PATH = BASE / "data" / "predictions.db"
DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
MODEL_URL = "https://huggingface.co/ashikaasriarun/MobileNet-V2-PlantVillage/resolve/main/best_mnv2_pv_original.keras"
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "smart-crop-disease-local-dev-key")
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
for path in (UPLOAD_DIR, MODEL_DIR, DB_PATH.parent): path.mkdir(parents=True, exist_ok=True)
_model = None
_labels = None
_lock = threading.Lock()

# Ten diseases plus a healthy class for each of the three supported crops.
TARGET_LABELS = [
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___healthy",
]
CROP_NAMES = {
    "Corn_(maize)": {"en": "Maize / corn", "hi": "मक्का", "mr": "मका"},
    "Potato": {"en": "Potato", "hi": "आलू", "mr": "बटाटा"},
    "Tomato": {"en": "Tomato", "hi": "टमाटर", "mr": "टोमॅटो"},
    "Apple": {"en": "Apple", "hi": "सेब", "mr": "सफरचंद"},
    "Blueberry": {"en": "Blueberry", "hi": "ब्लूबेरी", "mr": "ब्लूबेरी"},
    "Cherry_(including_sour)": {"en": "Cherry", "hi": "चेरी", "mr": "चेरी"},
    "Grape": {"en": "Grape", "hi": "अंगूर", "mr": "द्राक्षे"},
    "Orange": {"en": "Orange", "hi": "संतरा", "mr": "संत्रे"},
    "Peach": {"en": "Peach", "hi": "आड़ू", "mr": "पीच"},
    "Pepper,_bell": {"en": "Bell pepper", "hi": "शिमला मिर्च", "mr": "ढोबळी मिरची"},
    "Raspberry": {"en": "Raspberry", "hi": "रास्पबेरी", "mr": "रास्पबेरी"},
    "Soybean": {"en": "Soybean", "hi": "सोयाबीन", "mr": "सोयाबीन"},
    "Wheat": {"en": "Wheat", "hi": "गेहूँ", "mr": "गहू"},
    "Tur": {"en": "Tur / pigeonpea", "hi": "अरहर / तूर", "mr": "तूर / तूरडाळ"},
    "Cotton": {"en": "Cotton", "hi": "कपास", "mr": "कापूस"},
    "Squash": {"en": "Squash", "hi": "स्क्वैश", "mr": "स्क्वॅश"},
    "Strawberry": {"en": "Strawberry", "hi": "स्ट्रॉबेरी", "mr": "स्ट्रॉबेरी"},
}
# Soybean, wheat, and tur are available only when the expanded trained model is installed.
# crop_options_for_model filters these choices against the active model labels.
CROPS = {
    "auto": {"en": "Let the model choose", "hi": "मॉडल को चुनने दें", "mr": "मॉडेलला निवडू द्या"},
    **{key: value for key, value in CROP_NAMES.items() if key not in {"Blueberry", "Raspberry"}},
}
LANG = {"en": "English", "hi": "Hindi", "mr": "Marathi"}


class MySQLAdapter:
    """Small DB-API adapter so the app can keep the same parameterised query code."""
    def __init__(self, connection):
        self.connection = connection

    @property
    def row_factory(self):
        return None

    @row_factory.setter
    def row_factory(self, _value):
        # MySQL uses DictCursor, giving the same row['column'] access as sqlite3.Row.
        pass

    def execute(self, sql, params=()):
        cursor = self.connection.cursor()
        cursor.execute(sql.replace("?", "%s"), params)
        return cursor


@contextmanager
def db_connection():
    if DATABASE_URL.startswith(("mysql://", "mysql+pymysql://")):
        import pymysql
        parsed = urlparse(DATABASE_URL.replace("mysql+pymysql://", "mysql://", 1))
        connection = pymysql.connect(
            host=parsed.hostname or "localhost",
            port=parsed.port or 3306,
            user=unquote(parsed.username or ""),
            password=unquote(parsed.password or ""),
            database=(parsed.path or "/smartcrop").lstrip("/"),
            charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor,
            connect_timeout=5,
        )
        adapter = MySQLAdapter(connection)
        try:
            yield adapter
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
    else:
        connection = sqlite3.connect(DB_PATH)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()


def database_backend():
    return "mysql" if DATABASE_URL.startswith(("mysql://", "mysql+pymysql://")) else "sqlite"


def init_db():
    if database_backend() == "mysql":
        statement = """CREATE TABLE IF NOT EXISTS disease_knowledge(
            id BIGINT AUTO_INCREMENT PRIMARY KEY,
            crop VARCHAR(100) NOT NULL,
            disease VARCHAR(160) NOT NULL,
            lang VARCHAR(5) NOT NULL,
            description TEXT NOT NULL,
            prevention TEXT NOT NULL,
            sources_json TEXT NOT NULL,
            updated_at VARCHAR(32) NOT NULL,
            UNIQUE KEY unique_disease_language(crop, disease, lang)
        )"""
    else:
        statement = """CREATE TABLE IF NOT EXISTS disease_knowledge(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            crop TEXT NOT NULL,
            disease TEXT NOT NULL,
            lang TEXT NOT NULL,
            description TEXT NOT NULL,
            prevention TEXT NOT NULL,
            sources_json TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(crop, disease, lang)
        )"""
    with db_connection() as con:
        con.execute(statement)


KNOWLEDGE_CACHE_DAYS = 90


def get_cached_guidance(crop, disease, lang):
    with db_connection() as con:
        row = con.execute(
            "SELECT description, prevention, sources_json, updated_at FROM disease_knowledge WHERE crop=? AND disease=? AND lang=?",
            (crop, disease, lang),
        ).fetchone()
    if not row:
        return None
    # Ignore the old generic "no familiar pattern" text that was incorrectly cached
    # for real disease classes. It is not a disease description.
    description_text = (row["description"] or "").strip().lower()
    obsolete_fallbacks = (
        "the classifier did not find a familiar disease pattern",
        "मॉडल को इस तस्वीर में कोई परिचित रोग पैटर्न नहीं मिला",
        "या प्रतिमेत मॉडेलला ओळखीचा रोग-पॅटर्न आढळला नाही",
    )
    if any(description_text.startswith(text) for text in obsolete_fallbacks):
        return None
    try:
        updated = datetime.fromisoformat(row["updated_at"])
        if (datetime.now() - updated).days >= KNOWLEDGE_CACHE_DAYS:
            return None
        sources = json.loads(row["sources_json"] or "[]")
    except (ValueError, TypeError, json.JSONDecodeError):
        return None
    return {
        "status": "cached", "message": "", "description": row["description"],
        "prevention": row["prevention"], "sources": sources, "cached": True,
        "updated_at": row["updated_at"],
    }


def save_guidance(crop, disease, lang, guidance):
    # Cache reusable disease knowledge, including offline fallback guidance.
    if guidance.get("status") not in ("ok", "fallback") or not guidance.get("description") or not guidance.get("prevention"):
        return
    now = datetime.now().isoformat(timespec="seconds")
    params = (crop, disease, lang, guidance["description"], guidance["prevention"],
              json.dumps(guidance.get("sources", []), ensure_ascii=False), now)
    with db_connection() as con:
        if database_backend() == "mysql":
            con.execute(
                """INSERT INTO disease_knowledge(crop,disease,lang,description,prevention,sources_json,updated_at)
                   VALUES(?,?,?,?,?,?,?) ON DUPLICATE KEY UPDATE description=VALUES(description),
                   prevention=VALUES(prevention), sources_json=VALUES(sources_json), updated_at=VALUES(updated_at)""",
                params,
            )
        else:
            con.execute(
                """INSERT INTO disease_knowledge(crop,disease,lang,description,prevention,sources_json,updated_at)
                   VALUES(?,?,?,?,?,?,?) ON CONFLICT(crop,disease,lang) DO UPDATE SET
                   description=excluded.description, prevention=excluded.prevention,
                   sources_json=excluded.sources_json, updated_at=excluded.updated_at""",
                params,
            )


def disease_guidance(crop, disease, lang, fast=False):
    try:
        cached = get_cached_guidance(crop, disease, lang)
    except Exception:
        cached = None
    if cached:
        return cached
    if fast:
        # Avoid a network round-trip during a scan; use the local disease knowledge.
        fresh = offline_guidance(crop, disease, lang)
    else:
        fresh = research_disease(crop, disease, lang)
    if fresh.get("status") in ("ok", "fallback") and fresh.get("description") and fresh.get("prevention"):
        save_guidance(crop, disease, lang, fresh)
        fresh["cached"] = False
        fresh["updated_at"] = datetime.now().isoformat(timespec="seconds")
    return fresh

def cleanup_old_uploads(max_age_hours=24):
    """Keep recent result images available, but don't retain a permanent image history."""
    cutoff = datetime.now().timestamp() - max_age_hours * 3600
    for path in UPLOAD_DIR.glob("*.jpg"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError:
            app.logger.debug("Could not clean old upload %s", path.name)


def validate_plant_photo(image):
    """Conservative pre-screen for unusable or obviously unrelated uploads.

    This is a lightweight quality/vegetation heuristic, not a trained plant detector;
    ambiguous photos are not guaranteed to be caught.
    Returns (accepted, reason).
    """
    import numpy as np

    if image.width < 96 or image.height < 96:
        return False, "small"
    sample = image.convert("RGB").resize((128, 128))
    rgb = np.asarray(sample, dtype=np.uint8)
    gray = np.asarray(sample.convert("L"), dtype=np.float32)
    if float(gray.std()) < 9.0:
        return False, "detail"
    # Reject nearly textureless images and heavily blurred frames.
    edge_energy = float((np.abs(np.diff(gray, axis=0)).mean() +
                         np.abs(np.diff(gray, axis=1)).mean()) / 2.0)
    if edge_energy < 2.0:
        return False, "blur"

    hsv = np.asarray(sample.convert("HSV"), dtype=np.uint8)
    hue, saturation, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    # PIL hue is 0..255. This broad range covers many green, yellow and brown
    # plant tissues while avoiding a green-only rule that rejects diseased leaves.
    plant_tone = ((hue >= 4) & (hue <= 116) &
                  (saturation >= 35) & (value >= 25) & (value <= 250))
    if float(plant_tone.mean()) < 0.055:
        return False, "plant"
    return True, ""


def get_model():
    """Prefer the expanded model, then the focused model, then the original pretrained fallback."""
    global _model, _labels
    if _model is not None: return _model, _labels
    with _lock:
        if _model is not None: return _model, _labels
        import tensorflow as tf
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        if EXPANDED_MODEL_PATH.exists() and EXPANDED_LABELS_PATH.exists():
            model_path, labels_path, model_kind = EXPANDED_MODEL_PATH, EXPANDED_LABELS_PATH, "expanded55"
        elif ALL_MODEL_PATH.exists() and ALL_LABELS_PATH.exists():
            model_path, labels_path, model_kind = ALL_MODEL_PATH, ALL_LABELS_PATH, "all38"
        elif TEN_MODEL_PATH.exists() and TEN_LABELS_PATH.exists():
            model_path, labels_path, model_kind = TEN_MODEL_PATH, TEN_LABELS_PATH, "focused"
        else:
            model_path, labels_path, model_kind = PV_MODEL_PATH, PV_LABELS_PATH, "legacy"
        if not model_path.exists():
            tmp = model_path.with_suffix(".download")
            urllib.request.urlretrieve(MODEL_URL, tmp)
            tmp.replace(model_path)
        _labels = json.loads(labels_path.read_text(encoding="utf-8"))
        _model = tf.keras.models.load_model(str(model_path), compile=False)
        _model._crop_preprocess = (lambda x: x) if model_kind in ("expanded55", "all43", "all38", "focused") else preprocess_input
        _model._all_class_model = model_kind in ("expanded55", "all43", "all38")
        _model._ten_class_model = model_kind == "focused"
        _model._model_path = str(model_path)
        # Build the TensorFlow execution graph before the first visitor uploads a photo.
        import numpy as np
        _model(np.zeros((1, 160, 160, 3), dtype=np.float32), training=False)
    return _model, _labels


def pretty(label):
    raw_crop, _, disease = label.partition("___")
    crop = CROP_NAMES.get(raw_crop, {}).get("en", raw_crop.replace("_", " ").replace(",", ""))
    disease = disease.replace("_", " ").strip()
    if disease.lower() == "healthy": disease = "Healthy"
    return crop, disease


def label_allowed(label):
    if label in TARGET_LABELS:
        return True
    # Keep label validation aligned with whichever expanded classifier is active.
    for labels_path in (EXPANDED_LABELS_PATH, ALL_LABELS_PATH, TEN_LABELS_PATH, PV_LABELS_PATH):
        try:
            if labels_path.exists() and label in json.loads(labels_path.read_text(encoding="utf-8")):
                return True
        except (OSError, json.JSONDecodeError):
            continue
    return False


def reference_image_name(label):
    slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    return f"disease_images/plantvillage_{slug}.jpg"


def crop_options_for_model(lang):
    if EXPANDED_MODEL_PATH.exists() and EXPANDED_LABELS_PATH.exists():
        labels_path = EXPANDED_LABELS_PATH
    elif ALL_MODEL_PATH.exists() and ALL_LABELS_PATH.exists():
        labels_path = ALL_LABELS_PATH
    elif TEN_MODEL_PATH.exists() and TEN_LABELS_PATH.exists():
        labels_path = TEN_LABELS_PATH
    else:
        labels_path = None
    try:
        labels = json.loads(labels_path.read_text(encoding="utf-8")) if labels_path else TARGET_LABELS
    except (OSError, json.JSONDecodeError):
        labels = TARGET_LABELS
    supported = {label.partition("___")[0] for label in labels if "___" in label and not label.endswith("___healthy")}
    return [(key, value[lang]) for key, value in CROPS.items() if key == "auto" or key in supported]


@app.route("/", methods=["GET", "POST"])
def index():
    lang = request.args.get("lang", request.form.get("lang", "en"))
    if lang not in LANG: lang = "en"
    crop_choice = request.args.get("crop", request.form.get("crop", "auto"))
    crop_options = crop_options_for_model(lang)
    if crop_choice not in {key for key, _ in crop_options}: crop_choice = "auto"
    result = None
    if request.method == "POST":
        file = request.files.get("leaf_image")
        if not file or not file.filename:
            flash({"en":"Please choose a leaf image first.","hi":"कृपया पहले फसल की तस्वीर चुनें।","mr":"कृपया आधी पिकाचा फोटो निवडा."}[lang], "error")
            return redirect(url_for("index", lang=lang))
        cleanup_old_uploads()
        try:
            image = Image.open(file.stream).convert("RGB")
            accepted, reason = validate_plant_photo(image)
            if not accepted:
                message = {
                    "en": "This photo may not be a clear plant or leaf image. Please upload a sharp, well-lit photo showing the crop or leaf clearly.",
                    "hi": "यह तस्वीर स्पष्ट पौधे या पत्ती की तस्वीर नहीं लगती। कृपया फसल या पत्ती की साफ़ और अच्छी रोशनी वाली तस्वीर अपलोड करें।",
                    "mr": "हा फोटो स्पष्ट पिकाचा किंवा पानाचा फोटो वाटत नाही. कृपया पिकाचा किंवा पानाचा स्वच्छ आणि चांगल्या प्रकाशातील फोटो अपलोड करा.",
                }[lang]
                flash(message, "error")
                return redirect(url_for("index", lang=lang))
            image.thumbnail((1600, 1600))
            # Unique URLs prevent browsers from reusing the previous upload from cache.
            saved = f"scan_{uuid.uuid4().hex}.jpg"
            image.save(UPLOAD_DIR / saved, "JPEG", quality=88)
        except (UnidentifiedImageError, OSError, ValueError):
            flash({"en":"Unsupported image. Please upload a JPG, PNG, or WEBP leaf photo.","hi":"यह फ़ाइल समर्थित नहीं है। कृपया JPG, PNG या WEBP फसल की तस्वीर अपलोड करें।","mr":"ही फाइल समर्थित नाही. कृपया JPG, PNG किंवा WEBP पिकाचा फोटो अपलोड करा."}[lang], "error")
            return redirect(url_for("index", lang=lang))
        try:
            import numpy as np
            model, all_labels = get_model()
            arr = np.asarray(image.resize((160, 160)), dtype=np.float32)[None, ...]
            probs = model(model._crop_preprocess(arr), training=False).numpy()[0]
            allowed_indices = (
                [i for i, label in enumerate(all_labels) if label.partition("___")[0] in CROPS]
                if getattr(model, "_all_class_model", False)
                else [i for i, label in enumerate(all_labels) if label in TARGET_LABELS]
            )
            if not allowed_indices:
                raise RuntimeError("The loaded model does not contain any supported disease labels.")
            if crop_choice != "auto":
                crop_indices = [i for i in allowed_indices if all_labels[i].partition("___")[0] == crop_choice]
                if not crop_indices:
                    raise RuntimeError("The active model does not support the selected crop yet. Choose automatic crop detection or install the expanded model.")
                allowed_indices = crop_indices
            ranked = sorted(allowed_indices, key=lambda i: float(probs[i]), reverse=True)
            best = ranked[0]
            crop, disease = pretty(all_labels[best])
            # Keep the model's original softmax score. Renormalizing after crop/class
            # filtering can inflate a weak prediction into an apparently confident one.
            # This score is diagnostic, not a calibrated probability.
            score = float(probs[best])
            if score < 0.30:
                flash({
                    "en": "The model could not identify a supported crop disease confidently. Please upload a closer, clearer plant photo.",
                    "hi": "मॉडल किसी समर्थित फसल रोग की भरोसेमंद पहचान नहीं कर सका। कृपया पौधे की और स्पष्ट, पास से ली गई तस्वीर अपलोड करें।",
                    "mr": "मॉडेलला समर्थित पिकाच्या रोगाची खात्रीशीर ओळख पटली नाही. कृपया पिकाचा अधिक स्पष्ट आणि जवळून घेतलेला फोटो अपलोड करा.",
                }[lang], "error")
                return redirect(url_for("index", lang=lang))
            # Independently inspect the actual image. Softmax confidence alone is not
            # enough: confident false positives are common on out-of-domain field photos.
            verification = verify_diagnosis_image(UPLOAD_DIR / saved, crop, disease, lang)
            # Only an explicit contradiction or non-plant verdict should suppress the result.
            # Gemini can be uncertain or temporarily unavailable even when the classifier is right.
            confirmed = verification.get("verdict") not in {"disagree", "not_plant"}
            research = disease_guidance(crop, disease, lang, fast=True) if confirmed else {
                "status": "unverified", "message": "", "description": "", "prevention": "", "sources": []
            }
            # A real diagnosis must have matching disease-specific guidance. Never
            # display the old generic healthy/no-pattern paragraph for a disease class.
            if confirmed and (not research.get("description", "").strip() or not research.get("prevention", "").strip()):
                confirmed = False
                research = {"status": "unverified", "message": "", "description": "", "prevention": "", "sources": []}
            result = {
                "crop": crop,
                "disease": disease if confirmed else "Needs review — not confirmed",
                "predicted_disease": disease,
                "confidence": score,
                "image": saved, "research": research, "verification": verification,
                "unconfirmed": not confirmed,
                "low_confidence": score < 0.45,
                "model_name": "Crop image classifier",
                "healthy": disease.lower() == "healthy" and confirmed,
            }
        except Exception as exc:
            app.logger.exception("Prediction failed")
            flash({"en":"The disease model could not run. Please retry. Details: ","hi":"रोग पहचान मॉडल नहीं चल सका। कृपया फिर से प्रयास करें। विवरण: ","mr":"रोग ओळख मॉडेल चालू शकले नाही. कृपया पुन्हा प्रयत्न करा. तपशील: "}[lang] + str(exc)[:180], "error")
    return render_template("index.html", result=result, lang=lang, crop_choice=crop_choice, crop_options=crop_options, model_trained=(EXPANDED_MODEL_PATH.exists() and EXPANDED_LABELS_PATH.exists()) or (ALL_MODEL_PATH.exists() and ALL_LABELS_PATH.exists()) or (TEN_MODEL_PATH.exists() and TEN_LABELS_PATH.exists()))


@app.route("/library")
def library():
    lang = request.args.get("lang", "en")
    if lang not in LANG:
        lang = "en"

    # Keep this catalogue aligned with the disease labels allowed by prediction.
    # Healthy leaf labels are intentionally not presented as diseases.
    image_files = {
        "Corn_(maize)___Common_rust_": "corn_common_rust.jpg",
        "Corn_(maize)___Northern_Leaf_Blight": "corn_northern_leaf_blight.jpg",
        "Potato___Early_blight": "potato_early_blight.jpg",
        "Potato___Late_blight": "potato_late_blight.jpg",
        "Tomato___Bacterial_spot": "tomato_bacterial_spot.jpg",
        "Tomato___Early_blight": "tomato_early_blight.jpg",
        "Tomato___Late_blight": "tomato_late_blight.jpg",
        "Tomato___Leaf_Mold": "tomato_leaf_mold.jpg",
        "Tomato___Septoria_leaf_spot": "tomato_septoria_leaf_spot.jpg",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "tomato_yellow_leaf_curl_virus.jpg",
        "Cotton___Alternaria_leaf_spot": "plantvillage_cotton_alternaria_leaf_spot.jpg",
        "Cotton___Bacterial_blight": "plantvillage_cotton_bacterial_blight.jpg",
        "Cotton___Fusarium_wilt": "plantvillage_cotton_fusarium_wilt.jpg",
        "Cotton___Verticillium_wilt": "plantvillage_cotton_verticillium_wilt.jpg",
    }
    localized_names = {
        "Common rust": {"en": "Common rust", "hi": "कॉमन रस्ट", "mr": "कॉमन रस्ट"},
        "Northern Leaf Blight": {"en": "Northern Leaf Blight", "hi": "नॉर्दर्न लीफ ब्लाइट", "mr": "नॉर्दर्न लीफ ब्लाइट"},
        "Early blight": {"en": "Early blight", "hi": "अर्ली ब्लाइट", "mr": "अर्ली ब्लाइट"},
        "Late blight": {"en": "Late blight", "hi": "लेट ब्लाइट", "mr": "लेट ब्लाइट"},
        "Bacterial spot": {"en": "Bacterial spot", "hi": "बैक्टीरियल स्पॉट", "mr": "बॅक्टेरियल स्पॉट"},
        "Leaf Mold": {"en": "Leaf Mold", "hi": "लीफ मोल्ड", "mr": "लीफ मोल्ड"},
        "Septoria leaf spot": {"en": "Septoria leaf spot", "hi": "सेप्टोरिया लीफ स्पॉट", "mr": "सेप्टोरिया लीफ स्पॉट"},
        "Tomato Yellow Leaf Curl Virus": {"en": "Tomato Yellow Leaf Curl Virus", "hi": "टमाटर येलो लीफ कर्ल वायरस", "mr": "टोमॅटो येलो लीफ कर्ल व्हायरस"},
    }
    crop_labels = {
        "Maize / corn": {"en": "Maize / corn", "hi": "मक्का", "mr": "मका"},
        "Potato": {"en": "Potato", "hi": "आलू", "mr": "बटाटा"},
        "Tomato": {"en": "Tomato", "hi": "टमाटर", "mr": "टोमॅटो"},
    }
    active_labels = TARGET_LABELS
    if EXPANDED_MODEL_PATH.exists() and EXPANDED_LABELS_PATH.exists():
        try:
            active_labels = json.loads(EXPANDED_LABELS_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            active_labels = TARGET_LABELS
    elif ALL_MODEL_PATH.exists() and ALL_LABELS_PATH.exists():
        try:
            active_labels = json.loads(ALL_LABELS_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            active_labels = TARGET_LABELS
    catalog = []
    for label in active_labels:
        if label.endswith("___healthy"):
            continue
        image = "disease_images/" + image_files[label] if label in image_files else reference_image_name(label)
        if not (BASE / "static" / image).is_file():
            image = None
        crop, disease = pretty(label)
        cached = get_cached_guidance(crop, disease, lang)
        if not cached or not cached.get("description", "").strip() or not cached.get("prevention", "").strip():
            # Populate the library from verified local guidance even before a user
            # has requested AI research for this class. This performs no network call.
            fallback = offline_guidance(crop, disease, lang)
            if fallback.get("description", "").strip() and fallback.get("prevention", "").strip():
                cached = {**fallback, "cached": False}
        catalog.append({
            "crop": crop,
            "crop_label": crop_labels.get(crop, CROP_NAMES.get(label.partition("___")[0], {"en": crop, "hi": crop, "mr": crop})),
            "name": disease,
            "name_label": localized_names.get(disease, {"en": disease, "hi": disease, "mr": disease}),
            "image": image,
            "guidance": cached,
        })
    return render_template("library.html", catalog=catalog, lang=lang)

@app.route("/history")
def old_history_link():
    return redirect(url_for("library", lang=request.args.get("lang", "en")))


@app.route("/health")
def health():
    all43 = EXPANDED_MODEL_PATH.exists() and EXPANDED_LABELS_PATH.exists()
    all38 = ALL_MODEL_PATH.exists() and ALL_LABELS_PATH.exists()
    focused = TEN_MODEL_PATH.exists() and TEN_LABELS_PATH.exists()
    labels_path = EXPANDED_LABELS_PATH if all43 else ALL_LABELS_PATH if all38 else TEN_LABELS_PATH if focused else None
    try:
        active_labels = json.loads(labels_path.read_text(encoding="utf-8")) if labels_path and labels_path.exists() else TARGET_LABELS
    except (OSError, json.JSONDecodeError):
        active_labels = TARGET_LABELS
    if all43 or all38:
        active_labels = [label for label in active_labels if label.partition("___")[0] in CROPS]
    disease_count = sum(1 for label in active_labels if not label.endswith("___healthy"))
    healthy_count = sum(1 for label in active_labels if label.endswith("___healthy"))
    preference = "expanded55_incremental_mobilenetv2" if all43 else "all38_mobilenetv2" if all38 else "ten_disease_mobilenetv2" if focused else "existing_plantvillage_filtered"
    return {"app": "ok", "database": database_backend(), "model_file_present": (EXPANDED_MODEL_PATH.exists() and EXPANDED_LABELS_PATH.exists()) or (ALL_MODEL_PATH.exists() and ALL_LABELS_PATH.exists()) or (TEN_MODEL_PATH.exists() and TEN_LABELS_PATH.exists()) or PV_MODEL_PATH.exists(), "model_preference": preference, "target_disease_count": disease_count, "healthy_classes": healthy_count, "ai_role": "image_verification_and_description_and_prevention", "shared_disease_knowledge_cache": True, "knowledge_cache_refresh_days": KNOWLEDGE_CACHE_DAYS}


@app.errorhandler(413)
def too_large(_):
    lang = request.form.get("lang", request.args.get("lang", "en"))
    if lang not in LANG: lang = "en"
    flash({"en":"Please upload an image smaller than 8 MB.","hi":"कृपया 8 MB से छोटी तस्वीर अपलोड करें।","mr":"कृपया 8 MB पेक्षा लहान फोटो अपलोड करा."}[lang], "error")
    return redirect(url_for("index", lang=lang))


init_db()

def _preload_model_in_background():
    """Load and warm the classifier during service startup, not during a user's scan."""
    try:
        get_model()
        app.logger.info("Disease model loaded and warmed in background.")
    except Exception:
        app.logger.exception("Background model warm-up failed; it will retry on the first scan.")

# Render health checks can start immediately while TensorFlow initializes in the background.
threading.Thread(target=_preload_model_in_background, name="model-warmup", daemon=True).start()

if __name__ == "__main__": app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=False)
