import io
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
import app as app_module
from app import app, TARGET_LABELS


class DummyModel:
    _crop_preprocess = staticmethod(lambda x: x)
    _ten_class_model = True

    def predict(self, images, verbose=0):
        values = np.full((1, len(TARGET_LABELS)), 0.01, dtype=np.float32)
        values[0, 0] = 0.88
        return values


class PredictionFlowTests(unittest.TestCase):
    def setUp(self):
        app.config["TESTING"] = True
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_root = Path(self.temp_dir.name)
        self.db_path = self.temp_root / "predictions.db"
        self.upload_dir = self.temp_root / "uploads"
        self.upload_dir.mkdir()
        with sqlite3.connect(self.db_path) as con:
            con.execute("CREATE TABLE predictions(id INTEGER PRIMARY KEY AUTOINCREMENT,created_at TEXT NOT NULL,crop TEXT NOT NULL,disease TEXT NOT NULL,confidence REAL NOT NULL,image_name TEXT NOT NULL)")
        self.db_patch = patch.object(app_module, "DB_PATH", self.db_path)
        self.upload_patch = patch.object(app_module, "UPLOAD_DIR", self.upload_dir)
        self.db_patch.start()
        self.upload_patch.start()
        self.client = app.test_client()

    def tearDown(self):
        self.upload_patch.stop()
        self.db_patch.stop()
        self.temp_dir.cleanup()

    def test_each_new_upload_gets_a_unique_result_image_url(self):
        research_result = {
            "status": "ok", "message": "",
            "description": "Test description.", "prevention": "Test prevention.", "sources": []
        }
        result_image_names = []
        with patch("app.validate_plant_photo", return_value=(True, "")), \
             patch("app.get_model", return_value=(DummyModel(), TARGET_LABELS)), \
             patch("app.verify_diagnosis_image", return_value={"status": "verified", "verdict": "match", "message": "review agrees", "reason": "", "observed_crop": "corn", "observed_disease": "common rust"}), \
             patch("app.disease_guidance", return_value=research_result):
            for color, filename in [((30, 150, 40), "first.jpg"), ((180, 40, 30), "second.jpg")]:
                buffer = io.BytesIO()
                Image.new("RGB", (80, 80), color).save(buffer, format="JPEG")
                buffer.seek(0)
                response = self.client.post("/", data={
                    "lang": "en", "crop": "auto", "leaf_image": (buffer, filename)
                }, content_type="multipart/form-data")
                self.assertEqual(response.status_code, 200)
                html = response.get_data(as_text=True)
                import re
                match = re.search(r'/static/uploads/(scan_[a-f0-9]+\.jpg)', html)
                self.assertIsNotNone(match, "result should point to a uniquely named uploaded image")
                result_image_names.append(match.group(1))
        self.assertNotEqual(result_image_names[0], result_image_names[1])
        self.assertTrue((self.upload_dir / result_image_names[0]).exists())
        self.assertTrue((self.upload_dir / result_image_names[1]).exists())

    def test_unrelated_flat_photo_is_rejected_before_model_or_guidance(self):
        from app import validate_plant_photo
        image = Image.new("RGB", (240, 240), (220, 40, 35))
        accepted, reason = validate_plant_photo(image)
        self.assertFalse(accepted)
        self.assertIn(reason, {"detail", "blur", "plant"})

    def test_unrelated_upload_never_calls_disease_model_or_guidance(self):
        buffer = io.BytesIO()
        Image.new("RGB", (240, 240), (220, 40, 35)).save(buffer, format="JPEG")
        buffer.seek(0)
        with patch("app.get_model") as model, patch("app.disease_guidance") as guidance:
            response = self.client.post("/", data={
                "lang": "en", "crop": "auto", "leaf_image": (buffer, "car.jpg")
            }, content_type="multipart/form-data", follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"may not be a clear plant or leaf image", response.data)
        model.assert_not_called()
        guidance.assert_not_called()

    def test_classifier_predicts_and_ai_only_describes(self):
        buffer = io.BytesIO()
        Image.new("RGB", (80, 80), (60, 140, 50)).save(buffer, format="JPEG")
        buffer.seek(0)
        research_result = {
            "status": "ok", "message": "",
            "description": "A fungal disease with orange pustules.",
            "prevention": "Use resistant varieties and monitor plants.",
            "sources": []
        }
        with patch("app.validate_plant_photo", return_value=(True, "")), \
             patch("app.get_model", return_value=(DummyModel(), TARGET_LABELS)), \
             patch("app.verify_diagnosis_image", return_value={"status": "verified", "verdict": "match", "message": "review agrees", "reason": "", "observed_crop": "corn", "observed_disease": "common rust"}), \
             patch("app.disease_guidance", return_value=research_result) as ai:
            response = self.client.post("/", data={
                "lang": "en", "crop": "auto",
                "leaf_image": (buffer, "leaf.jpg")
            }, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("Common rust", html)
        self.assertIn("A fungal disease with orange pustules.", html)
        self.assertIn("Use resistant varieties and monitor plants.", html)
        self.assertNotIn("Gemini vision second opinion", html)
        self.assertNotIn("Other possibilities", html)
        self.assertEqual(ai.call_args.args[:2], ("Maize / corn", "Common rust"))


    def test_conflicting_vision_review_withholds_disease_guidance(self):
        buffer = io.BytesIO()
        Image.new("RGB", (120, 120), (60, 140, 50)).save(buffer, format="JPEG")
        buffer.seek(0)
        disagreement = {
            "status": "disagree", "verdict": "disagree",
            "message": "The AI second opinion did not support the model's prediction.",
            "reason": "The visible symptoms are not distinctive.", "observed_crop": "unknown",
            "observed_disease": "uncertain",
        }
        with (
            patch("app.validate_plant_photo", return_value=(True, "")),
            patch("app.get_model", return_value=(DummyModel(), TARGET_LABELS)),
            patch("app.verify_diagnosis_image", return_value=disagreement),
            patch("app.disease_guidance") as guidance,
        ):
            response = self.client.post("/", data={
                "lang": "en", "crop": "auto",
                "leaf_image": (buffer, "uncertain-leaf.jpg")
            }, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("Prediction failed", html)
        self.assertNotIn("Diagnosis not confirmed", html)
        self.assertNotIn("Common rust", html)
        self.assertNotIn("Disease description and prevention", html)
        self.assertNotIn("98.9", html)
        guidance.assert_not_called()

    def test_uncertain_vision_review_does_not_hide_a_model_prediction(self):
        buffer = io.BytesIO()
        Image.new("RGB", (120, 120), (60, 140, 50)).save(buffer, format="JPEG")
        buffer.seek(0)
        research_result = {
            "status": "ok", "message": "",
            "description": "Test disease description.",
            "prevention": "Test prevention steps.",
            "sources": [],
        }
        uncertain = {
            "status": "uncertain", "verdict": "uncertain",
            "message": "The image is ambiguous.", "reason": "",
            "observed_crop": "unknown", "observed_disease": "uncertain",
        }
        with (
            patch("app.validate_plant_photo", return_value=(True, "")),
            patch("app.get_model", return_value=(DummyModel(), TARGET_LABELS)),
            patch("app.verify_diagnosis_image", return_value=uncertain),
            patch("app.disease_guidance", return_value=research_result) as guidance,
        ):
            response = self.client.post("/", data={
                "lang": "en", "crop": "auto",
                "leaf_image": (buffer, "uncertain-but-valid-leaf.jpg")
            }, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertNotIn("prediction-failed", html)
        self.assertIn("Common rust", html)
        self.assertIn("Test disease description.", html)
        guidance.assert_called_once()


if __name__ == "__main__":
    unittest.main()
