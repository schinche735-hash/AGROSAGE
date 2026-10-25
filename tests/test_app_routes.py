import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from app import app, TARGET_LABELS, CROPS, label_allowed, disease_guidance, crop_options_for_model, EXPANDED_MODEL_PATH, EXPANDED_LABELS_PATH, ALL_MODEL_PATH, ALL_LABELS_PATH, PV_LABELS_PATH, reference_image_name


class AppRouteTests(unittest.TestCase):
    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_home_languages_render(self):
        for lang in ("en", "hi", "mr"):
            with self.subTest(lang=lang):
                response = self.client.get(f"/?lang={lang}")
                self.assertEqual(response.status_code, 200)
                self.assertIn(b"plant", response.data.lower())
                self.assertNotIn(b"10 diseases across 3 crops", response.data)
                self.assertNotIn("3 फसलों में 10 रोग श्रेणियाँ".encode("utf-8"), response.data)
                self.assertNotIn("3 पिकांमधील 10 रोग श्रेणी".encode("utf-8"), response.data)

    def test_crop_selector_covers_expanded_plantvillage_crops(self):
        self.assertEqual(len(CROPS), 16)
        self.assertTrue({"auto", "Corn_(maize)", "Potato", "Tomato", "Apple", "Grape", "Pepper,_bell"}.issubset(CROPS))
        self.assertFalse({"Blueberry", "Raspberry"}.intersection(CROPS))
        self.assertTrue({"Soybean", "Wheat", "Tur"}.issubset(CROPS))
        html = self.client.get("/").get_data(as_text=True)
        self.assertIn('name="crop"', html)
        self.assertIn("Let the model choose", html)
        self.assertIn("Maize / corn", html)
        self.assertIn("Potato", html)
        self.assertIn("Tomato", html)
        self.assertNotIn("Gemini review only", html)
        self.assertIn("Crop Health Guide", html)
        options = {key for key, _ in crop_options_for_model("en")}
        if not (ALL_MODEL_PATH.exists() and ALL_LABELS_PATH.exists()):
            self.assertEqual(options, {"auto", "Corn_(maize)", "Potato", "Tomato"})

    def test_reference_thumbnails_cover_all_plantvillage_diseases(self):
        import json
        project = Path(__file__).resolve().parents[1]
        labels = json.loads(PV_LABELS_PATH.read_text(encoding="utf-8"))
        diseases = [label for label in labels if not label.endswith("___healthy")]
        self.assertEqual(len(diseases), 26)
        for label in diseases:
            with self.subTest(label=label):
                image_path = project / "static" / reference_image_name(label)
                self.assertTrue(image_path.is_file(), str(image_path))

    def test_cotton_disease_reference_images_exist(self):
        project = Path(__file__).resolve().parents[1]
        for filename in (
            "plantvillage_cotton_alternaria_leaf_spot.jpg",
            "plantvillage_cotton_bacterial_blight.jpg",
            "plantvillage_cotton_fusarium_wilt.jpg",
            "plantvillage_cotton_verticillium_wilt.jpg",
        ):
            with self.subTest(filename=filename):
                self.assertTrue((project / "static" / "disease_images" / filename).is_file())

    def test_focused_targets_and_expanded_plantvillage_labels(self):
        diseases = [label for label in TARGET_LABELS if not label.endswith("___healthy")]
        healthy = [label for label in TARGET_LABELS if label.endswith("___healthy")]
        self.assertEqual(len(diseases), 10)
        self.assertEqual(len(healthy), 3)
        self.assertTrue(label_allowed("Corn_(maize)___Common_rust_"))
        self.assertTrue(label_allowed("Apple___Apple_scab"))
        self.assertTrue(label_allowed("Wheat___Septoria"))
        self.assertTrue(label_allowed("Tur___Sterility_mosaic_disease"))
        self.assertFalse(label_allowed("Unlisted_crop___Unknown_disease"))

    def test_new_soybean_wheat_and_tur_diseases_have_reference_images(self):
        response = self.client.get("/library?lang=en")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        project = Path(__file__).resolve().parents[1]
        expected = (
            "plantvillage_soybean_bacterial_leaf_blight.jpg",
            "plantvillage_soybean_dry_leaf.jpg",
            "plantvillage_soybean_septoria_brown_spot.jpg",
            "plantvillage_soybean_vein_necrosis.jpg",
            "plantvillage_wheat_brown_rust.jpg",
            "plantvillage_wheat_powdery_mildew.jpg",
            "plantvillage_wheat_septoria.jpg",
            "plantvillage_wheat_yellow_rust.jpg",
            "plantvillage_tur_cercospora_leaf_spot.jpg",
            "plantvillage_tur_sterility_mosaic_disease.jpg",
        )
        for filename in expected:
            with self.subTest(filename=filename):
                self.assertIn(filename, html)
                self.assertTrue((project / "static" / "disease_images" / filename).is_file())

    def test_shared_knowledge_library_renders(self):
        # The library must show built-in guidance even when no cache entries exist.
        with patch("app.get_cached_guidance", return_value=None):
            response = self.client.get("/library?lang=en")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("Crop Health Guide", html)
        self.assertIn("40 disease entries", html)
        self.assertIn("Apple scab is a fungal disease caused by Venturia inaequalis", html)
        self.assertIn("Cotton Fusarium wilt is a soil-associated fungal disease", html)
        for filename in (
            "plantvillage_cotton_alternaria_leaf_spot.jpg",
            "plantvillage_cotton_bacterial_blight.jpg",
            "plantvillage_cotton_fusarium_wilt.jpg",
            "plantvillage_cotton_verticillium_wilt.jpg",
        ):
            self.assertIn(filename, html)
            self.assertTrue((Path(__file__).resolve().parents[1] / "static" / "disease_images" / filename).is_file())
        self.assertNotIn("The classifier did not find a familiar disease pattern", html)

    def test_old_history_url_redirects_to_library(self):
        response = self.client.get("/history", follow_redirects=False)
        self.assertEqual(response.status_code, 302)
        self.assertIn("/library", response.headers["Location"])

    def test_cached_guidance_skips_gemini_request(self):
        cached = {"status": "cached", "message": "", "description": "Saved description",
                  "prevention": "Saved prevention", "sources": [], "cached": True}
        with patch("app.get_cached_guidance", return_value=cached), \
             patch("app.research_disease") as research_mock:
            result = disease_guidance("Tomato", "Early blight", "en")
        self.assertEqual(result["description"], "Saved description")
        self.assertTrue(result["cached"])
        research_mock.assert_not_called()

    def test_successful_guidance_is_saved_for_reuse(self):
        fresh = {"status": "ok", "message": "", "description": "Detailed description",
                 "prevention": "Safe prevention", "sources": []}
        with patch("app.get_cached_guidance", return_value=None), \
             patch("app.research_disease", return_value=fresh), \
             patch("app.save_guidance") as save_mock:
            result = disease_guidance("Tomato", "Early blight", "en")
        save_mock.assert_called_once_with("Tomato", "Early blight", "en", fresh)
        self.assertFalse(result["cached"])

    def test_offline_fallback_guidance_is_saved_for_library(self):
        fallback = {"status": "fallback", "message": "", "description": "Offline description",
                    "prevention": "Offline prevention", "sources": []}
        with patch("app.get_cached_guidance", return_value=None):
            with patch("app.research_disease", return_value=fallback):
                with patch("app.save_guidance") as save_mock:
                    result = disease_guidance("Tomato", "Early blight", "en")
        save_mock.assert_called_once_with("Tomato", "Early blight", "en", fallback)
        self.assertFalse(result["cached"])
        self.assertEqual(result["description"], "Offline description")

    def test_knowledge_cache_round_trip_uses_disease_and_language(self):
        import app as app_module
        fresh = {"status": "ok", "message": "", "description": "Detailed description",
                 "prevention": "Safe prevention", "sources": [{"title": "Guide", "url": "https://example.org"}]}
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(app_module, "DB_PATH", Path(tmp) / "knowledge.db"):
                app_module.init_db()
                app_module.save_guidance("Tomato", "Early blight", "en", fresh)
                cached = app_module.get_cached_guidance("Tomato", "Early blight", "en")
                hindi = app_module.get_cached_guidance("Tomato", "Early blight", "hi")
        self.assertEqual(cached["description"], "Detailed description")
        self.assertEqual(cached["sources"][0]["url"], "https://example.org")
        self.assertTrue(cached["cached"])
        self.assertIsNone(hindi)

    def test_health_reports_ai_role(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["app"], "ok")
        self.assertIn(payload["database"], ("sqlite", "mysql"))
        self.assertEqual(payload["target_disease_count"], 40)
        self.assertEqual(payload["healthy_classes"], 13)
        self.assertEqual(payload["ai_role"], "image_verification_and_description_and_prevention")


if __name__ == "__main__":
    unittest.main()
