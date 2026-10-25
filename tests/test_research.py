import os
import unittest
from unittest.mock import patch
import research


class ResearchTests(unittest.TestCase):
    def test_json_cleaner_accepts_fenced_json(self):
        result = research._clean_json('```json\n{"description":"A disease","prevention":"Rotate crops"}\n```')
        self.assertEqual(result["description"], "A disease")

    def test_json_cleaner_rejects_non_json(self):
        with self.assertRaises(ValueError):
            research._clean_json("not a report")

    def test_missing_key_returns_only_guidance_status(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
            result = research.research_disease("Maize / corn", "Common rust", "mr")
        self.assertEqual(result["status"], "fallback")
        self.assertIn("मका", result["description"])
        self.assertTrue(result["prevention"])

    def test_ai_output_is_description_and_prevention_only(self):
        response = {"candidates": [{
            "content": {"parts": [{"text": '{"description":"Fungal disease description","prevention":"Use resistant varieties and monitor plants."}'}]},
            "groundingMetadata": {"groundingChunks": [{"web": {"uri": "https://example.org/agriculture", "title": "Agriculture guide"}}]}
        }]}
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key", "GEMINI_MODEL": "test-model"}), \
             patch.object(research, "_post_json", return_value=response) as mocked, \
             patch.object(research, "_cache_get", return_value=None), \
             patch.object(research, "_cache_put"):
            result = research.research_disease("Maize / corn", "Common rust", "en")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["description"], "Fungal disease description")
        self.assertEqual(result["prevention"], "Use resistant varieties and monitor plants.")
        self.assertEqual(set(result) , {"status", "message", "description", "prevention", "sources"})
        prompt = mocked.call_args.args[1]["contents"][0]["parts"][0]["text"]
        self.assertIn("Do NOT re-analyse the image", prompt)
        self.assertIn("exactly two string fields: description and prevention", prompt)

    def test_research_failure_is_graceful(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}), \
             patch.object(research, "_post_json", side_effect=TimeoutError()):
            result = research.research_disease("Tomato", "Early blight", "en")
        self.assertEqual(result["status"], "fallback")
        self.assertIn("Early blight", result["description"])
        self.assertTrue(result["prevention"])

    def test_expanded_model_diseases_have_specific_offline_guidance(self):
        cases = [
            ("Apple", "Apple scab", "Apple scab"),
            ("Grape", "Esca (Black Measles)", "Esca"),
            ("Orange", "Haunglongbing (Citrus greening)", "Citrus greening"),
            ("Cotton", "Fusarium wilt", "Fusarium wilt"),
            ("Tomato", "Tomato mosaic virus", "Tomato mosaic virus"),
            ("Tomato", "Spider mites Two-spotted spider mite", "spider mites"),
        ]
        for crop, disease, expected in cases:
            with self.subTest(crop=crop, disease=disease):
                result = research._offline_guidance(crop, disease, "en", "")
                self.assertEqual(result["status"], "fallback")
                self.assertIn(expected.lower(), result["description"].lower())
                self.assertTrue(result["prevention"])
                self.assertNotIn("classifier did not find a familiar disease pattern", result["description"].lower())

    def test_new_soybean_wheat_and_tur_classes_have_specific_guidance(self):
        cases = [
            ("Soybean", "Bacterial leaf blight", "soybean bacterial leaf blight"),
            ("Soybean", "Dry leaf", "broad visual condition"),
            ("Soybean", "Septoria brown spot", "soybean septoria brown spot"),
            ("Soybean", "Vein necrosis", "not by itself a definitive pathogen diagnosis"),
            ("Wheat", "Brown rust", "wheat brown rust"),
            ("Wheat", "Yellow rust", "wheat yellow rust"),
            ("Wheat", "Septoria", "wheat septoria leaf blotch"),
            ("Wheat", "Powdery mildew", "wheat powdery mildew"),
            ("Tur", "Cercospora leaf spot", "tur (pigeonpea) cercospora leaf spot"),
            ("Tur", "Sterility mosaic disease", "pigeonpea sterility mosaic disease"),
        ]
        for crop, disease, expected in cases:
            with self.subTest(crop=crop, disease=disease):
                result = research._offline_guidance(crop, disease, "en", "")
                self.assertEqual(result["status"], "fallback")
                self.assertIn(expected.lower(), result["description"].lower())
                self.assertTrue(result["prevention"])

    def test_unknown_disease_never_falls_back_to_healthy_description(self):
        result = research._offline_guidance("Unknown crop", "Unknown disease", "en", "")
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["description"], "")


if __name__ == "__main__":
    unittest.main()
