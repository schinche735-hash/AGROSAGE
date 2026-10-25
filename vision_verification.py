"""Independent Gemini Vision second opinion for crop-disease predictions.

The vision response is a safety check, not ground truth. Fail closed: unavailable,
ambiguous, or conflicting reviews never count as verified.
"""
import base64
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

LANG = {"en": "English", "hi": "Hindi", "mr": "Marathi"}


def verify_diagnosis_image(image_path, predicted_crop, predicted_disease, lang="en"):
    """Ask Gemini Vision to inspect the actual uploaded image independently."""
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        return {
            "status": "unavailable",
            "message": {
                "en": "This result was not independently checked because AI image review is not configured.",
                "hi": "AI छवि समीक्षा सेट नहीं है, इसलिए इस परिणाम की स्वतंत्र जाँच नहीं हुई।",
                "mr": "AI प्रतिमा पुनरावलोकन सेट केलेले नाही, त्यामुळे या निकालाची स्वतंत्र पडताळणी झाली नाही.",
            }.get(lang, "This result was not independently checked because AI image review is not configured."),
            "observed_crop": "", "observed_disease": "", "reason": "",
        }
    model = os.environ.get("GEMINI_VISION_MODEL", os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")).strip()
    if not model:
        model = "gemini-3.1-flash-lite"
    try:
        path = Path(image_path)
        raw = path.read_bytes()
        suffix = path.suffix.lower()
        mime = "image/png" if suffix == ".png" else "image/webp" if suffix == ".webp" else "image/jpeg"
        prompt = f"""You are an independent plant-health image reviewer. Inspect the attached image itself; do not assume the classifier is correct.
The separate image classifier predicted crop "{predicted_crop}" and disease/status "{predicted_disease}".
Independently inspect the visible image and compare it with the predicted crop and diagnosis. Image symptoms can be ambiguous and many diseases look alike. Do not guess or infer a disease from the supplied prediction. Use "match" when the prediction is reasonably consistent with visible signs; do not require textbook-perfect symptoms. Use "uncertain" whenever the photo is ambiguous, symptoms are not distinctive, the view is limited, or you cannot confidently distinguish similar diseases. Use "disagree" ONLY when there is clear visual evidence that the predicted crop/disease is wrong or a different diagnosis is visibly much more likely; lack of proof by itself is not disagreement. Similar rust pustules can occur on multiple crops, so do not identify wheat instead of maize from rust pustules alone; use uncertain unless crop-specific leaf structure or other clear evidence distinguishes them. If the image is clearly not a plant/crop, say not_plant.
Return JSON only with exactly these fields:
verdict: one of "match", "disagree", "uncertain", "not_plant"
observed_crop: short crop name or "unknown"
observed_disease: short visible diagnosis/status or "uncertain"
reason: one short sentence describing visible evidence and limitations
A "match" means the prediction is reasonably supported by the visible signs. If signs are not decisive, use "uncertain" rather than "disagree". Respond in {LANG.get(lang, "English")}."""
        payload = {
            "contents": [{"role": "user", "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": mime, "data": base64.b64encode(raw).decode("ascii")}},
            ]}],
            "generationConfig": {"temperature": 0.1, "responseMimeType": "application/json"},
        }
        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        request = urllib.request.Request(
            endpoint, data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "x-goog-api-key": key}, method="POST"
        )
        with urllib.request.urlopen(request, timeout=2) as response:
            body = json.loads(response.read().decode("utf-8"))
        candidate = (body.get("candidates") or [{}])[0]
        parts = (candidate.get("content") or {}).get("parts") or []
        text = "\n".join(p.get("text", "") for p in parts if p.get("text"))
        text = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text, flags=re.I)
        report = json.loads(text)
        verdict = str(report.get("verdict", "uncertain")).strip().lower()
        if verdict not in {"match", "disagree", "uncertain", "not_plant"}:
            verdict = "uncertain"
        reason = str(report.get("reason", "")).strip()[:500]
        messages = {
            "match": {
                "en": "The AI second opinion found the prediction broadly consistent with visible symptoms, but this is not a confirmed diagnosis.",
                "hi": "AI की दूसरी राय में अनुमान दिखाई देने वाले लक्षणों से broadly मेल खाता है, लेकिन यह पक्की पुष्टि नहीं है।",
                "mr": "AI च्या दुसऱ्या मतानुसार अंदाज दिसणाऱ्या लक्षणांशी साधारण जुळतो; तरीही ही निश्चित निदान-पुष्टी नाही.",
            },
            "disagree": {
                "en": "The AI second opinion did not support the model's prediction. No disease diagnosis is confirmed.",
                "hi": "AI की दूसरी राय मॉडल के अनुमान का समर्थन नहीं करती। किसी रोग की पुष्टि नहीं हुई है।",
                "mr": "AI चे दुसरे मत मॉडेलच्या अंदाजाला समर्थन देत नाही. कोणत्याही रोगाची खात्री झालेली नाही.",
            },
            "uncertain": {
                "en": "The image does not provide enough clear evidence to verify the predicted disease.",
                "hi": "तस्वीर में अनुमानित रोग की पुष्टि के लिए पर्याप्त स्पष्ट प्रमाण नहीं हैं।",
                "mr": "अंदाजित रोगाची पडताळणी करण्यासाठी प्रतिमेत पुरेसे स्पष्ट पुरावे नाहीत.",
            },
            "not_plant": {
                "en": "The image reviewer could not identify a plant/crop suitable for disease diagnosis.",
                "hi": "छवि समीक्षक को रोग पहचान के लिए उपयुक्त पौधा/फसल नहीं दिखी।",
                "mr": "प्रतिमा तपासणाऱ्याला रोग ओळखण्यासाठी योग्य वनस्पती/पीक दिसले नाही.",
            },
        }
        return {
            "status": "verified" if verdict == "match" else verdict,
            "verdict": verdict,
            "message": messages[verdict].get(lang, messages[verdict]["en"]),
            "observed_crop": str(report.get("observed_crop", "unknown"))[:100],
            "observed_disease": str(report.get("observed_disease", "uncertain"))[:160],
            "reason": reason,
        }
    except Exception:
        # Fail closed: errors never count as agreement.
        return {
            "status": "unavailable",
            "verdict": "uncertain",
            "message": {
                "en": "The AI image review could not be completed, so this prediction remains unverified.",
                "hi": "AI छवि समीक्षा पूरी नहीं हुई, इसलिए यह अनुमान सत्यापित नहीं है।",
                "mr": "AI प्रतिमा पुनरावलोकन पूर्ण झाले नाही, त्यामुळे हा अंदाज पडताळलेला नाही.",
            }.get(lang, "The AI image review could not be completed, so this prediction remains unverified."),
            "observed_crop": "", "observed_disease": "", "reason": "",
        }
