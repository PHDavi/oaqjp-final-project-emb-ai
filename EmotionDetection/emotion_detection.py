"""Emotion detection using the IBM Skills Network Watson NLP endpoint."""

import requests

WATSON_URL = (
    "https://sn-watson-emotion.labs.skills.network/v1/"
    "watson.runtime.nlp.v1/NlpService/EmotionPredict"
)
MODEL_ID = "emotion_aggregated-workflow_lang_en_stock"
EMOTIONS = ("anger", "disgust", "fear", "joy", "sadness")


def _empty_result():
    """Return the required response for invalid input."""
    return {**dict.fromkeys(EMOTIONS), "dominant_emotion": None}


def emotion_detector(text_to_analyze):
    """Analyze text and return emotion scores plus the dominant emotion."""
    if not isinstance(text_to_analyze, str) or not text_to_analyze.strip():
        return _empty_result()

    response = requests.post(
        WATSON_URL,
        headers={"grpc-metadata-mm-model-id": MODEL_ID},
        json={"raw_document": {"text": text_to_analyze}},
        timeout=15,
    )
    if response.status_code == 400:
        return _empty_result()
    response.raise_for_status()

    scores = response.json()["emotionPredictions"][0]["emotion"]
    result = {name: scores[name] for name in EMOTIONS}
    result["dominant_emotion"] = max(EMOTIONS, key=result.get)
    return result
