"""Unit tests for Watson NLP emotion detection."""

import unittest
from unittest.mock import Mock, patch

from EmotionDetection import emotion_detector


class TestEmotionDetector(unittest.TestCase):
    """Check the five emotions and invalid input behavior."""

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_five_dominant_emotions(self, mock_post):
        """Each emotion can be identified as dominant."""
        for emotion in ("joy", "anger", "disgust", "sadness", "fear"):
            with self.subTest(emotion=emotion):
                scores = dict.fromkeys(
                    ("anger", "disgust", "fear", "joy", "sadness"), 0.05
                )
                scores[emotion] = 0.8
                mock_post.return_value = Mock(
                    status_code=200,
                    **{"json.return_value": {
                        "emotionPredictions": [{"emotion": scores}]
                    }},
                )
                result = emotion_detector(f"Example of {emotion}")
                self.assertEqual(result["dominant_emotion"], emotion)
                self.assertEqual(result[emotion], 0.8)

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_blank_input(self, mock_post):
        """Blank text should not call the service."""
        result = emotion_detector("   ")
        self.assertTrue(all(value is None for value in result.values()))
        mock_post.assert_not_called()

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_http_400(self, mock_post):
        """HTTP 400 should return an empty emotion result."""
        mock_post.return_value = Mock(status_code=400)
        result = emotion_detector("Invalid input")
        self.assertIsNone(result["dominant_emotion"])


if __name__ == "__main__":
    unittest.main()
