from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np
from tensorflow.keras.models import load_model

from audio_pipeline import extract_features_from_file
from config import MODEL_PATH, SCALER_PATH, ENCODER_PATH


def load_artifacts(
    model_path: str = MODEL_PATH,
    scaler_path: str = SCALER_PATH,
    encoder_path: str = ENCODER_PATH,
):
    """
    Load the trained model, scaler, and encoder from disk.
    """
    model_file = Path(model_path)
    scaler_file = Path(scaler_path)
    encoder_file = Path(encoder_path)

    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")
    if not scaler_file.exists():
        raise FileNotFoundError(f"Scaler file not found: {scaler_file}")
    if not encoder_file.exists():
        raise FileNotFoundError(f"Encoder file not found: {encoder_file}")

    model = load_model(str(model_file))

    with open(scaler_file, "rb") as f:
        scaler = pickle.load(f)

    with open(encoder_file, "rb") as f:
        encoder = pickle.load(f)

    return model, scaler, encoder


def decode_prediction(predictions: np.ndarray, encoder) -> tuple[str, float]:
    """
    Convert model output probabilities into:
    - predicted label
    - confidence score

    First, try to preserve the project's current decoding behavior.
    If that fails, fall back to a LabelEncoder-style decode.
    """
    confidence_score = float(np.max(predictions))

    # Keep backward compatibility with the current project behavior
    try:
        emotion_label = encoder.inverse_transform(predictions)[0][0]
        return str(emotion_label), confidence_score
    except Exception:
        pass

    # Fallback for encoders that expect class indices
    try:
        predicted_index = np.argmax(predictions, axis=1)
        emotion_label = encoder.inverse_transform(predicted_index)[0]
        return str(emotion_label), confidence_score
    except Exception as e:
        raise RuntimeError(f"Failed to decode prediction with encoder: {e}") from e


def predict_emotion(audio_path: str, model, scaler, encoder) -> tuple[str, float]:
    """
    Process an audio file and return:
    - predicted emotion label
    - confidence score
    """
    audio_file = Path(audio_path)
    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")

    # Step A: Feature extraction through the shared audio pipeline
    features = extract_features_from_file(audio_file)

    # Step B: Scaling (scaler expects 2D input)
    features_scaled = scaler.transform(features.reshape(1, -1))

    # Step C: Reshape for the current 1D CNN input format
    # [batch, features, channel]
    features_cnn = np.expand_dims(features_scaled, axis=2)

    # Step D: Model prediction
    predictions = model.predict(features_cnn, verbose=0)

    # Step E: Decode label + confidence
    emotion_label, confidence_score = decode_prediction(predictions, encoder)

    return emotion_label, confidence_score


def main():
    """
    Manual test entrypoint.
    Replace the test_audio path with the file you want to test.
    """
    test_audio = r"C:\Users\liad7\Projects\ASVP-ESD\actor_53\ASVD_EDS-53-Happy-1-dup7.wav"

    try:
        model, scaler, encoder = load_artifacts()
        label, score = predict_emotion(test_audio, model, scaler, encoder)

        print("-" * 30)
        print(f"File: {test_audio}")
        print(f"Predicted Emotion: {label}")
        print(f"Confidence: {score * 100:.2f}%")
        print("-" * 30)

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()