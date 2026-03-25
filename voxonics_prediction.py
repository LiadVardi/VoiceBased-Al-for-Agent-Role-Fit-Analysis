import os
import pickle
import warnings
import numpy as np
from tensorflow.keras.models import load_model

from src.audio_pipeline import extract_features_from_file
from src.config import MODEL_PATH, SCALER_PATH, ENCODER_PATH

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def load_engine():
    print("--- Loading Voxonics Engine ---")
    model = load_model(MODEL_PATH)

    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

    with open(ENCODER_PATH, 'rb') as f:
        encoder = pickle.load(f)

    return model, scaler, encoder


def predict_emotions_report(audio_path):
    model, scaler, encoder = load_engine()

    # Uses FULL pipeline from audio_pipeline.py:
    # RMS normalization + TRIM_TOP_DB=30 + crop/pad + MFCC
    features = extract_features_from_file(audio_path)

    features_scaled = scaler.transform(features.reshape(1, -1))
    features_cnn = np.expand_dims(features_scaled, axis=2)

    predictions = model.predict(features_cnn, verbose=0)[0]
    emotion_labels = encoder.categories_[0]

    return {label: prob * 100 for label, prob in zip(emotion_labels, predictions)}


if __name__ == "__main__":
    import sys

    test_audio = sys.argv[1] if len(sys.argv) > 1 else "test_voice.wav"

    if os.path.exists(test_audio):
        print(f"Analyzing: {test_audio}")
        results = predict_emotions_report(test_audio)

        print("=" * 40)
        for emotion, confidence in sorted(results.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(confidence / 5)
            print(f"{emotion.ljust(12)} | {confidence:6.2f}% {bar}")
        print("=" * 40)

    else:
        print(f"File {test_audio} not found.")