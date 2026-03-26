"""
Voxonics Emotion Predictions
==============================
Predicts the emotion of a speaker from a .wav audio file.

HOW TO RUN:
-----------
🔹 Activate virtual environment (recommended):
.venv\Scripts\activate

🔹 Run a single file:
python Voxonics_Predictions.py Voxonics_audio\your_file.wav

🔹 Run all files in Voxonics_audio folder:
python Voxonics_Predictions.py Voxonics_audio
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import pickle
import warnings
import numpy as np
from tensorflow.keras.models import load_model

from audio_pipeline import extract_spectrogram_from_file
from config import MODEL_PATH, ENCODER_PATH

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def load_engine():
    print("--- Loading Voxonics Engine ---")
    model = load_model(MODEL_PATH)

    with open(ENCODER_PATH, "rb") as f:
        encoder = pickle.load(f)

    return model, encoder


def predict_emotions_report(audio_path):
    model, encoder = load_engine()

    spec = extract_spectrogram_from_file(audio_path)
    features_cnn = spec[np.newaxis, ..., np.newaxis]  # (1, 128, 128, 1)

    predictions = model.predict(features_cnn, verbose=0)[0]
    emotion_labels = encoder.categories_[0]

    return {label: float(prob * 100) for label, prob in zip(emotion_labels, predictions)}


if __name__ == "__main__":
    input_path = sys.argv[1] if len(sys.argv) > 1 else "test_audio"

    if os.path.isdir(input_path):
        print(f"Analyzing folder: {input_path}\n")

        for file in os.listdir(input_path):
            if file.endswith(".wav"):
                path = os.path.join(input_path, file)

                results = predict_emotions_report(path)
                sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

                print("=" * 40)
                print(file)
                print(f"Prediction: {sorted_results[0][0]} ({sorted_results[0][1]:.2f}%)")
                print("-" * 40)

                for emotion, confidence in sorted_results:
                    bar = "█" * int(confidence / 5)
                    print(f"{emotion.ljust(12)} | {confidence:6.2f}% {bar}")

                print("=" * 40)

    elif os.path.isfile(input_path):
        print(f"Analyzing: {input_path}")
        results = predict_emotions_report(input_path)

        print("=" * 40)
        for emotion, confidence in sorted(results.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(confidence / 5)
            print(f"{emotion.ljust(12)} | {confidence:6.2f}% {bar}")
        print("=" * 40)

    else:
        print(f"Path not found: {input_path}")
