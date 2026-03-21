import os
import pickle
import warnings
import numpy as np
from tensorflow.keras.models import load_model

from audio_pipeline import extract_features_from_file
from config import MODEL_PATH, SCALER_PATH, ENCODER_PATH

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


FILE_TO_TEST = r"C:\Users\orbit_24ts2or\OneDrive\שולחן העבודה\angry_30.wav"


def load_engine():
    print("--- Loading Voxonics Engine ---")
    model = load_model(MODEL_PATH)
    scaler = pickle.load(open(SCALER_PATH, 'rb'))
    encoder = pickle.load(open(ENCODER_PATH, 'rb'))
    return model, scaler, encoder


def run_prediction():
    if not os.path.exists(FILE_TO_TEST):
        print(f" Error: File not found at {FILE_TO_TEST}")
        return

    model, scaler, encoder = load_engine()

    # Uses the exact same pipeline as training:
    # load → resample → trim silence → crop/pad → extract MFCCs
    # All parameters come from config.py
    features = extract_features_from_file(FILE_TO_TEST)

    features_scaled = scaler.transform(features.reshape(1, -1))
    features_cnn = np.expand_dims(features_scaled, axis=2)

    predictions = model.predict(features_cnn, verbose=0)[0]
    emotion_labels = encoder.categories_[0]

    print(f"\nResults for: {os.path.basename(FILE_TO_TEST)}")
    print("=" * 40)
    results = {label: prob * 100 for label, prob in zip(emotion_labels, predictions)}
    for emotion, confidence in sorted(results.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(confidence / 5)
        print(f"{emotion.ljust(12)} | {confidence:6.2f}% {bar}")
    print("=" * 40)


if __name__ == "__main__":
    run_prediction()