"""
Call_Analysis.py  —  Full emotional timeline of a phone call
=============================================================
Slides a 2.5-second window through the entire audio file and
reports the emotion at every point — showing how emotion shifts
throughout the conversation.

HOW TO RUN:
-----------
Get-ChildItem .\Voxonics_audio\* -Include *.mp3,*.wav | ForEach-Object { .venv\Scripts\python.exe Call_Analysis.py $_.FullName }
"""

from __future__ import annotations

import os
import sys
import pickle
import argparse
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import numpy as np
import librosa
from tensorflow.keras.models import load_model

from audio_pipeline import extract_spectrogram_from_audio_array
from config import MODEL_PATH, ENCODER_PATH, TARGET_SR, MONO, TARGET_DURATION_SEC


def load_engine():
    model = load_model(str(MODEL_PATH))
    with open(ENCODER_PATH, "rb") as f:
        encoder = pickle.load(f)
    return model, encoder


def predict_window(audio_window: np.ndarray, sr: int, model, encoder) -> dict:
    try:
        spec = extract_spectrogram_from_audio_array(audio_window, sr)
        features = spec[np.newaxis, ..., np.newaxis]
        probs = model.predict(features, verbose=0)[0]
        labels = encoder.categories_[0]
        return {label: float(prob) for label, prob in zip(labels, probs)}
    except Exception:
        return {}


def format_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def analyze_call(audio_path: str, stride_sec: float):
    print("\n" + "=" * 56)
    print(f"EMOTION TIMELINE — {os.path.basename(audio_path)}")
    print("=" * 56)

    print("Loading model...")
    model, encoder = load_engine()
    emotion_labels = list(encoder.categories_[0])

    audio, sr = librosa.load(audio_path, sr=TARGET_SR, mono=MONO)
    total_sec = len(audio) / sr

    print(f"Duration : {format_time(total_sec)} ({total_sec:.1f}s)")
    print(f"Stride   : every {stride_sec}s\n")

    window_samples = int(TARGET_DURATION_SEC * sr)
    stride_samples = int(stride_sec * sr)

    timeline = []
    start = 0

    while start + window_samples <= len(audio):
        window = audio[start : start + window_samples]
        start_sec = start / sr
        end_sec = (start + window_samples) / sr

        probs = predict_window(window, sr, model, encoder)

        if probs:
            top_emotion = max(probs, key=probs.get)
            timeline.append((start_sec, end_sec, top_emotion, probs))

        start += stride_samples

    if not timeline:
        print("No speech detected in the file.")
        return

    emotions_raw = [e for _, _, e, _ in timeline]
    emotions_smoothed = list(emotions_raw)

    for i in range(1, len(emotions_raw) - 1):
        prev_e = emotions_raw[i - 1]
        curr_e = emotions_raw[i]
        next_e = emotions_raw[i + 1]

        if curr_e == "angry" and prev_e != "angry" and next_e != "angry":
            emotions_smoothed[i] = prev_e

    timeline = [
        (s, e, emotions_smoothed[i], p)
        for i, (s, e, _, p) in enumerate(timeline)
    ]

    print(f"{'TIME':<14}  {'EMOTION':<10}  CONFIDENCE")
    print("-" * 48)

    for i, (start_sec, end_sec, emotion, probs) in enumerate(timeline):
        conf = probs[emotion] * 100
        smoothed = " *" if emotion != emotions_raw[i] else ""
        print(f"{format_time(start_sec)} - {format_time(end_sec):<8}  {emotion:<10}  {conf:5.1f}%{smoothed}")

    if any(e != r for e, r in zip([e for _, _, e, _ in timeline], emotions_raw)):
        print("  (* = corrected by smoothing)")

    print("\n" + "-" * 48)
    print("SUMMARY")
    print("-" * 48)

    emotion_counts = {e: 0 for e in emotion_labels}
    for _, _, emotion, _ in timeline:
        emotion_counts[emotion] += 1

    total_windows = len(timeline)

    for emotion in emotion_labels:
        pct = emotion_counts[emotion] / total_windows * 100
        print(f"{emotion:<10}  {pct:5.1f}%")

    print("=" * 56)


def main():
    parser = argparse.ArgumentParser(description="Analyze emotion timeline of a phone call.")
    parser.add_argument("audio", help="Path to the audio file (.wav or .mp3)")
    parser.add_argument(
        "--stride", "-s", type=float, default=2.0,
        help="Seconds between analysis windows (default: 2)"
    )

    args = parser.parse_args()

    if not os.path.exists(args.audio):
        print(f"File not found: {args.audio}")
        sys.exit(1)

    analyze_call(args.audio, stride_sec=args.stride)


if __name__ == "__main__":
    main()