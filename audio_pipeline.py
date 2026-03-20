"""
audio_pipeline.py
=================
Single source of truth for ALL audio loading, preprocessing, and feature
extraction in this project.

Official pipeline (order matters):
  1. Load  ->  resample to TARGET_SR, convert to mono
  2. Normalize  (only if NORMALIZE_AUDIO=True in config)
  3. Trim silence
  4. Crop or pad to fixed length
  5. Extract features: 40 MFCC means + 40 MFCC stds  ->  shape (80,)

Both training (notebook) and inference (voxonics_prediction.py) MUST call
functions from this module.  Do NOT copy-paste audio logic anywhere else.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Union

import librosa
import numpy as np

from config import (
    TARGET_SR,
    MONO,
    NORMALIZE_AUDIO,
    TRIM_SILENCE,
    TRIM_TOP_DB,
    TARGET_DURATION_SEC,
    MIN_DURATION_SEC,
    CROP_MODE,
    PAD_MODE,
    N_MFCC,
    EPSILON,
)

class AudioProcessingError(Exception):
    """Base exception for all audio preprocessing failures."""

class AudioEmptyError(AudioProcessingError):
    """Raised when the audio array is empty or purely silent (max_val == 0)."""

class AudioTooShortError(AudioProcessingError):
    """Raised when the audio array length after trimming is below the minimum allowed duration."""

PathLike = Union[str, Path]


# 1. Loading

def load_audio( #loads the audio file from the computer
    file_path: PathLike,
    target_sr: int = TARGET_SR,
    mono: bool = MONO,
) -> tuple[np.ndarray, int]:
    audio, sr = librosa.load(str(file_path), sr=target_sr, mono=mono)
    return audio.astype(np.float32), sr


def load_audio_from_bytes( #loads the audio file from azure
    audio_bytes: bytes | io.BytesIO,
    target_sr: int = TARGET_SR,
    mono: bool = MONO,
) -> tuple[np.ndarray, int]:
    
    if isinstance(audio_bytes, bytes):
        audio_bytes = io.BytesIO(audio_bytes)
    audio, sr = librosa.load(audio_bytes, sr=target_sr, mono=mono)
    return audio.astype(np.float32), sr


# 2. Cleaning helpers
def normalize_audio(
    audio: np.ndarray,
    epsilon: float = EPSILON,
) -> np.ndarray:

    #Peak-normalize audio to [-1, 1].

    if audio.size == 0:
        return audio.astype(np.float32)
    peak = float(np.max(np.abs(audio)))
    if peak < epsilon:
        return audio.astype(np.float32)
    return (audio / peak).astype(np.float32)


def trim_silence(
    audio: np.ndarray,
    top_db: int = TRIM_TOP_DB,
) -> np.ndarray:
    #Remove leading and trailing silence
    if audio.size == 0:
        return audio.astype(np.float32)
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed.astype(np.float32)


def crop_or_pad_audio(
    audio: np.ndarray,
    sr: int,
    target_duration_sec: float = TARGET_DURATION_SEC,
    crop_mode: str = CROP_MODE,
    pad_mode: str = PAD_MODE,
) -> np.ndarray:
    #Force audio to a be 2.5 sec.
    #If longer -> crop from start or center.
    #If shorter -> zero-pad (or wrap).
    target_len = int(sr * target_duration_sec)
    current_len = len(audio)

    if current_len == target_len:
        return audio.astype(np.float32)

    if current_len > target_len:
        if crop_mode == "start":
            return audio[:target_len].astype(np.float32)
        elif crop_mode == "center":
            start = (current_len - target_len) // 2
            return audio[start : start + target_len].astype(np.float32)
        else:
            raise ValueError(f"Unsupported crop_mode: {crop_mode!r}")

    pad_len = target_len - current_len
    if pad_mode == "constant":
        padded = np.pad(audio, (0, pad_len), mode="constant")
    elif pad_mode == "wrap":
        padded = (
            np.zeros(target_len, dtype=np.float32)
            if current_len == 0
            else np.pad(audio, (0, pad_len), mode="wrap")
        )
    else:
        raise ValueError(f"Unsupported pad_mode: {pad_mode!r}")

    return padded.astype(np.float32)


# 3. Full preprocessing pipeline
def preprocess_audio(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Steps (all controlled by config.py):
      1. Optional peak normalization  (NORMALIZE_AUDIO)
      2. Trim silence                 (TRIM_SILENCE / TRIM_TOP_DB)
      3. Crop or pad to fixed length  (TARGET_DURATION_SEC / CROP_MODE / PAD_MODE)
    """
    if audio.size == 0 or np.max(np.abs(audio)) == 0:
        raise AudioEmptyError("Audio is completely silent or max_val == 0.")

    processed = audio.astype(np.float32)

    if NORMALIZE_AUDIO:
        processed = normalize_audio(processed)

    if TRIM_SILENCE:
        processed = trim_silence(processed, top_db=TRIM_TOP_DB)

    duration_sec = len(processed) / sr
    if duration_sec < MIN_DURATION_SEC:
        raise AudioTooShortError(f"Audio duration after trimming ({duration_sec:.2f}s) is below minimum ({MIN_DURATION_SEC}s).")

    processed = crop_or_pad_audio(
        processed,
        sr=sr,
        target_duration_sec=TARGET_DURATION_SEC,
        crop_mode=CROP_MODE,
        pad_mode=PAD_MODE,
    )

    return processed.astype(np.float32)


def preprocess_audio_file(file_path: PathLike) -> tuple[np.ndarray, int]:
    """Load a file from disk and run it through the preprocessing pipeline."""
    audio, sr = load_audio(file_path)
    return preprocess_audio(audio, sr), sr



# 4. Feature extraction
def extract_features(audio: np.ndarray, sr: int, n_mfcc: int = N_MFCC) -> np.ndarray:
    """
    Extract an 80-dimensional feature vector from a preprocessed audio array.

    Feature layout (must match training):
        indices  0..39  ->  MFCC means
        indices 40..79  ->  MFCC standard deviations
    """
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    mfcc_std  = np.std(mfcc.T,  axis=0)
    return np.concatenate([mfcc_mean, mfcc_std]).astype(np.float32)


# --- Convenience aliases used across the project ----------------------------

def extract_features_from_audio_array(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Preprocess an already-loaded audio array and extract features.

    This is the function the training notebook calls for each sample
    (original + augmented variants).

    Pipeline: preprocess_audio -> extract_features.
    """
    processed = preprocess_audio(audio, sr)
    return extract_features(processed, sr)


def extract_features_from_file(file_path: PathLike) -> np.ndarray:
    """
    One-shot: load from disk -> preprocess -> extract features.

    This is the function voxonics_prediction.py calls for inference.
    """
    audio, sr = load_audio(file_path)
    processed = preprocess_audio(audio, sr)
    return extract_features(processed, sr)