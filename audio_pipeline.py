from __future__ import annotations

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
    CROP_MODE,
    PAD_MODE,
    N_MFCC,
    EPSILON,
)

PathLike = Union[str, Path]


def load_audio_file(
    file_path: PathLike,
    target_sr: int = TARGET_SR,
    mono: bool = MONO,
) -> tuple[np.ndarray, int]:
    """
    Load an audio file with librosa.
    Resamples to target_sr and converts to mono if requested.
    """
    audio, sr = librosa.load(str(file_path), sr=target_sr, mono=mono)
    return audio.astype(np.float32), sr


def normalize_peak(audio: np.ndarray, epsilon: float = EPSILON) -> np.ndarray:
    """
    Peak-normalize audio to [-1, 1] range.
    If the signal is effectively silent, return it unchanged.
    """
    peak = float(np.max(np.abs(audio))) if audio.size > 0 else 0.0
    if peak < epsilon:
        return audio.astype(np.float32)
    return (audio / peak).astype(np.float32)


def trim_silence_fn(audio: np.ndarray, top_db: int = TRIM_TOP_DB) -> np.ndarray:
    """
    Remove leading and trailing silence.
    """
    if audio.size == 0:
        return audio.astype(np.float32)

    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed.astype(np.float32)


def fix_length(
    audio: np.ndarray,
    sr: int,
    target_duration_sec: float = TARGET_DURATION_SEC,
    crop_mode: str = CROP_MODE,
    pad_mode: str = PAD_MODE,
) -> np.ndarray:
    """
    Force audio to a fixed length in samples.
    - If too long: crop
    - If too short: pad
    """
    target_len = int(sr * target_duration_sec)
    current_len = len(audio)

    if current_len == target_len:
        return audio.astype(np.float32)

    if current_len > target_len:
        if crop_mode == "start":
            cropped = audio[:target_len]
        elif crop_mode == "center":
            start = (current_len - target_len) // 2
            cropped = audio[start:start + target_len]
        else:
            raise ValueError(f"Unsupported crop_mode: {crop_mode}")
        return cropped.astype(np.float32)

    # current_len < target_len
    pad_len = target_len - current_len

    if pad_mode == "constant":
        padded = np.pad(audio, (0, pad_len), mode="constant")
    elif pad_mode == "wrap":
        if current_len == 0:
            padded = np.zeros(target_len, dtype=np.float32)
        else:
            padded = np.pad(audio, (0, pad_len), mode="wrap")
    else:
        raise ValueError(f"Unsupported pad_mode: {pad_mode}")

    return padded.astype(np.float32)


def preprocess_audio_array(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Apply the project's official preprocessing pipeline to an audio array.
    """
    processed = audio.astype(np.float32)

    if NORMALIZE_AUDIO:
        processed = normalize_peak(processed)

    if TRIM_SILENCE:
        processed = trim_silence_fn(processed, top_db=TRIM_TOP_DB)

    processed = fix_length(
        processed,
        sr=sr,
        target_duration_sec=TARGET_DURATION_SEC,
        crop_mode=CROP_MODE,
        pad_mode=PAD_MODE,
    )

    return processed.astype(np.float32)


def preprocess_audio_file(file_path: PathLike) -> tuple[np.ndarray, int]:
    """
    Load + preprocess an audio file.
    """
    audio, sr = load_audio_file(file_path)
    processed = preprocess_audio_array(audio, sr)
    return processed, sr


def extract_mfcc_mean(
    audio: np.ndarray,
    sr: int,
    n_mfcc: int = N_MFCC,
) -> np.ndarray:
    """
    Extract MFCC features and return the mean over time.
    This preserves compatibility with the current model.
    """
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    features = np.mean(mfcc.T, axis=0)
    return features.astype(np.float32)


def extract_features_from_audio_array(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Preprocess an audio array and extract the current project's feature vector.
    """
    processed = preprocess_audio_array(audio, sr)
    return extract_mfcc_mean(processed, sr)


def extract_features_from_file(file_path: PathLike) -> np.ndarray:
    """
    Load, preprocess, and extract features from an audio file.
    """
    processed_audio, sr = preprocess_audio_file(file_path)
    return extract_mfcc_mean(processed_audio, sr)