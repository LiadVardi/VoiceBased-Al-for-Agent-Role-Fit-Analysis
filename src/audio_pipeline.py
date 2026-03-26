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
    NORMALIZE_TYPE,
    TARGET_RMS,
    TRIM_SILENCE,
    TRIM_TOP_DB,
    TARGET_DURATION_SEC,
    MIN_DURATION_SEC,
    CROP_MODE,
    PAD_MODE,
    N_MFCC,
    N_FEATURES,
    EPSILON,
    N_MELS,
    HOP_LENGTH,
    FIXED_FRAMES,
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
    method: str = NORMALIZE_TYPE,
    target_rms: float = TARGET_RMS,
    epsilon: float = EPSILON,
) -> np.ndarray:
    """Normalize audio level using Peak or RMS."""
    if audio.size == 0:
        return audio.astype(np.float32)

    if method == "peak":
        peak = float(np.max(np.abs(audio)))
        if peak < epsilon:
            return audio.astype(np.float32)
        return (audio / peak).astype(np.float32)
        
    elif method == "rms":
        current_rms = float(np.sqrt(np.mean(audio**2)))
        if current_rms < epsilon:
            return audio.astype(np.float32)
        return (audio * (target_rms / current_rms)).astype(np.float32)
        
    else:
        raise ValueError(f"Unknown NORMALIZE_TYPE: {method}")


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
    Extract a 252-dimensional feature vector from a preprocessed audio array.

    Feature layout (must match training — see get_feature_names() for full schema):
        [0   : 40 ]  MFCC mean              (n_mfcc values)
        [40  : 80 ]  MFCC std
        [80  : 120]  delta-MFCC mean
        [120 : 160]  delta-MFCC std
        [160 : 200]  delta²-MFCC mean
        [200 : 240]  delta²-MFCC std
        [240 : 242]  RMS energy             (mean, std)
        [242 : 244]  Zero-crossing rate     (mean, std)
        [244 : 246]  Spectral centroid      (mean, std)
        [246 : 248]  Spectral bandwidth     (mean, std)
        [248 : 250]  Spectral rolloff       (mean, std)
        [250 : 252]  F0 / pitch             (mean, std over voiced frames)

    Total: N_FEATURES = 252
    """
    def _ms(x: np.ndarray) -> np.ndarray:
        """Mean and std of any array, flattened to a 2-element float32 vector."""
        flat = x.ravel().astype(np.float64)
        return np.array([flat.mean(), flat.std()], dtype=np.float32)

    # --- MFCCs and temporal derivatives ---
    mfcc    = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)  # (n_mfcc, T)
    d_mfcc  = librosa.feature.delta(mfcc)                           # (n_mfcc, T)
    dd_mfcc = librosa.feature.delta(mfcc, order=2)                  # (n_mfcc, T)

    # --- Energy / spectral / prosodic features ---
    rms       = librosa.feature.rms(y=audio)                        # (1, T)
    zcr       = librosa.feature.zero_crossing_rate(y=audio)         # (1, T)
    centroid  = librosa.feature.spectral_centroid(y=audio, sr=sr)   # (1, T)
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)  # (1, T)
    rolloff   = librosa.feature.spectral_rolloff(y=audio, sr=sr)    # (1, T)

    # --- Pitch (F0) — voiced frames only; zero-padded if none detected ---
    try:
        f0, voiced_flag, _ = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz("C2"),   # ~65 Hz  (below normal speech)
            fmax=librosa.note_to_hz("C7"),   # ~2093 Hz (above normal speech)
            sr=sr,
        )
        f0_voiced = f0[voiced_flag] if (voiced_flag is not None and voiced_flag.any()) else np.zeros(1)
    except Exception:
        f0_voiced = np.zeros(1)

    parts = [
        np.mean(mfcc,    axis=1).astype(np.float32),   # 40
        np.std(mfcc,     axis=1).astype(np.float32),   # 40
        np.mean(d_mfcc,  axis=1).astype(np.float32),   # 40
        np.std(d_mfcc,   axis=1).astype(np.float32),   # 40
        np.mean(dd_mfcc, axis=1).astype(np.float32),   # 40
        np.std(dd_mfcc,  axis=1).astype(np.float32),   # 40
        _ms(rms),        # 2
        _ms(zcr),        # 2
        _ms(centroid),   # 2
        _ms(bandwidth),  # 2
        _ms(rolloff),    # 2
        _ms(f0_voiced),  # 2
    ]

    vec = np.concatenate(parts).astype(np.float32)
    assert len(vec) == N_FEATURES, f"Feature vector length mismatch: {len(vec)} != {N_FEATURES}"
    return vec


def get_feature_names(n_mfcc: int = N_MFCC) -> list[str]:
    """
    Return the ordered list of feature names that matches the extract_features output.

    Use this to build a labelled DataFrame of features:
        pd.DataFrame([extract_features(a, sr)], columns=get_feature_names())
    """
    names: list[str] = []
    for prefix in ("mfcc", "delta_mfcc", "delta2_mfcc"):
        names += [f"{prefix}_mean_{i}" for i in range(n_mfcc)]
        names += [f"{prefix}_std_{i}"  for i in range(n_mfcc)]
    for feat in ("rms", "zcr", "spectral_centroid",
                 "spectral_bandwidth", "spectral_rolloff", "f0"):
        names += [f"{feat}_mean", f"{feat}_std"]
    assert len(names) == N_FEATURES
    return names


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


# ---------------------------------------------------------------------------
# 5. Log-Mel Spectrogram extraction (for 2D CNN)
# ---------------------------------------------------------------------------

def extract_spectrogram(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Compute a normalised log-mel spectrogram with a fixed time dimension.

    Pipeline:
      1. Mel-power spectrogram  → shape (N_MELS, T_actual)
      2. Convert to dB          → human-perceptual scale
      3. Crop or zero-pad T axis to exactly FIXED_FRAMES columns
      4. Normalize to [0, 1]    → matches image CNN expectations

    Returns
    -------
    np.ndarray, shape (N_MELS, FIXED_FRAMES), dtype float32
    """
    # Step 1 — Mel power spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr,
        n_mels=N_MELS,
        hop_length=HOP_LENGTH,
    )  # shape: (N_MELS, T_actual)

    # Step 2 — Convert to dB (log scale)
    mel_db = librosa.power_to_db(mel, ref=np.max)  # values in [-80, 0]

    # Step 3 — Fix time axis to FIXED_FRAMES
    n_frames = mel_db.shape[1]
    if n_frames >= FIXED_FRAMES:
        mel_db = mel_db[:, :FIXED_FRAMES]          # crop
    else:
        pad_width = FIXED_FRAMES - n_frames
        mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode="constant")  # zero-pad

    # Step 4 — Normalize to [0, 1]
    mel_min, mel_max = mel_db.min(), mel_db.max()
    if mel_max - mel_min > EPSILON:
        mel_db = (mel_db - mel_min) / (mel_max - mel_min)

    return mel_db.astype(np.float32)  # shape: (N_MELS, FIXED_FRAMES)


def extract_spectrogram_from_audio_array(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Preprocess an already-loaded audio array and extract a log-mel spectrogram.

    This is the function 02_extract_features.py calls for each sample.
    Pipeline: preprocess_audio → extract_spectrogram.

    Returns shape (N_MELS, FIXED_FRAMES) = (128, 128).
    """
    processed = preprocess_audio(audio, sr)
    return extract_spectrogram(processed, sr)


def extract_spectrogram_from_file(file_path: PathLike) -> np.ndarray:
    """
    One-shot: load from disk → preprocess → extract log-mel spectrogram.

    This is the function voxonics_predictions.py calls for 2D CNN inference.
    Returns shape (N_MELS, FIXED_FRAMES) = (128, 128).
    """
    audio, sr = load_audio(file_path)
    processed = preprocess_audio(audio, sr)
    return extract_spectrogram(processed, sr)