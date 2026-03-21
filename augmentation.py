"""
augmentation.py
===============
Parameterized audio augmentation for Speech Emotion Recognition.

Three profiles are supported (controlled by AUGMENTATION_PROFILE in config.py):

  "none"  → returns only the original audio (no augmentation)
  "light" → noise + mild pitch shift
  "full"  → noise, time-stretch, time-shift, and pitch shift

All parameter ranges are centralised in config.py so they can be
adjusted without touching this file.

Usage inside build_feature_dataframe
-------------------------------------
    from augmentation import get_augmentations

    for aug_name, aug_fn in get_augmentations().items():
        aug_data = aug_fn(data, sr)
        aug_records.append({"source_original_path": path,
                            "augmentation_type":   aug_name})
"""
from __future__ import annotations

import numpy as np
import librosa

from config import (
    AUGMENTATION_PROFILE,
    NOISE_AMP_RANGE,
    STRETCH_RATE_RANGE,
    SHIFT_MS_RANGE,
    PITCH_STEPS_RANGE,
    TARGET_SR,
)


# ---------------------------------------------------------------------------
# Individual augmentation functions  (each takes (data, sr) → np.ndarray)
# ---------------------------------------------------------------------------

def augment_noise(data: np.ndarray, sr: int) -> np.ndarray:
    """Add Gaussian noise scaled to a random fraction of the signal peak."""
    lo, hi = NOISE_AMP_RANGE
    amp = np.random.uniform(lo, hi) * np.amax(np.abs(data))
    noise = amp * np.random.normal(size=data.shape[0])
    return (data + noise).astype(np.float32)


def augment_stretch(data: np.ndarray, sr: int) -> np.ndarray:
    """Randomly speed up or slow down the audio without changing pitch."""
    lo, hi = STRETCH_RATE_RANGE
    rate = np.random.uniform(lo, hi)
    return librosa.effects.time_stretch(y=data, rate=rate).astype(np.float32)


def augment_shift(data: np.ndarray, sr: int) -> np.ndarray:
    """Randomly shift the audio forward or backward in time."""
    lo_ms, hi_ms = SHIFT_MS_RANGE
    shift_samples = int(np.random.uniform(lo_ms, hi_ms) / 1000 * sr)
    return np.roll(data, shift_samples).astype(np.float32)


def augment_pitch(data: np.ndarray, sr: int) -> np.ndarray:
    """Randomly shift the pitch up or down by a number of semitones."""
    lo, hi = PITCH_STEPS_RANGE
    n_steps = np.random.uniform(lo, hi)
    return librosa.effects.pitch_shift(y=data, sr=sr, n_steps=n_steps).astype(np.float32)


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------

# All available techniques in a fixed order (also defines metadata labels)
_ALL_AUGMENTATIONS: dict[str, callable] = {
    "noise":   augment_noise,
    "stretch": augment_stretch,
    "shift":   augment_shift,
    "pitch":   augment_pitch,
}

_LIGHT_AUGMENTATIONS: dict[str, callable] = {
    "noise": augment_noise,
    "pitch": augment_pitch,
}

_PROFILES: dict[str, dict[str, callable]] = {
    "none":  {},
    "light": _LIGHT_AUGMENTATIONS,
    "full":  _ALL_AUGMENTATIONS,
}


def get_augmentations(profile: str | None = None) -> dict[str, callable]:
    """
    Return the augmentation functions for the current profile.

    Parameters
    ----------
    profile : str | None
        One of "none", "light", "full".
        If None, reads AUGMENTATION_PROFILE from config.py.

    Returns
    -------
    dict mapping augmentation name -> function(data, sr) -> np.ndarray
    An empty dict means no augmentation (profile="none").

    Example
    -------
        for aug_name, aug_fn in get_augmentations().items():
            aug_data = aug_fn(original_audio, sr)
    """
    chosen = profile or AUGMENTATION_PROFILE
    if chosen not in _PROFILES:
        raise ValueError(
            f"Unknown augmentation profile: {chosen!r}. "
            f"Choose from: {list(_PROFILES)}"
        )
    return _PROFILES[chosen]


def profile_summary(profile: str | None = None) -> None:
    """Print a human-readable description of the active profile."""
    chosen = profile or AUGMENTATION_PROFILE
    augs   = get_augmentations(chosen)

    print(f"Augmentation profile : '{chosen}'")
    if not augs:
        print("  No augmentation will be applied.")
        return

    print(f"  Techniques ({len(augs)}): {', '.join(augs)}")
    print(f"  Noise amplitude   : {NOISE_AMP_RANGE[0]:.3f} – {NOISE_AMP_RANGE[1]:.3f} x peak")
    print(f"  Time-stretch rate : {STRETCH_RATE_RANGE[0]:.2f} – {STRETCH_RATE_RANGE[1]:.2f}x")
    print(f"  Time-shift        : {SHIFT_MS_RANGE[0]}ms – {SHIFT_MS_RANGE[1]}ms")
    print(f"  Pitch shift       : {PITCH_STEPS_RANGE[0]:+.1f} – {PITCH_STEPS_RANGE[1]:+.1f} semitones")
