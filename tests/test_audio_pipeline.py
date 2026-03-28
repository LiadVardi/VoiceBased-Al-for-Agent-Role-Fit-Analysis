"""tests/test_audio_pipeline.py
Tests for all audio loading, preprocessing, and feature extraction functions.
Uses synthetic audio (sine wave + silence) so no real files are needed.
"""
import io
import unittest

import numpy as np
import soundfile as sf

# ── make the project root importable when running from /tests ─────────────────
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import TARGET_SR, N_FEATURES, TARGET_DURATION_SEC, TARGET_RMS
from audio_pipeline import (
    load_audio_from_bytes,
    normalize_audio,
    trim_silence,
    crop_or_pad_audio,
    extract_features,
    preprocess_audio,
    AudioEmptyError,
    AudioTooShortError,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _sine(duration_sec: float = 2.5, freq: float = 440.0, sr: int = TARGET_SR) -> np.ndarray:
    """Generate a pure sine wave as a float32 array."""
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _wav_bytes(audio: np.ndarray, sr: int = TARGET_SR) -> bytes:
    """Encode a numpy array as WAV bytes in memory."""
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


SR = TARGET_SR


# ── Test classes ──────────────────────────────────────────────────────────────

class TestLoadAudio(unittest.TestCase):
    """Audio loading from bytes."""

    def test_load_from_bytes_returns_correct_sr(self):
        audio = _sine(2.0)
        data, sr = load_audio_from_bytes(_wav_bytes(audio))
        self.assertEqual(sr, SR)

    def test_load_from_bytes_is_float32(self):
        audio = _sine(2.0)
        data, _ = load_audio_from_bytes(_wav_bytes(audio))
        self.assertEqual(data.dtype, np.float32)

    def test_load_from_bytes_mono(self):
        audio = _sine(1.0)
        data, _ = load_audio_from_bytes(_wav_bytes(audio))
        self.assertEqual(data.ndim, 1)


class TestNormalizeAudio(unittest.TestCase):
    """normalize_audio() — RMS and peak modes."""

    def test_rms_normalization_target_level(self):
        audio = _sine(1.0) * 10          # very loud
        out = normalize_audio(audio, method="rms", target_rms=TARGET_RMS)
        actual_rms = float(np.sqrt(np.mean(out ** 2)))
        self.assertAlmostEqual(actual_rms, TARGET_RMS, places=4)

    def test_peak_normalization_max_is_one(self):
        audio = _sine(1.0) * 0.1         # quiet
        out = normalize_audio(audio, method="peak")
        self.assertAlmostEqual(float(np.max(np.abs(out))), 1.0, places=5)

    def test_silent_audio_unchanged(self):
        silent = np.zeros(SR, dtype=np.float32)
        out = normalize_audio(silent, method="rms")
        np.testing.assert_array_equal(out, silent)

    def test_output_is_float32(self):
        audio = _sine(1.0)
        out = normalize_audio(audio, method="rms")
        self.assertEqual(out.dtype, np.float32)


class TestTrimSilence(unittest.TestCase):
    """trim_silence() removes leading/trailing silence."""

    def test_trim_removes_silence(self):
        silence = np.zeros(SR, dtype=np.float32)
        speech  = _sine(1.0)
        audio   = np.concatenate([silence, speech, silence])
        trimmed = trim_silence(audio, top_db=30)
        # Trimmed result must be shorter than the padded version
        self.assertLess(len(trimmed), len(audio))

    def test_trim_keeps_content(self):
        audio   = _sine(1.0)
        trimmed = trim_silence(audio, top_db=30)
        # A pure sine has no silence to remove; length should be very similar
        self.assertGreater(len(trimmed), 0)

    def test_empty_array_unchanged(self):
        empty = np.array([], dtype=np.float32)
        out = trim_silence(empty)
        self.assertEqual(len(out), 0)


class TestCropOrPad(unittest.TestCase):
    """crop_or_pad_audio() enforces TARGET_DURATION_SEC."""

    TARGET_LEN = int(SR * TARGET_DURATION_SEC)

    def test_pads_short_audio(self):
        short = _sine(1.0)                               # shorter than 2.5s
        out = crop_or_pad_audio(short, SR)
        self.assertEqual(len(out), self.TARGET_LEN)

    def test_crops_long_audio(self):
        long = _sine(5.0)                                # longer than 2.5s
        out = crop_or_pad_audio(long, SR)
        self.assertEqual(len(out), self.TARGET_LEN)

    def test_exact_length_unchanged(self):
        exact = _sine(TARGET_DURATION_SEC)
        out = crop_or_pad_audio(exact, SR)
        self.assertEqual(len(out), self.TARGET_LEN)

    def test_output_is_float32(self):
        out = crop_or_pad_audio(_sine(1.0), SR)
        self.assertEqual(out.dtype, np.float32)

    def test_invalid_crop_mode_raises(self):
        with self.assertRaises(ValueError):
            crop_or_pad_audio(_sine(5.0), SR, crop_mode="invalid_mode")

    def test_invalid_pad_mode_raises(self):
        with self.assertRaises(ValueError):
            crop_or_pad_audio(_sine(1.0), SR, pad_mode="invalid_mode")


class TestExtractFeatures(unittest.TestCase):
    """extract_features() returns correct shape."""

    def test_feature_vector_length(self):
        audio = _sine(TARGET_DURATION_SEC)
        feat  = extract_features(audio, SR)
        self.assertEqual(len(feat), N_FEATURES)

    def test_feature_vector_is_float32(self):
        audio = _sine(TARGET_DURATION_SEC)
        feat  = extract_features(audio, SR)
        self.assertEqual(feat.dtype, np.float32)

    def test_feature_vector_no_nan(self):
        audio = _sine(TARGET_DURATION_SEC)
        feat  = extract_features(audio, SR)
        self.assertFalse(np.any(np.isnan(feat)))

    def test_feature_vector_no_inf(self):
        audio = _sine(TARGET_DURATION_SEC)
        feat  = extract_features(audio, SR)
        self.assertFalse(np.any(np.isinf(feat)))


class TestPreprocessAudio(unittest.TestCase):
    """preprocess_audio() guards for empty / too-short audio."""

    def test_empty_array_raises_audio_empty_error(self):
        with self.assertRaises(AudioEmptyError):
            preprocess_audio(np.array([], dtype=np.float32), SR)

    def test_silent_array_raises_audio_empty_error(self):
        with self.assertRaises(AudioEmptyError):
            preprocess_audio(np.zeros(SR, dtype=np.float32), SR)

    def test_too_short_after_trim_raises_audio_too_short_error(self):
        # 50ms of audio — will be shorter than MIN_DURATION_SEC after trim
        tiny = _sine(0.05)
        with self.assertRaises(AudioTooShortError):
            preprocess_audio(tiny, SR)

    def test_valid_audio_returns_correct_length(self):
        audio = _sine(2.0)
        out = preprocess_audio(audio, SR)
        expected_len = int(SR * TARGET_DURATION_SEC)
        self.assertEqual(len(out), expected_len)


if __name__ == "__main__":
    unittest.main()
