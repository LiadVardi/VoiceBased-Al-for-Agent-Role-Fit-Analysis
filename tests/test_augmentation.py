"""tests/test_augmentation.py
Tests for the parameterized augmentation module.
Uses synthetic audio so no real files are needed.
"""
import unittest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import TARGET_SR, TARGET_DURATION_SEC
from augmentation import (
    get_augmentations,
    augment_noise,
    augment_stretch,
    augment_shift,
    augment_pitch,
)


SR = TARGET_SR

def _sine(duration_sec: float = TARGET_DURATION_SEC) -> np.ndarray:
    t = np.linspace(0, duration_sec, int(SR * duration_sec), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)


class TestProfiles(unittest.TestCase):
    """get_augmentations() returns the correct set for each profile."""

    def test_none_profile_returns_empty(self):
        self.assertEqual(len(get_augmentations("none")), 0)

    def test_light_profile_returns_two_techniques(self):
        augs = get_augmentations("light")
        self.assertEqual(len(augs), 2)
        self.assertIn("noise", augs)
        self.assertIn("pitch", augs)

    def test_full_profile_returns_four_techniques(self):
        augs = get_augmentations("full")
        self.assertEqual(len(augs), 4)
        for key in ("noise", "stretch", "shift", "pitch"):
            self.assertIn(key, augs)

    def test_invalid_profile_raises_value_error(self):
        with self.assertRaises(ValueError):
            get_augmentations("super_heavy")


class TestAugmentNoise(unittest.TestCase):
    def test_output_same_shape(self):
        audio = _sine()
        out = augment_noise(audio, SR)
        self.assertEqual(out.shape, audio.shape)

    def test_output_is_float32(self):
        out = augment_noise(_sine(), SR)
        self.assertEqual(out.dtype, np.float32)

    def test_output_differs_from_input(self):
        audio = _sine()
        out = augment_noise(audio, SR)
        self.assertFalse(np.array_equal(out, audio))


class TestAugmentStretch(unittest.TestCase):
    def test_output_is_float32(self):
        out = augment_stretch(_sine(), SR)
        self.assertEqual(out.dtype, np.float32)

    def test_output_not_empty(self):
        out = augment_stretch(_sine(), SR)
        self.assertGreater(len(out), 0)


class TestAugmentShift(unittest.TestCase):
    def test_output_same_shape(self):
        audio = _sine()
        out = augment_shift(audio, SR)
        self.assertEqual(out.shape, audio.shape)

    def test_output_is_float32(self):
        out = augment_shift(_sine(), SR)
        self.assertEqual(out.dtype, np.float32)


class TestAugmentPitch(unittest.TestCase):
    def test_output_is_float32(self):
        out = augment_pitch(_sine(), SR)
        self.assertEqual(out.dtype, np.float32)

    def test_output_not_empty(self):
        out = augment_pitch(_sine(), SR)
        self.assertGreater(len(out), 0)


class TestAugmentationsIntegration(unittest.TestCase):
    """Run the full pipeline: extract features from every augmented variant."""

    def test_all_variants_produce_valid_feature_vectors(self):
        from audio_pipeline import extract_features_from_audio_array
        from config import N_FEATURES

        audio = _sine()
        for aug_name, aug_fn in get_augmentations("full").items():
            with self.subTest(augmentation=aug_name):
                aug_audio = aug_fn(audio, SR)
                feat = extract_features_from_audio_array(aug_audio, SR)
                self.assertEqual(len(feat), N_FEATURES)
                self.assertFalse(np.any(np.isnan(feat)), f"NaN in features for {aug_name}")
                self.assertFalse(np.any(np.isinf(feat)), f"Inf in features for {aug_name}")


if __name__ == "__main__":
    unittest.main()
