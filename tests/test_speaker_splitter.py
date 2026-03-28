"""tests/test_speaker_splitter.py
Tests for speaker ID parsing and speaker-aware splitting logic.
Uses a synthetic in-memory DataFrame — no Azure connection needed.
"""
import unittest
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from speaker_splitter import (
    extract_speaker_id,
    add_speaker_column,
    speaker_three_way_split,
)


class TestExtractSpeakerId(unittest.TestCase):
    """extract_speaker_id() correctly parses all four dataset formats."""

    def test_ravdess_extracts_actor_folder(self):
        path = "RAVDESS/Actor_01/RAVDESS-01-Angry-1-dup0.wav"
        self.assertEqual(extract_speaker_id(path), "RAVDESS_Actor_01")

    def test_ravdess_different_actor(self):
        path = "RAVDESS/Actor_12/RAVDESS-12-Happy-2-dup1.wav"
        self.assertEqual(extract_speaker_id(path), "RAVDESS_Actor_12")

    def test_cremad_extracts_actor_id(self):
        path = "CREMAD/1001_DFA_ANG_XX.wav"
        self.assertEqual(extract_speaker_id(path), "CREMAD_1001")

    def test_cremad_different_actor(self):
        path = "CREMAD/1045_MTI_HAP_HI.wav"
        self.assertEqual(extract_speaker_id(path), "CREMAD_1045")

    def test_tess_oaf_speaker(self):
        path = "TESS/OAF_angry/OAF_angry_01.wav"
        self.assertEqual(extract_speaker_id(path), "TESS_OAF")

    def test_tess_yaf_speaker(self):
        path = "TESS/YAF_happy/YAF_happy_01.wav"
        self.assertEqual(extract_speaker_id(path), "TESS_YAF")

    def test_savee_dc_speaker(self):
        path = "SAVEE/DC/DC_a01.wav"
        self.assertEqual(extract_speaker_id(path), "SAVEE_DC")

    def test_savee_je_speaker(self):
        path = "SAVEE/JE/JE_h01.wav"
        self.assertEqual(extract_speaker_id(path), "SAVEE_JE")


class TestAddSpeakerColumn(unittest.TestCase):
    """add_speaker_column() enriches a DataFrame correctly."""

    def _make_df(self):
        return pd.DataFrame({
            "Path": [
                "RAVDESS/Actor_01/RAVDESS-01-Angry-1-dup0.wav",
                "CREMAD/1001_DFA_ANG_XX.wav",
                "TESS/OAF_angry/OAF_angry_01.wav",
                "SAVEE/DC/DC_a01.wav",
            ],
            "Emotions": ["angry", "angry", "angry", "angry"],
        })

    def test_speaker_id_column_added(self):
        df = add_speaker_column(self._make_df())
        self.assertIn("speaker_id", df.columns)

    def test_correct_speaker_ids(self):
        df = add_speaker_column(self._make_df())
        expected = ["RAVDESS_Actor_01", "CREMAD_1001", "TESS_OAF", "SAVEE_DC"]
        self.assertEqual(list(df["speaker_id"]), expected)

    def test_original_df_not_mutated(self):
        original = self._make_df()
        _ = add_speaker_column(original)
        self.assertNotIn("speaker_id", original.columns)


class TestSpeakerThreeWaySplit(unittest.TestCase):
    """speaker_three_way_split() produces disjoint speaker sets."""

    def _make_large_df(self) -> pd.DataFrame:
        """Build a synthetic 120-row DataFrame with 12 unique speakers."""
        rows = []
        speakers = [
            ("RAVDESS/Actor_01/", "RAVDESS_Actor_01"),
            ("RAVDESS/Actor_02/", "RAVDESS_Actor_02"),
            ("RAVDESS/Actor_03/", "RAVDESS_Actor_03"),
            ("RAVDESS/Actor_04/", "RAVDESS_Actor_04"),
            ("CREMAD/1001_DFA_ANG_XX.wav", "CREMAD_1001"),
            ("CREMAD/1002_MTI_SAD_LO.wav", "CREMAD_1002"),
            ("CREMAD/1003_IEO_HAP_MD.wav", "CREMAD_1003"),
            ("CREMAD/1004_DFA_NEU_XX.wav", "CREMAD_1004"),
            ("TESS/OAF_angry/OAF_angry_01.wav", "TESS_OAF"),
            ("TESS/YAF_happy/YAF_happy_01.wav", "TESS_YAF"),
            ("SAVEE/DC/DC_a01.wav", "SAVEE_DC"),
            ("SAVEE/JE/JE_h01.wav", "SAVEE_JE"),
        ]
        for i, (path_prefix, _) in enumerate(speakers):
            for j in range(10):
                rows.append({
                    "Path":     f"{path_prefix}file_{j}.wav" if not path_prefix.endswith(".wav") else path_prefix,
                    "Emotions": "angry",
                })
        df = pd.DataFrame(rows)
        return add_speaker_column(df)

    def test_no_speaker_overlap(self):
        df = self._make_large_df()
        train_df, val_df, test_df = speaker_three_way_split(
            df, val_size=0.2, test_size=0.2, random_state=42
        )
        train_spk = set(train_df["speaker_id"])
        val_spk   = set(val_df["speaker_id"])
        test_spk  = set(test_df["speaker_id"])
        self.assertEqual(len(train_spk & val_spk), 0)
        self.assertEqual(len(train_spk & test_spk), 0)
        self.assertEqual(len(val_spk & test_spk), 0)

    def test_all_rows_accounted_for(self):
        df = self._make_large_df()
        train_df, val_df, test_df = speaker_three_way_split(
            df, val_size=0.2, test_size=0.2, random_state=42
        )
        self.assertEqual(len(train_df) + len(val_df) + len(test_df), len(df))

    def test_test_is_non_empty(self):
        df = self._make_large_df()
        _, _, test_df = speaker_three_way_split(df, val_size=0.15, test_size=0.15)
        self.assertGreater(len(test_df), 0)

    def test_missing_speaker_column_raises(self):
        df = pd.DataFrame({"Path": ["RAVDESS/Actor_01/file.wav"]})
        with self.assertRaises(ValueError):
            speaker_three_way_split(df)


if __name__ == "__main__":
    unittest.main()
