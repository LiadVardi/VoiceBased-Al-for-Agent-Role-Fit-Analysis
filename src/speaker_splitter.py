# This code makes sure that the same speaker will not be in the train and test set.

from __future__ import annotations

import re
import warnings
from pathlib import PurePosixPath

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


# 1. Per-dataset speaker parsers

def _speaker_ravdess(parts: list[str]) -> str:
    if len(parts) >= 3:
        return f"RAVDESS_{parts[1]}"
    filename = parts[-1]
    match = re.search(r"RAVDESS-(\d+)-", filename, re.IGNORECASE)
    if match:
        return f"RAVDESS_Actor_{int(match.group(1)):02d}"
    return f"RAVDESS_unknown"


def _speaker_cremad(parts: list[str]) -> str:
   
    filename = parts[-1]
    actor_id = filename.split("_")[0]
    return f"CREMAD_{actor_id}"


def _speaker_tess(parts: list[str]) -> str:
    if len(parts) >= 3:
        folder = parts[1]          
        speaker = folder.split("_")[0]   
        return f"TESS_{speaker}"
    filename = parts[-1]
    speaker = filename.split("_")[0]
    return f"TESS_{speaker}"


def _speaker_savee(parts: list[str]) -> str:
    if len(parts) >= 3:
        return f"SAVEE_{parts[1]}"
    filename = parts[-1]
    return f"SAVEE_{filename[:2]}"


def _speaker_meld(parts: list[str]) -> str:
    """
    MELD path format: MELD/{Speaker}_dia{Dialogue_ID}_utt{Utterance_ID}.wav
    Example: MELD/Ross_dia1_utt3.wav  → speaker = 'Ross'
    """
    filename = parts[-1]          # e.g. Ross_dia1_utt3.wav
    speaker = filename.split("_dia")[0]   # everything before '_dia'
    return f"MELD_{speaker}"


# 2. Dispatcher

_PARSERS = {
    "RAVDESS": _speaker_ravdess,
    "CREMAD":  _speaker_cremad,
    "TESS":    _speaker_tess,
    "SAVEE":   _speaker_savee,
    "MELD":    _speaker_meld,
}


def extract_speaker_id(blob_path: str) -> str:
    path = blob_path.replace("\\", "/")
    parts = path.split("/")

    dataset = parts[0].upper()
    parser = _PARSERS.get(dataset)

    if parser is None:
        warnings.warn(f"Unknown dataset prefix '{dataset}' in path: {blob_path}")
        return f"UNKNOWN_{parts[-1]}"

    return parser(parts)



def add_speaker_column(df: pd.DataFrame, path_col: str = "Path") -> pd.DataFrame:
    df = df.copy()
    df["speaker_id"] = df[path_col].apply(extract_speaker_id)
    return df

def speaker_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    speaker_col: str = "speaker_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:

    train_df, test_df, _ = speaker_three_way_split(
        df,
        val_size=0.0,
        test_size=test_size,
        random_state=random_state,
        speaker_col=speaker_col,
    )
    return train_df, test_df


def speaker_three_way_split( # This is the main function that splits the data into train, val, and test sets.
    df: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    speaker_col: str = "speaker_id",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Speaker-aware three-way split: train / val / test.

    Guarantees no speaker appears in more than one split.

    Strategy
    --------
    Step 1: Split all speakers -> (train+val) | test
    Step 2: Split (train+val)  -> train       | val

    Parameters
    ----------
    val_size  : fraction of ALL speakers going to val  (e.g. 0.15)
    test_size : fraction of ALL speakers going to test (e.g. 0.15)

    Returns
    -------
    train_df, val_df, test_df
    """
    if speaker_col not in df.columns:
        raise ValueError(f"Column '{speaker_col}' not found. Call add_speaker_column(df) first.")

    groups = df[speaker_col].values
    total  = len(df)

    # ── Step 1: carve off the test set ───────────────────────────────────────
    splitter1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    trainval_idx, test_idx = next(splitter1.split(df, groups=groups))

    df_trainval = df.iloc[trainval_idx].reset_index(drop=True)
    test_df     = df.iloc[test_idx].reset_index(drop=True)

    # ── Step 2: carve off the val set from the remaining pool ─────────────────
    if val_size > 0:
        # Rescale val_size: we want val_size% of TOTAL, but we're splitting (1-test_size)
        rescaled_val = val_size / (1.0 - test_size)
        groups2      = df_trainval[speaker_col].values
        splitter2    = GroupShuffleSplit(n_splits=1, test_size=rescaled_val, random_state=random_state)
        train_idx2, val_idx2 = next(splitter2.split(df_trainval, groups=groups2))
        train_df = df_trainval.iloc[train_idx2].reset_index(drop=True)
        val_df   = df_trainval.iloc[val_idx2].reset_index(drop=True)
    else:
        train_df = df_trainval
        val_df   = df.iloc[[]].reset_index(drop=True)  # empty

    # ── Sanity check ─────────────────────────────────────────────────────────
    train_spk = set(train_df[speaker_col])
    val_spk   = set(val_df[speaker_col])
    test_spk  = set(test_df[speaker_col])
    overlap   = (train_spk & val_spk) | (train_spk & test_spk) | (val_spk & test_spk)
    if overlap:
        raise RuntimeError(f"BUG: Speaker overlap detected: {overlap}")

    print(f"Speaker-aware 3-way split complete")
    print(f"   Train: {len(train_df):>5} files | {len(train_spk):>3} speakers")
    print(f"   Val:   {len(val_df):>5} files | {len(val_spk):>3} speakers  (used for EarlyStopping)")
    print(f"   Test:  {len(test_df):>5} files | {len(test_spk):>3} speakers  (LOCKED until final eval)")
    print(f"   Overlap between any two splits: none ✓")

    return train_df, val_df, test_df


if __name__ == "__main__":
    
    test_paths = [
        "RAVDESS/Actor_01/RAVDESS-01-Angry-1-dup0.wav",
        "RAVDESS/Actor_01/RAVDESS-01-Happy-1-dup1.wav",
        "RAVDESS/Actor_02/RAVDESS-02-Sad-1-dup0.wav",
        "RAVDESS/Actor_03/RAVDESS-03-Neutral-1-dup0.wav",
        "CREMAD/1001_DFA_ANG_XX.wav",
        "CREMAD/1001_DFA_HAP_XX.wav",
        "CREMAD/1002_MTI_SAD_XX.wav",
        "CREMAD/1003_IEO_NEU_XX.wav",
        "TESS/OAF_angry/OAF_angry_01.wav",
        "TESS/OAF_happy/OAF_happy_01.wav",
        "TESS/YAF_sad/YAF_sad_01.wav",
        "TESS/YAF_neutral/YAF_neutral_01.wav",
        "SAVEE/DC/DC_a01.wav",
        "SAVEE/DC/DC_h01.wav",
        "SAVEE/JE/JE_s01.wav",
        "SAVEE/KL/KL_n01.wav",
        "MELD/Ross_dia1_utt0.wav",
        "MELD/Rachel_dia1_utt1.wav",
        "MELD/Monica_dia2_utt0.wav",
        "MELD/Chandler_dia3_utt2.wav",
    ]
    emotions = ["angry", "happy", "sad", "neutral"] * 5

    df = pd.DataFrame({"Path": test_paths, "Emotions": emotions})
    df = add_speaker_column(df)

    print("Parsed speaker IDs:")
    print(df[["Path", "speaker_id"]].to_string(index=False))
    print()

    train_df, test_df = speaker_split(df, test_size=0.25, random_state=42)

    print(f"\nTrain speakers: {sorted(set(train_df['speaker_id']))}")
    print(f"Test  speakers: {sorted(set(test_df['speaker_id']))}")
