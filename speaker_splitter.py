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


# 2. Dispatcher

_PARSERS = {
    "RAVDESS": _speaker_ravdess,
    "CREMAD":  _speaker_cremad,
    "TESS":    _speaker_tess,
    "SAVEE":   _speaker_savee,
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

    if speaker_col not in df.columns:
        raise ValueError(
            f"Column '{speaker_col}' not found. "
            f"Call add_speaker_column(df) first."
        )

    groups = df[speaker_col].values

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )

    train_idx, test_idx = next(splitter.split(df, groups=groups))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df  = df.iloc[test_idx].reset_index(drop=True)

    train_speakers = set(train_df[speaker_col])
    test_speakers  = set(test_df[speaker_col])
    overlap = train_speakers & test_speakers

    if overlap:
        raise RuntimeError(
            f"BUG: {len(overlap)} speaker(s) appear in both train and test! "
            f"Overlapping: {overlap}"
        )

    print(f" Speaker-aware split complete")
    print(f"   Train: {len(train_df):>5} files | {len(train_speakers):>3} unique speakers")
    print(f"   Test:  {len(test_df):>5} files | {len(test_speakers):>3} unique speakers")
    print(f"   Overlap: none ✓")

    return train_df, test_df


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
    ]
    emotions = ["angry", "happy", "sad", "neutral"] * 4

    df = pd.DataFrame({"Path": test_paths, "Emotions": emotions})
    df = add_speaker_column(df)

    print("Parsed speaker IDs:")
    print(df[["Path", "speaker_id"]].to_string(index=False))
    print()

    train_df, test_df = speaker_split(df, test_size=0.25, random_state=42)

    print(f"\nTrain speakers: {sorted(set(train_df['speaker_id']))}")
    print(f"Test  speakers: {sorted(set(test_df['speaker_id']))}")
