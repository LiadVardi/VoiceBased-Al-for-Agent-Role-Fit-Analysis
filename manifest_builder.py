"""
manifest_builder.py
===================
Builds a fully-traceable dataset manifest for the Speech Emotion Recognition project.

Every row in the manifest represents one sample that enters (or could enter) the model:
  - original recordings
  - augmented variants derived from those recordings

Columns produced
----------------
filepath              : blob storage path of the source audio file
dataset_name          : RAVDESS | CREMAD | TESS | SAVEE
speaker_id            : globally unique speaker key (from speaker_splitter.py)
emotion               : lowercase emotion label (angry, happy, sad, neutral)
emotion_intensity     : normal | strong | low | medium | high | unknown
is_augmented          : False for originals, True for all augmented variants
augmentation_type     : original | noise | stretch | shift | pitch | <other>
source_original_path  : filepath of the original (same as filepath for originals)
split                 : train | test | unassigned

Usage in the notebook
---------------------
    from manifest_builder import build_manifest, save_manifest

    df = pd.read_csv("data_path.csv")        # Emotions + Path columns
    manifest = build_manifest(df)            # originals only, split=unassigned

    # After speaker_split:
    manifest = assign_splits(manifest, train_df, test_df)

    # After augmentation loop:
    manifest = add_augmented_rows(manifest, augmentation_records)

    save_manifest(manifest)                  # -> dataset_manifest.csv
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from speaker_splitter import extract_speaker_id

MANIFEST_PATH = Path("dataset_manifest.csv")

# ── Augmentation type labels (must match what the notebook uses) ─────────────
AUG_ORIGINAL = "original"
AUG_NOISE    = "noise"
AUG_STRETCH  = "stretch"
AUG_SHIFT    = "shift"
AUG_PITCH    = "pitch"

# Ordered list that maps augmentation index → type name
# index 0 = original, 1..N = augmented variants
AUGMENTATION_ORDER = [AUG_ORIGINAL, AUG_NOISE, AUG_STRETCH, AUG_SHIFT, AUG_PITCH]


# ---------------------------------------------------------------------------
# 1. Metadata parsers per dataset
# ---------------------------------------------------------------------------

def _parse_ravdess(parts: list[str], blob_path: str) -> dict:
    """
    RAVDESS/Actor_01/RAVDESS-01-Angry-1-dup0.wav
    Filename parts split by '-':  RAVDESS | 01 | Angry | 1 | dup0.wav
    Intensity: 1 = normal, 2 = strong
    """
    filename = parts[-1]
    fn_parts = filename.split("-")
    emotion = fn_parts[2].lower() if len(fn_parts) > 2 else "unknown"
    raw_intensity = fn_parts[3] if len(fn_parts) > 3 else "0"
    intensity_map = {"1": "normal", "2": "strong"}
    intensity = intensity_map.get(raw_intensity, "unknown")
    return {"emotion": emotion, "emotion_intensity": intensity}


def _parse_cremad(parts: list[str], blob_path: str) -> dict:
    """
    CREMAD/1001_DFA_ANG_XX.wav
    Filename parts split by '_':  1001 | DFA | ANG | XX.wav
    Intensity codes: LO=low, MD=medium, HI=high, XX=unknown
    """
    filename = parts[-1].replace(".wav", "")
    fn_parts = filename.split("_")
    emotion_code_map = {
        "ANG": "angry", "DIS": "disgust", "FEA": "fear",
        "HAP": "happy", "NEU": "neutral", "SAD": "sad",
    }
    intensity_code_map = {"LO": "low", "MD": "medium", "HI": "high", "XX": "unknown"}

    emotion   = emotion_code_map.get(fn_parts[2].upper(), "unknown") if len(fn_parts) > 2 else "unknown"
    intensity = intensity_code_map.get(fn_parts[3].upper(), "unknown") if len(fn_parts) > 3 else "unknown"
    return {"emotion": emotion, "emotion_intensity": intensity}


def _parse_tess(parts: list[str], blob_path: str) -> dict:
    """
    TESS/OAF_angry/OAF_angry_happy.wav
    Emotion is in the folder name (second '_'-separated part).
    TESS has no intensity information.
    """
    if len(parts) >= 3:
        folder = parts[1]
        folder_parts = folder.split("_")
        emotion = folder_parts[1].lower() if len(folder_parts) > 1 else "unknown"
    else:
        emotion = "unknown"
    return {"emotion": emotion, "emotion_intensity": "unknown"}


def _parse_savee(parts: list[str], blob_path: str) -> dict:
    """
    SAVEE/DC/DC_a01.wav
    Emotion code is the letter after the actor prefix in the filename.
    a=angry, d=disgust, f=fear, h=happy, n=neutral, sa=sad, su=surprise
    SAVEE has no intensity information.
    """
    filename = parts[-1].lower().replace(".wav", "")
    # Remove actor prefix (2 chars) and underscore: 'dc_a01' → 'a01'
    code_part = "_".join(filename.split("_")[1:])
    emotion_map = {
        "a":  "angry", "d": "disgust", "f": "fear",
        "h":  "happy", "n": "neutral", "sa": "sad", "su": "surprise",
    }
    # Match on leading letters
    emotion = "unknown"
    for code, label in sorted(emotion_map.items(), key=lambda x: -len(x[0])):
        if code_part.startswith(code):
            emotion = label
            break
    return {"emotion": emotion, "emotion_intensity": "unknown"}


_PARSERS = {
    "RAVDESS": _parse_ravdess,
    "CREMAD":  _parse_cremad,
    "TESS":    _parse_tess,
    "SAVEE":   _parse_savee,
}


def parse_file_metadata(blob_path: str) -> dict:
    """
    Extract all available metadata from a single blob path.

    Returns a dict with keys:
        filepath, dataset_name, speaker_id, emotion, emotion_intensity
    """
    path = blob_path.replace("\\", "/")
    parts = path.split("/")
    dataset = parts[0].upper()

    parser = _PARSERS.get(dataset)
    if parser is None:
        warnings.warn(f"Unknown dataset '{dataset}' in path: {blob_path}")
        parsed = {"emotion": "unknown", "emotion_intensity": "unknown"}
    else:
        parsed = parser(parts, blob_path)

    return {
        "filepath":         blob_path,
        "dataset_name":     dataset,
        "speaker_id":       extract_speaker_id(blob_path),
        "emotion":          parsed["emotion"],
        "emotion_intensity": parsed["emotion_intensity"],
    }


# ---------------------------------------------------------------------------
# 2. Build the manifest from the initial data_path DataFrame
# ---------------------------------------------------------------------------

def build_manifest(df: pd.DataFrame, path_col: str = "Path") -> pd.DataFrame:
    """
    Build the manifest DataFrame from the raw file list (original files only).

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe produced by the Azure blob listing step.
        Must contain a column with file paths (named `path_col`).
    path_col : str
        Column name that holds blob paths.

    Returns
    -------
    pd.DataFrame
        Manifest with one row per original file.
        split = 'unassigned' until assign_splits() is called.
    """
    rows = []
    for blob_path in df[path_col]:
        meta = parse_file_metadata(blob_path)
        meta.update({
            "is_augmented":        False,
            "augmentation_type":   AUG_ORIGINAL,
            "source_original_path": blob_path,
            "split":               "unassigned",
        })
        rows.append(meta)

    manifest = pd.DataFrame(rows, columns=[
        "filepath", "dataset_name", "speaker_id",
        "emotion", "emotion_intensity",
        "is_augmented", "augmentation_type",
        "source_original_path", "split",
    ])
    print(f"Manifest built: {len(manifest)} original files")
    print(manifest["dataset_name"].value_counts().to_string())
    return manifest


# ---------------------------------------------------------------------------
# 3. Assign train / test split labels
# ---------------------------------------------------------------------------

def assign_splits(
    manifest: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    path_col: str = "Path",
) -> pd.DataFrame:
    """
    Label each row in the manifest as 'train' or 'test' based on the split.

    Parameters
    ----------
    manifest : pd.DataFrame
        Output of build_manifest().
    train_df, test_df : pd.DataFrame
        DataFrames of original files after speaker_split().
    path_col : str
        Column in train_df/test_df that holds blob paths.

    Returns
    -------
    pd.DataFrame
        Manifest with 'split' column populated.
    """
    manifest = manifest.copy()
    train_paths = set(train_df[path_col])
    test_paths  = set(test_df[path_col])

    def _label(fp):
        if fp in train_paths:
            return "train"
        if fp in test_paths:
            return "test"
        return "unassigned"

    manifest["split"] = manifest["filepath"].apply(_label)

    train_count = (manifest["split"] == "train").sum()
    test_count  = (manifest["split"] == "test").sum()
    print(f"Splits assigned — train: {train_count}, test: {test_count}")
    return manifest


# ---------------------------------------------------------------------------
# 4. Add augmented rows
# ---------------------------------------------------------------------------

def add_augmented_rows(
    manifest: pd.DataFrame,
    augmentation_records: list[dict],
) -> pd.DataFrame:
    """
    Append rows for every augmented sample to the manifest.

    Parameters
    ----------
    manifest : pd.DataFrame
        Manifest after assign_splits().
    augmentation_records : list[dict]
        Each dict must contain:
            source_original_path  – blob path of the original file
            augmentation_type     – one of: noise, stretch, shift, pitch, ...

    Returns
    -------
    pd.DataFrame
        Extended manifest, original rows + one row per augmented variant.

    Example
    -------
    Inside your augmentation loop, collect records like this:

        records = []
        for path, emotion in zip(train_df["Path"], train_df["Emotions"]):
            for aug_type in ["noise", "stretch", "shift", "pitch"]:
                records.append({
                    "source_original_path": path,
                    "augmentation_type": aug_type,
                })
        manifest = add_augmented_rows(manifest, records)
    """
    if not augmentation_records:
        return manifest

    # Build a lookup: original path → its manifest row metadata
    orig_lookup = manifest.set_index("filepath").to_dict("index")

    aug_rows = []
    for rec in augmentation_records:
        src = rec["source_original_path"]
        orig = orig_lookup.get(src)
        if orig is None:
            warnings.warn(f"Source path not found in manifest: {src}")
            continue

        aug_rows.append({
            "filepath":            src,           # same blob path (audio read from here)
            "dataset_name":        orig["dataset_name"],
            "speaker_id":          orig["speaker_id"],
            "emotion":             orig["emotion"],
            "emotion_intensity":   orig["emotion_intensity"],
            "is_augmented":        True,
            "augmentation_type":   rec["augmentation_type"],
            "source_original_path": src,
            "split":               orig["split"],
        })

    aug_df = pd.DataFrame(aug_rows)
    extended = pd.concat([manifest, aug_df], ignore_index=True)
    print(f"Augmented rows added: {len(aug_rows)} → total manifest rows: {len(extended)}")
    return extended


# ---------------------------------------------------------------------------
# 5. Save / load
# ---------------------------------------------------------------------------

def save_manifest(manifest: pd.DataFrame, path: Path = MANIFEST_PATH) -> None:
    """Save the manifest to CSV."""
    manifest.to_csv(path, index=False)
    print(f"Manifest saved → {path}  ({len(manifest)} rows)")


def load_manifest(path: Path = MANIFEST_PATH) -> pd.DataFrame:
    """Load the manifest from CSV."""
    df = pd.read_csv(path)
    print(f"Manifest loaded from {path}  ({len(df)} rows)")
    return df


# ---------------------------------------------------------------------------
# 6. Sanity checker
# ---------------------------------------------------------------------------

def validate_manifest(manifest: pd.DataFrame) -> None:
    """
    Run a suite of consistency checks on the manifest.
    Prints a summary and raises AssertionError on any failure.
    """
    errors = []

    # No missing values in critical columns
    for col in ["filepath", "dataset_name", "speaker_id", "emotion", "split",
                "is_augmented", "augmentation_type", "source_original_path"]:
        nulls = manifest[col].isna().sum()
        if nulls:
            errors.append(f"  [{col}] has {nulls} missing values")

    # Every augmented row links to a valid source
    orig_paths = set(manifest.loc[~manifest["is_augmented"], "filepath"])
    aug_sources = manifest.loc[manifest["is_augmented"], "source_original_path"]
    bad_sources = set(aug_sources) - orig_paths
    if bad_sources:
        errors.append(f"  {len(bad_sources)} augmented rows point to unknown sources: {bad_sources}")

    # No speaker appears in both train and test
    train_speakers = set(manifest.loc[manifest["split"] == "train", "speaker_id"])
    test_speakers  = set(manifest.loc[manifest["split"] == "test",  "speaker_id"])
    overlap = train_speakers & test_speakers
    if overlap:
        errors.append(f"  Speaker overlap between train/test: {overlap}")

    if errors:
        msg = "Manifest validation FAILED:\n" + "\n".join(errors)
        raise AssertionError(msg)

    total      = len(manifest)
    originals  = (~manifest["is_augmented"]).sum()
    augmented  = manifest["is_augmented"].sum()
    train_rows = (manifest["split"] == "train").sum()
    test_rows  = (manifest["split"] == "test").sum()

    print("Manifest validation passed")
    print(f"   Total rows:  {total:>6}")
    print(f"   Originals:   {originals:>6}")
    print(f"   Augmented:   {augmented:>6}")
    print(f"   Train split: {train_rows:>6}")
    print(f"   Test split:  {test_rows:>6}")
    print(f"   Speakers in train: {len(train_speakers)}, in test: {len(test_speakers)}, overlap: 0 ✓")
