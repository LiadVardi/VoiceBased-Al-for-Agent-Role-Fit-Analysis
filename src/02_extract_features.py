"""
Reads all local audio from raw_audio/, splits by speaker, applies multiprocessing
feature extraction and augmentation, caches .npy files so crashes are painless,
and produces the identical _ready_for_model.csv files your notebook expects.

Run this AFTER 01_sync_data.py completes.
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))

import time
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import librosa
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from config import (RAW_DIR, CACHE_DIR, CSV_DIR, VAL_SIZE, TEST_SIZE,
                   TARGET_SR, TARGET_DURATION_SEC, MONO, MELD_EMOTION_MAP,
                   SLIDING_WINDOW_ENABLED, SLIDING_WINDOW_STRIDE_SEC)
from speaker_splitter import add_speaker_column, speaker_three_way_split
from manifest_builder import build_manifest, assign_splits, validate_manifest, save_manifest
from audio_pipeline import extract_spectrogram_from_audio_array
from augmentation import get_augmentations, profile_summary

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

RAW_PATH = Path(RAW_DIR)


def _get_meld_dataframe(raw_dir: Path) -> pd.DataFrame:
    meld_dir = raw_dir / "MELD"
    if not meld_dir.exists():
        return pd.DataFrame(columns=["Path", "Emotions"])

    csv_path = meld_dir / "meld_labels.csv"
    if not csv_path.exists():
        print(f"  [WARN] meld_labels.csv not found at {csv_path} — MELD will be skipped.")
        return pd.DataFrame(columns=["Path", "Emotions"])

    meld_df = pd.read_csv(csv_path)
    meld_df = meld_df[meld_df["Emotion"].isin(MELD_EMOTION_MAP)].copy()
    meld_df["Emotions"] = meld_df["Emotion"].map(MELD_EMOTION_MAP)

    meld_df["Path"] = meld_df.apply(
        lambda row: str(
            meld_dir
            / f"{str(row['Speaker']).strip().replace(' ', '_')}"
              f"_dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.wav"
        ),
        axis=1,
    )

    meld_df = meld_df[meld_df["Path"].apply(lambda p: Path(p).exists())].copy()
    print(f"  MELD: {len(meld_df)} utterances found locally")
    return meld_df[["Path", "Emotions"]].reset_index(drop=True)


def _get_voxonics_dataframe(raw_dir: Path) -> pd.DataFrame:
    voxonics_dir = raw_dir / "VOXONICS"
    if not voxonics_dir.exists():
        return pd.DataFrame(columns=["Path", "Emotions"])

    TARGET_EMOTIONS = {"angry", "happy", "neutral", "sad"}
    SUPPORTED_EXT   = {".wav", ".mp3", ".m4a", ".flac"}
    rows = []

    for emotion_dir in sorted(voxonics_dir.iterdir()):
        if not emotion_dir.is_dir():
            continue
        emotion = emotion_dir.name.lower()
        if emotion not in TARGET_EMOTIONS:
            continue
        for audio_file in emotion_dir.iterdir():
            if audio_file.suffix.lower() in SUPPORTED_EXT:
                rows.append({"Path": str(audio_file), "Emotions": emotion})

    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["Path", "Emotions"])
    print(f"  VOXONICS: {len(df)} clips found locally")
    return df.reset_index(drop=True)


def _extract_dataset_source(local_path: str) -> str:
    try:
        rel = Path(local_path).relative_to(RAW_PATH)
        return rel.parts[0] if rel.parts else "UNKNOWN"
    except ValueError:
        return "UNKNOWN"


def _get_local_dataframe(raw_dir: Path) -> pd.DataFrame:
    file_paths, emotions = [], []

    for wav_file in raw_dir.rglob("*.wav"):
        try:
            dataset_name = wav_file.relative_to(raw_dir).parts[0].upper()
        except (ValueError, IndexError):
            dataset_name = ""

        if dataset_name in ("MELD", "VOXONICS"):
            continue

        filename = wav_file.name
        parts    = filename.split("-")
        emotion  = parts[2].lower() if len(parts) >= 3 else "unknown"

        file_paths.append(str(wav_file))
        emotions.append(emotion)

    non_special_df = pd.DataFrame({"Path": file_paths, "Emotions": emotions})
    meld_df        = _get_meld_dataframe(raw_dir)
    voxonics_df    = _get_voxonics_dataframe(raw_dir)

    combined = pd.concat([non_special_df, meld_df, voxonics_df], ignore_index=True)
    return combined


def _process_single_file_worker(path: str, emotion: str, split: str, augment: bool) -> list[dict]:
    path_obj  = Path(path)
    base_name = path_obj.stem

    split_dir = CACHE_DIR / split
    split_dir.mkdir(parents=True, exist_ok=True)

    results = []

    def process_and_cache(aug_type: str, data: np.ndarray, sr: int):
        cache_path = split_dir / f"{base_name}_{aug_type}.npy"

        if not cache_path.exists():
            spec = extract_spectrogram_from_audio_array(data, sr)
            np.save(cache_path, spec)

        results.append({
            "npy_path": str(cache_path),
            "Labels": emotion,
            "source_original_path": path,
            "dataset_source": _extract_dataset_source(path),
            "is_augmented": aug_type != "original",
            "augmentation_type": aug_type,
            "split": split
        })

    try:
        audio, sr = librosa.load(str(path_obj), sr=TARGET_SR, mono=MONO)
    except Exception:
        return []

    duration_sec = len(audio) / sr

    if SLIDING_WINDOW_ENABLED and duration_sec > TARGET_DURATION_SEC:
        window_samples = int(TARGET_DURATION_SEC * sr)
        stride_samples = int(SLIDING_WINDOW_STRIDE_SEC * sr)
        start   = 0
        win_idx = 0
        while start + window_samples <= len(audio):
            window = audio[start : start + window_samples]
            try:
                process_and_cache(f"window_{win_idx}", window, sr)
            except Exception:
                pass
            start   += stride_samples
            win_idx += 1
    else:
        try:
            process_and_cache("original", audio, sr)
        except Exception:
            return []

    if augment:
        AUGMENTATIONS = get_augmentations()
        for aug_name, aug_fn in AUGMENTATIONS.items():
            try:
                aug_audio = aug_fn(audio, sr=sr)
                process_and_cache(aug_name, aug_audio, sr)
            except Exception:
                pass

    return results


def build_feature_dataframe_parallel(df: pd.DataFrame, split_name: str, augment: bool) -> pd.DataFrame:
    total_files = len(df)
    print(f"\nStarting {split_name}: {total_files} files (augment={augment})")
    t0 = time.time()

    all_rows = []

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for _, row in df.iterrows():
            futures.append(
                executor.submit(
                    _process_single_file_worker,
                    str(row["Path"]),
                    str(row["Emotions"]),
                    split_name,
                    augment
                )
            )

        for future in tqdm(as_completed(futures), total=total_files, desc=f"Processing {split_name}"):
            res = future.result()
            if res:
                all_rows.extend(res)

    elapsed_min = (time.time() - t0) / 60
    print(f"  Finished {split_name} in {elapsed_min:.1f} min ({len(all_rows)} total samples extracted)")

    if len(all_rows) == 0:
        return pd.DataFrame()

    final_df = pd.DataFrame(all_rows)
    return final_df


def main():
    profile_summary()

    if not RAW_PATH.exists() or len(list(RAW_PATH.iterdir())) == 0:
        print(f"Error: {RAW_PATH} is empty! Did you run 01_sync_data.py?")
        return

    print("\n1. Building local DataFrame...")
    data_path = _get_local_dataframe(RAW_PATH)
    print(f"  Found {len(data_path)} original files locally.")

    print("\n2. Speaker-aware 3-way split...")
    data_path = add_speaker_column(data_path)
    train_df, val_df, test_df = speaker_three_way_split(
        data_path, val_size=VAL_SIZE, test_size=TEST_SIZE, random_state=42
    )

    print("\n3. Building Manifest...")
    manifest = build_manifest(data_path)
    manifest = assign_splits(manifest, train_df, test_df, val_df=val_df)

    print("\n4. Extracting Features (Multiprocessing)...")
    train_out = build_feature_dataframe_parallel(train_df, "train", augment=True)
    val_out   = build_feature_dataframe_parallel(val_df,   "val",   augment=False)
    test_out  = build_feature_dataframe_parallel(test_df,  "test",  augment=False)

    print("\n5. Saving final CSVs...")
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    train_out.to_csv(CSV_DIR / "train_features_ready_for_model.csv", index=False)
    val_out.to_csv(CSV_DIR / "val_features_ready_for_model.csv",     index=False)
    test_out.to_csv(CSV_DIR / "test_features_ready_for_model.csv",   index=False)

    save_manifest(manifest)
    print("\nPipeline Complete! You can now run your Notebook's Data Prep cell.")


if __name__ == "__main__":
    main()
