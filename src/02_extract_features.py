"""
Reads all local audio from raw_audio/, splits by speaker, applies multiprocessing
feature extraction and augmentation, caches .npy files so crashes are painless,
and produces the identical _ready_for_model.csv files your notebook expects.

Run this AFTER 01_sync_data.py completes.
"""
from __future__ import annotations

import os
import sys
# Ensure src/ siblings are importable regardless of working directory
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))

import time
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import librosa
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from config import RAW_DIR, CACHE_DIR, CSV_DIR, VAL_SIZE, TEST_SIZE, TARGET_SR, MONO
from speaker_splitter import add_speaker_column, speaker_three_way_split
from manifest_builder import build_manifest, assign_splits, validate_manifest, save_manifest
from audio_pipeline import extract_features_from_audio_array
from augmentation import get_augmentations, profile_summary

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

RAW_PATH = Path(RAW_DIR)


def _get_local_dataframe(raw_dir: Path) -> pd.DataFrame:
    """Replicates exactly what get_df_from_blob_dataset did, but locally."""
    file_paths = []
    emotions = []
    
    for wav_file in raw_dir.rglob("*.wav"):
        filename = wav_file.name
        parts = filename.split("-")
        
        # the emotion is the 3rd part of the ravdess string format
        if len(parts) >= 3:
            emotion = parts[2].lower()
        else:
            emotion = "unknown"
            
        file_paths.append(str(wav_file))
        emotions.append(emotion)
        
    return pd.DataFrame({
        "Path": file_paths,
        "Emotions": emotions
    })


def _process_single_file_worker(path: str, emotion: str, split: str, augment: bool) -> list[dict]:
    """
    Global top-level function that ProcessPoolExecutor can serialize.
    Reads local audio, checks the .npy cache, computes features, returns dicts.
    """
    path_obj = Path(path)
    base_name = path_obj.stem
    
    split_dir = CACHE_DIR / split
    split_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    def process_and_cache(aug_type: str, data: np.ndarray, sr: int):
        cache_path = split_dir / f"{base_name}_{aug_type}.npy"
        
        if cache_path.exists():
            # CACHE HIT: super fast load
            feat = np.load(cache_path)
        else:
            # CACHE MISS: compute heavy librosa features and save
            feat = extract_features_from_audio_array(data, sr)
            np.save(cache_path, feat)
            
        results.append({
            "features": feat.tolist(),  # flatten so dataframe packs properly
            "Labels": emotion,
            "source_original_path": path,
            "is_augmented": aug_type != "original",
            "augmentation_type": aug_type,
            "split": split
        })
    
    # We only load the original audio once per file!
    try:
        audio, sr = librosa.load(str(path_obj), sr=TARGET_SR, mono=MONO)
    except Exception as e:
        return [] # Return empty on load error

    # 1. Original 
    try:
        process_and_cache("original", audio, sr)
    except Exception as e:
        return [] # If original fails, skip entirely
        
    # 2. Augmentations (only for train)
    if augment:
        AUGMENTATIONS = get_augmentations()
        for aug_name, aug_fn in AUGMENTATIONS.items():
            try:
                aug_audio = aug_fn(audio, sr=sr)
                process_and_cache(aug_name, aug_audio, sr)
            except Exception:
                pass  # silently ignore failed augments (like pitch blowing up)
                
    return results


def build_feature_dataframe_parallel(df: pd.DataFrame, split_name: str, augment: bool) -> pd.DataFrame:
    """Distributes work across all CPUs."""
    total_files = len(df)
    print(f"\nStarting {split_name}: {total_files} files (augment={augment})")
    t0 = time.time()
    
    all_rows = []
    
    # ProcessPoolExecutor uses all available CPU cores perfectly!
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
        
    # Reconstruct the exact DataFrame layout your notebook expects
    # all_rows[i]["features"] is a list of N_FEATURES floats
    
    # Pluck the pure features into an array
    X = [r.pop("features") for r in all_rows]
    
    # Create the DataFrame
    features_df = pd.DataFrame(X)
    
    # Add metadata back on
    meta_df = pd.DataFrame(all_rows)
    final_df = pd.concat([features_df, meta_df], axis=1)
    
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
    
    # ── CPU-Crushing Multiprocessing ────────────────────────────
    print("\n4. Extracting Features (Multiprocessing)...")
    train_out = build_feature_dataframe_parallel(train_df, "train", augment=True)
    val_out   = build_feature_dataframe_parallel(val_df,   "val",   augment=False)
    test_out  = build_feature_dataframe_parallel(test_df,  "test",  augment=False)
    
    # ── Final Save ──────────────────────────────────────────────
    print("\n5. Saving final CSVs...")
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    
    train_out.to_csv(CSV_DIR / "train_features_ready_for_model.csv", index=False)
    val_out.to_csv(CSV_DIR / "val_features_ready_for_model.csv",     index=False)
    test_out.to_csv(CSV_DIR / "test_features_ready_for_model.csv",   index=False)
    
    # Save the manifest 
    save_manifest(manifest)
    print("\nPipeline Complete! You can now run your Notebook's Data Prep cell.")

if __name__ == "__main__":
    main()
