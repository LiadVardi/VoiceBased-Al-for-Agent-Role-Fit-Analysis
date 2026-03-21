# Voice-Based AI for Agent Role Fit Analysis — Documentation

## What This Project Does

This project trains a **speech emotion recognition** model that predicts a speaker's emotion (e.g. angry, happy, sad, neutral) from a raw audio file.

Use case: analysing whether a call-centre agent's voice profile is a fit for their role.

**High-level flow:**

```
Raw audio (Azure Blob)
       ↓
Audio pipeline  (load → trim silence → crop/pad → extract features)
       ↓
Speaker-aware train / test split  (no speaker appears in both sets)
       ↓
Augmentation on train set only  (noise / stretch / shift / pitch)
       ↓
Dataset manifest  (trace every sample back to its source)
       ↓
CNN model training  (VoiceBasedAi.ipynb)
       ↓
Saved assets  (model_assets/: model + scaler + encoder)
       ↓
Inference  (voxonics_predictions.py)
```

---

## File Overview

| File | Role |
|------|------|
| `config.py` | Central configuration — all shared constants live here |
| `audio_pipeline.py` | **Single source of truth** for audio loading, preprocessing, and feature extraction |
| `preprocess.py` | Batch-converts `raw_audio/` → `clean_audio/` using the shared pipeline |
| `speaker_splitter.py` | Speaker-aware train/test split that prevents speaker leakage |
| `manifest_builder.py` | Builds and validates a CSV manifest tracking every sample and its augmentation lineage |
| `VoiceBasedAi.ipynb` | Main training notebook — orchestrates all of the above |
| `voxonics_predictions.py` | Inference script — loads saved assets and predicts emotion for one file |
| `model_assets/` | Saved CNN model (`.keras`), scaler (`.pickle`), and label encoder (`.pickle`) |
| `dataset_manifest.csv` | Output manifest — every sample with split label and augmentation traceability |

---

## `config.py` — Shared Configuration

All tuneable values are declared here and imported everywhere else. **Do not hardcode any of these values in other files.**

| Constant | Value | Purpose |
|----------|-------|---------|
| `TARGET_SR` | 22050 | Resample rate for all audio (librosa default) |
| `MONO` | `True` | Convert all audio to mono |
| `NORMALIZE_AUDIO` | `False` | Peak normalisation **must stay False** to match the training distribution |
| `TRIM_SILENCE` | `True` | Remove leading/trailing silence |
| `TRIM_TOP_DB` | 20 | dB threshold for silence detection |
| `TARGET_DURATION_SEC` | 2.5 | Fixed audio length after crop/pad |
| `MIN_DURATION_SEC` | 0.5 | Reject files shorter than this after trimming |
| `CROP_MODE` | `"start"` | Take the first 2.5 s of longer files |
| `PAD_MODE` | `"constant"` | Zero-pad shorter files on the right |
| `N_MFCC` | 40 | Number of MFCC coefficients |
| `N_FEATURES` | 252 | Total feature vector size (see below) |
| `EPSILON` | 1e-10 | Division-by-zero guard |
| `MODEL_PATH` | `model_assets/final_emotion_model.keras` | Saved Keras model |
| `SCALER_PATH` | `model_assets/scaler.pickle` | Saved `StandardScaler` |
| `ENCODER_PATH` | `model_assets/encoder.pickle` | Saved `OneHotEncoder` |
| `RAW_DIR` | `"raw_audio"` | Raw input directory |
| `CLEAN_DIR` | `"clean_audio"` | Preprocessed output directory |

> **Warning — `NORMALIZE_AUDIO`:** If this flag is flipped to `True`, peak normalisation shifts the MFCC distribution, causing inference to disagree with training. It must be the same value in both runs and the current setting (`False`) must not be changed without retraining.

---

## `audio_pipeline.py` — The Audio Pipeline

This is the most important file. Both training and inference call functions from here. **No audio logic should be reimplemented anywhere else.**

### Pipeline steps (order is fixed)

```
1. Load        librosa.load() → TARGET_SR, mono
2. Normalize   optional peak normalisation (NORMALIZE_AUDIO must match training)
3. Trim        librosa.effects.trim() removes leading/trailing silence
4. Crop / Pad  force to TARGET_DURATION_SEC (2.5 s)
5. Features    extract 252-dimensional feature vector
```

### Feature vector layout (252 dimensions)

| Indices | Feature | Count |
|---------|---------|-------|
| 0–39 | MFCC mean (per coefficient) | 40 |
| 40–79 | MFCC std | 40 |
| 80–119 | Δ-MFCC mean | 40 |
| 120–159 | Δ-MFCC std | 40 |
| 160–199 | ΔΔ-MFCC mean | 40 |
| 200–239 | ΔΔ-MFCC std | 40 |
| 240–241 | RMS energy (mean, std) | 2 |
| 242–243 | Zero-crossing rate (mean, std) | 2 |
| 244–245 | Spectral centroid (mean, std) | 2 |
| 246–247 | Spectral bandwidth (mean, std) | 2 |
| 248–249 | Spectral rolloff (mean, std) | 2 |
| 250–251 | F0 / pitch, voiced frames only (mean, std) | 2 |
| **Total** | | **252** |

### Key functions

| Function | Used by | Description |
|----------|---------|-------------|
| `load_audio(file_path)` | Anywhere | Load from disk → `(np.ndarray, sr)` |
| `load_audio_from_bytes(bytes)` | Notebook (Azure) | Load from in-memory bytes → `(np.ndarray, sr)` |
| `preprocess_audio(audio, sr)` | Internal | Steps 2–4 above |
| `extract_features(audio, sr)` | Internal | Step 5 above |
| `extract_features_from_audio_array(audio, sr)` | **Notebook training loop** | preprocess → extract features |
| `extract_features_from_file(file_path)` | **`voxonics_predictions.py`** | load → preprocess → extract features |
| `get_feature_names()` | Debugging / DataFrames | Returns an ordered list of 252 feature name strings |

### Error types

| Exception | When raised |
|-----------|-------------|
| `AudioEmptyError` | Audio is completely silent |
| `AudioTooShortError` | Duration after trimming is below `MIN_DURATION_SEC` |

---

## `preprocess.py` — Batch Preprocessing

Converts all `.wav` files from `raw_audio/` to `clean_audio/` in one pass.

- Preserves the original subfolder structure inside `clean_audio/`.
- Calls `preprocess_audio_file()` (from `audio_pipeline.py`) for every file.
- Skips and logs files that raise `AudioProcessingError` (empty or too short).
- Prints a summary report at the end (total, processed, rejected by reason).

**Run it directly:**
```bash
python preprocess.py
```

Input directory and output directory are read from `config.py` (`RAW_DIR`, `CLEAN_DIR`).

---

## `speaker_splitter.py` — Speaker-Aware Train/Test Split

Prevents the model from seeing speakers during testing that it was trained on.

### How it works

1. **Parse speaker ID** from the blob path using a per-dataset rule:

   | Dataset | Speaker ID format | Parsed from |
   |---------|-------------------|-------------|
   | RAVDESS | `RAVDESS_Actor_01` | subfolder name or filename prefix |
   | CREMAD | `CREMAD_1001` | first `_`-delimited token in filename |
   | TESS | `TESS_OAF` | speaker prefix in the subfolder name |
   | SAVEE | `SAVEE_DC` | subfolder name or first 2 chars of filename |

2. **`add_speaker_column(df)`** adds a `speaker_id` column to a DataFrame.

3. **`speaker_split(df, test_size=0.2)`** uses `sklearn.GroupShuffleSplit` on the `speaker_id` column:
   - Whole speakers go to either train or test, never both.
   - After the split, the function **asserts zero overlap** and raises `RuntimeError` if any speaker appears in both sets.

### Key constraint

If you add a new dataset or change how filenames are structured, you must add a new parser to `_PARSERS` and test it by running `python speaker_splitter.py`.

---

## `manifest_builder.py` — Dataset Manifest

Builds `dataset_manifest.csv` — a complete, traceable record of every sample.

### Manifest columns

| Column | Description |
|--------|-------------|
| `filepath` | Blob storage path of the audio file |
| `dataset_name` | RAVDESS, CREMAD, TESS, or SAVEE |
| `speaker_id` | Globally unique speaker key |
| `emotion` | Lowercase emotion label |
| `emotion_intensity` | normal / strong / low / medium / high / unknown |
| `is_augmented` | `False` for originals, `True` for augmented variants |
| `augmentation_type` | original / noise / stretch / shift / pitch |
| `source_original_path` | Blob path of the original file this row derives from |
| `split` | train / test / unassigned |

### Build sequence (mirrors the notebook)

```python
# 1. Build from original file list
manifest = build_manifest(data_path)

# 2. Assign split labels after speaker_split
manifest = assign_splits(manifest, train_df, test_df)

# 3. Append augmented rows after the augmentation loop
manifest = add_augmented_rows(manifest, aug_records)

# 4. Validate and save
validate_manifest(manifest)
save_manifest(manifest)
```

### `validate_manifest()` checks

- No missing values in any critical column.
- Every augmented row points to a valid original in the manifest.
- No speaker appears in both train and test.

### Augmentation record format

```python
{
    "source_original_path": "RAVDESS/Actor_01/RAVDESS-01-Happy-1.wav",
    "augmentation_type":    "noise",   # noise | stretch | shift | pitch
}
```

---

## `VoiceBasedAi.ipynb` — Training Notebook

The main notebook. Runs end to end to produce trained model assets.

### Notebook sections

| Section | What it does |
|---------|-------------|
| Importing | Imports all libraries and project modules |
| Integration with Azure | Connects to Azure Blob Storage, lists `.wav` files from RAVDESS / CREMAD / TESS / SAVEE, saves `data_path.csv` |
| Splitting the data | Calls `add_speaker_column` + `speaker_split` to partition files by speaker |
| Data Augmentation | Defines four augmentation functions: `noise`, `stretch`, `shift`, `pitch` |
| Build Feature Extraction | `build_feature_dataframe()` loads each file from Azure, calls `extract_features_from_audio_array()`, and applies augmentations to the train set only |
| Speaker Split & Manifest Creation | Runs the full manifest build sequence |
| Data Preparation | Fits `StandardScaler` and `OneHotEncoder` on train data; applies (not re-fits) both to test data |
| CNN Architecture | 1D-CNN with 5 Conv1D blocks, BatchNorm, Dropout, final Dense + softmax |
| CNN Model (training) | `model.fit()` up to 50 epochs with EarlyStopping, ModelCheckpoint, ReduceLROnPlateau |
| Evaluation | Confusion matrix and classification report on the test set |
| Saving the model | Saves `model_assets/final_emotion_model.keras`, `scaler.pickle`, `encoder.pickle` |

### CNN architecture summary

```
Conv1D(512) → BN → MaxPool
Conv1D(512) → BN → MaxPool → Dropout(0.2)
Conv1D(256) → BN → MaxPool
Conv1D(256) → BN → MaxPool → Dropout(0.2)
Conv1D(128) → BN → MaxPool → Dropout(0.2)
Flatten → Dense(512) → BN → Dropout(0.2)
Dense(n_classes, softmax)
```

Input shape: `(252, 1)` — the 252-dimensional feature vector treated as a 1D signal.

---

## `voxonics_predictions.py` — Inference

Loads saved model assets and predicts emotion for a single audio file.

```
FILE_TO_TEST  →  extract_features_from_file()
              →  scaler.transform()
              →  np.expand_dims(..., axis=2)
              →  model.predict()
              →  encoder.categories_[0]
              →  print confidence per emotion
```

Key design points:
- Uses `extract_features_from_file()` from `audio_pipeline.py` — **the exact same pipeline as training**.
- All audio constants (`TARGET_SR`, `TRIM_SILENCE`, `CROP_MODE`, etc.) come from `config.py`.
- The scaler and encoder must be the ones saved during the last training run.

**To predict on a different file:** change `FILE_TO_TEST` at the top of the script.

---

## How Training and Inference Stay Aligned

This is the most important invariant in the project.

```
Training (notebook)                 Inference (voxonics_predictions.py)
─────────────────────────────────   ──────────────────────────────────────
extract_features_from_audio_array   extract_features_from_file
          ↓ (both call)                        ↓ (both call)
     preprocess_audio                     preprocess_audio
     extract_features                     extract_features
          ↑                                         ↑
          └──────── audio_pipeline.py ──────────────┘
                    config.py (shared settings)
```

Both paths end up with the same 252-feature vector, scaled by the same scaler, decoded by the same encoder.

**Things that break alignment:**
- Changing `NORMALIZE_AUDIO` after training without retraining.
- Changing `N_MFCC`, `TARGET_DURATION_SEC`, or any preprocessing step without regenerating the scaler and model.
- Using a different scaler or encoder than the one saved in `model_assets/`.
- Adding or removing features (changes `N_FEATURES`).

---

## Important Risks and Assumptions

### 1. Feature size must match saved assets
`N_FEATURES = 252`. The CNN input layer, scaler, and encoder are all frozen to this size at training time. If you change `N_MFCC` or add/remove features, you must retrain from scratch and regenerate all three assets.

### 2. `NORMALIZE_AUDIO` must stay `False`
The model was trained without peak normalisation. Enabling it shifts the MFCC distribution and breaks inference. If you ever want to enable it, you must retrain with it enabled, and update this flag in `config.py` consistently.

### 3. Augmentation is train-only
The `build_feature_dataframe()` call passes `augment=True` for train and `augment=False` for test. Augmented rows inherit their split label from the original. Augmenting test data would inflate test scores and defeat the purpose of evaluation.

### 4. Speaker leakage is forbidden
`speaker_split()` raises a `RuntimeError` at runtime if any speaker appears in both train and test. Never bypass this check. Do not use a random file-level split (e.g. `train_test_split` without groups).

### 5. Augmented rows point to the original blob path
`manifest_builder.add_augmented_rows()` sets `filepath` to the **original** blob path for augmented rows. The augmentation is applied in memory — no separate augmented files are stored. `is_augmented=True` and `augmentation_type` distinguish them from the originals.

### 6. Azure connection string is required
`AZURE_STORAGE_CONNECTION_STRING` must be set in a `.env` file before running the notebook. Without it the blob listing step raises `ValueError`.

### 7. Datasets must follow expected path conventions
Speaker and emotion parsing in `speaker_splitter.py` and `manifest_builder.py` depends on blob paths matching the format: `DATASET/subfolder/filename.wav`. An unknown prefix issues a warning and falls back to `UNKNOWN_<filename>`.

### 8. `voxonics_predictions.py` has a hardcoded file path
`FILE_TO_TEST` is a hardcoded Windows path. It must be updated before running the inference script on a different machine or file.

---

## Documentation Files

| File | What it covers | When to read |
|------|---------------|--------------|
| `PROJECT_GUIDE.md` | Project rules, architecture, and change procedures | Before every change |
| `DOCUMENTATION.md` (this file) | Detailed explanation of each file and system behaviour | For onboarding or when understanding a specific component |
