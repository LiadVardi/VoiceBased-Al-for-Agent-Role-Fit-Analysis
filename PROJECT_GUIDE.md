# Project Instructions

## General Rules

- Do NOT scan the whole project by default.
- Only read the files directly relevant to the current task.
- Prefer reading `PROJECT_GUIDE.md` first, then the specific code files related to the task.
- Before changing preprocessing, feature extraction, or inference behavior, always read:
  - `audio_pipeline.py`
  - `config.py`
- Before changing dataset preprocessing flow, always read:
  - `preprocess.py`
  - `audio_pipeline.py`
- Before changing prediction or inference behavior, always read:
  - `voxonics_predictions.py`
  - `audio_pipeline.py`
  - `config.py`
- Before changing speaker-based train/test split behavior, always read:
  - `speaker_splitter.py`
- Before changing manifest, metadata parsing, augmentation tracking, or dataset traceability, always read:
  - `manifest_builder.py`
  - `speaker_splitter.py`
- Keep training and inference aligned.
- Do NOT duplicate audio preprocessing or feature extraction logic in multiple places.
- `audio_pipeline.py` is the single source of truth for audio loading, preprocessing, and feature extraction.
- Always ask before changing `PROJECT_GUIDE.md`.

---

## Documentation Management

- When code behavior changes, update the relevant documentation.
- Keep documentation concise and practical.
- Documentation should explain:
  - what the file or module does
  - its main inputs and outputs
  - important dependencies
  - assumptions that must stay aligned with other files
- If new documentation files are created, add them to the **Documentation Files** section below.
- Keep the **Documentation Files** section up to date.

### Tiered Documentation Structure

Recommended navigation:

- `PROJECT_GUIDE.md` → project-level rules and navigation
- optional documentation files → focused explanations for important areas
- code files → implementation details

If extra documentation files are added later, they should be listed in this guide.

---

## Project Overview

This project is an audio-based emotion recognition project.

Main responsibilities in the project:
- preprocess raw audio files
- apply one shared audio pipeline
- extract fixed-size audio features
- keep training and inference aligned
- split data by speaker to avoid leakage
- build and maintain a dataset manifest with metadata and augmentation traceability
- run emotion prediction using saved model assets

---

## Architecture

```text
VoiceBasedAi.ipynb         - Main notebook for training / experimentation
config.py                  - Central configuration for preprocessing, feature extraction, and model asset paths
audio_pipeline.py          - Single source of truth for audio loading, preprocessing, crop/pad, and feature extraction
preprocess.py              - Batch preprocessing script for raw audio files
speaker_splitter.py        - Speaker-aware split logic to prevent train/test speaker leakage
manifest_builder.py        - Builds dataset manifest, parses metadata, tracks augmentation, and assigns splits
voxonics_predictions.py    - Inference script that loads model assets and predicts emotion for one audio file
model_assets/              - Saved trained model, scaler, and encoder files
raw_audio/                 - Raw input audio
clean_audio/               - Preprocessed audio output
dataset_manifest.csv       - Manifest containing metadata, split labels, and augmentation traceability
```

---

## Important File Responsibilities

### `config.py`

Central place for shared project settings, including:
- sample rate
- mono or stereo handling
- normalization behavior
- silence trimming
- target duration
- crop or pad behavior
- MFCC count
- total feature count
- model, scaler, and encoder paths
- raw and clean audio directories

Do not hardcode these values in multiple places.

### `audio_pipeline.py`

This is the most important file in the project.

It is the single source of truth for:
- loading audio from file or bytes
- optional normalization
- silence trimming
- crop or pad to fixed duration
- feature extraction
- shared preprocessing behavior used by training and inference

Training and inference must both depend on this file instead of reimplementing logic elsewhere.

### `preprocess.py`

Used to preprocess raw audio files in batch form.

Its job is to:
- read files from the raw audio area
- preprocess them using the shared audio pipeline
- save processed output to the clean audio area

It should stay aligned with `audio_pipeline.py` and `config.py`.

### `speaker_splitter.py`

Responsible for speaker-aware splitting.

Its purpose is to ensure:
- train and test data do not share the same speaker
- speaker IDs are extracted consistently
- dataset splitting avoids speaker leakage

### `manifest_builder.py`

Responsible for dataset traceability and metadata organization.

Its job includes:
- parsing metadata from dataset-specific file paths
- extracting speaker IDs
- building the dataset manifest
- assigning train or test split labels
- tracking augmented samples
- preserving links between augmented data and original source files

### `voxonics_predictions.py`

Responsible for inference.

Its job is to:
- load the saved model
- load the scaler and encoder
- extract features using the shared audio pipeline
- scale features correctly
- run prediction
- print emotion confidence results

It must stay compatible with:
- `audio_pipeline.py`
- `config.py`
- saved assets in `model_assets/`

### `VoiceBasedAi.ipynb`

Main notebook for training and experimentation.

It should:
- use the same shared preprocessing and feature extraction logic as inference
- avoid reimplementing audio logic that already exists in `audio_pipeline.py`
- stay aligned with `config.py`

---

## Documentation Files

### `DOCUMENTATION.md`

Detailed explanation of every main file, the audio pipeline steps and feature layout, how training and inference stay aligned, how speaker splitting works, how the manifest is built, and a full list of important risks and assumptions.  
**Read when:** onboarding, or when you need to understand how a specific component works before making a change.

---

## Project-Specific Rules

### 1. Training and inference must stay aligned

- The same preprocessing and feature extraction logic must be used in both training and inference.
- Do not implement one pipeline in the notebook and another in the inference script.
- Shared behavior must come from `audio_pipeline.py`.

### 2. `audio_pipeline.py` is the single source of truth

- Do not duplicate audio logic in other files.
- If preprocessing or feature extraction changes, update it in `audio_pipeline.py` and keep dependent files aligned.

### 3. `config.py` controls shared behavior

- Shared audio settings should be controlled through `config.py`.
- Do not scatter sample rate, trimming, duration, MFCC count, or feature size values across multiple files.

### 4. Speaker leakage is forbidden

- Train or test split must remain speaker-aware.
- Any change to split logic or speaker parsing must preserve zero speaker overlap between train and test.

### 5. Dataset traceability must be preserved

- The manifest must keep track of:
  - original files
  - augmented files
  - source-original relationships
  - split labels
- Augmented samples must remain traceable back to the original sample.

### 6. Do not change behavior silently

If changing:
- normalization
- silence trimming
- crop or pad behavior
- target duration
- feature extraction
- speaker split logic
- manifest format

then also:
- document the change
- note the expected behavioral impact
- verify whether existing model assets are still compatible

### 7. Feature changes may require retraining

- If feature layout, feature count, MFCC count, or preprocessing behavior changes, existing scaler, model, or encoder assets may no longer match.
- Check compatibility before using old assets.

---

## Known Tasks

### Update preprocessing behavior

Relevant files:
- `audio_pipeline.py`
- `config.py`
- `preprocess.py`

Steps:
1. Read the shared settings in `config.py`.
2. Read the preprocessing flow in `audio_pipeline.py`.
3. Check whether `preprocess.py` depends on that behavior.
4. Update logic in the shared pipeline instead of duplicating code elsewhere.
5. Verify impact on both training and inference.

### Change feature extraction

Relevant files:
- `audio_pipeline.py`
- `config.py`
- `voxonics_predictions.py`
- `VoiceBasedAi.ipynb`

Steps:
1. Read how features are generated in `audio_pipeline.py`.
2. Read feature-related constants in `config.py`.
3. Keep total feature size consistent.
4. Check whether the scaler and model assets are still valid.
5. Retrain or regenerate assets if needed.

### Change dataset preprocessing flow

Relevant files:
- `preprocess.py`
- `audio_pipeline.py`
- `config.py`

Steps:
1. Read batch preprocessing logic in `preprocess.py`.
2. Read shared audio behavior in `audio_pipeline.py`.
3. Keep file-level preprocessing aligned with the shared pipeline.
4. Verify output structure and saved files.

### Change speaker split behavior

Relevant files:
- `speaker_splitter.py`
- `manifest_builder.py`

Steps:
1. Read speaker ID extraction logic.
2. Read train or test split behavior.
3. Confirm that speaker overlap is still impossible after the change.
4. Verify downstream manifest labeling still works.

### Change manifest logic

Relevant files:
- `manifest_builder.py`
- `speaker_splitter.py`

Steps:
1. Read metadata parsing logic.
2. Read split assignment logic.
3. Read augmentation tracking logic.
4. Preserve source-original traceability.
5. Keep manifest fields consistent.

### Change inference flow

Relevant files:
- `voxonics_predictions.py`
- `audio_pipeline.py`
- `config.py`

Steps:
1. Read model, scaler, and encoder loading logic.
2. Read shared feature extraction flow.
3. Keep preprocessing, feature order, scaler input, and model input aligned.
4. Verify prediction still matches the trained pipeline.
