# Speech Emotion Recognition - Voxonics

Automatically detects the emotional state of speakers in telephonic call-center recordings.  
Classifies audio into four emotions: **Angry · Happy · Neutral · Sad**



## What It Does

- **Single-file prediction** — analyze any `.wav` or `.mp3` file and get an emotion confidence score in seconds
- **Full call timeline** - slide a window through an entire call and see how emotion shifts minute by minute
- **Continuously improvable** - label new clips, retrain, deploy; no code changes required

---

## Results

| Emotion | Precision | Recall | F1 |
|---------|-----------|--------|-----|
| Angry   | 0.75 | 0.72 | 0.74 |
| Happy   | 0.61 | 0.59 | 0.60 |
| Neutral | 0.70 | 0.77 | 0.73 |
| Sad     | 0.72 | 0.63 | 0.68 |
| **Overall** | | | **0.70** |

- **83%** accuracy on direct telephonic call tests (Voxonics recordings)
- **70%** accuracy on the full mixed test set (academic + telephonic)

---

## Architecture

```
Raw audio (Azure Blob Storage)
       ↓  01_sync_data.py       - download all datasets
       ↓  02_extract_features.py - extract 128×128 Log-Mel Spectrograms
       ↓  VoiceBasedAi.ipynb    - train 2D CNN (75 epochs, 4 emotion classes)
       ↓  model_assets/         - final_emotion_model.keras + encoder.pickle
       ↓
Voxonics_Predictions.py  - single-file inference
Call_Analysis.py         - full call emotion timeline
```

**Model:** 4-block 2D CNN on Log-Mel Spectrograms (128×128)  
**Training data:** ~11,000+ clips from 7 datasets  
**Domain adaptation:** VOXONICS telephonic clips weighted 5× in training

---

## Project Structure

```
├── src/
│   ├── config.py               # All hyperparameters and paths
│   ├── audio_pipeline.py       # Single source of truth for audio processing
│   ├── 01_sync_data.py         # Azure Blob → local disk
│   ├── 02_extract_features.py  # Spectrogram extraction + caching
│   ├── 03_finetune.py          # Quick domain adaptation 
│   ├── split_audio.py          # Split long calls into 4s segments for labeling
│   ├── speaker_splitter.py     # Speaker-aware train/val/test split
│   ├── manifest_builder.py     # Dataset manifest with augmentation traceability
│   └── augmentation.py         # Noise / stretch / shift / pitch augmentation
├── Voxonics_Predictions.py     # Inference: single file
├── Call_Analysis.py            # Inference: full call timeline
├── VoiceBasedAi.ipynb          # Main training notebook
├── model_assets/
│   ├── final_emotion_model.keras
│   └── encoder.pickle
├── DOCUMENTATION.md            # Detailed file-by-file documentation
├── PROJECT_GUIDE.md            # Architecture rules and change procedures
└── AI_WORKFLOW.md              # Reusable AI prompts for this project
```

---

## Setup

### Requirements

- Python 3.10+
- `ffmpeg` installed and on PATH (required for MP3 support)

### Install dependencies

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Configure environment

Create a `.env` file in the project root:

```
AZURE_STORAGE_CONNECTION_STRING=your_connection_string_here
```

---

## How to Run

### 1. Sync data from Azure

```powershell
.venv\Scripts\python.exe src\01_sync_data.py
```

Downloads RAVDESS, CREMAD, TESS, SAVEE, ASVP-ESD, MELD, and VOXONICS datasets to `data/raw_audio/`.

### 2. Extract features

```powershell
.venv\Scripts\python.exe src\02_extract_features.py
```

Extracts Log-Mel Spectrograms from all audio, caches `.npy` files, and outputs CSVs for the notebook. Only new files are processed on re-runs.

### 3. Train the model

Open and run `VoiceBasedAi.ipynb` from top to bottom.  
Saves the trained model to `model_assets/final_emotion_model.keras`.

### 4. Predict emotion - single file

```powershell
.venv\Scripts\python.exe Voxonics_Predictions.py path\to\audio.wav
```

**Example output:**
```
angry        |  91.20% ██████████████████
neutral      |   6.40% █
happy        |   1.80%
sad          |   0.60%
```

### 5. Analyze a full call

```powershell
.venv\Scripts\python.exe Call_Analysis.py path\to\call.wav
```

**Example output:**
```
EMOTION TIMELINE - call_001.wav
Duration : 1:24 (84.0s)

TIME            EMOTION     CONFIDENCE
0:00 - 0:02     neutral      78.4%
0:02 - 0:04     neutral      82.1%
0:04 - 0:06     angry        91.5%
...

SUMMARY
angry        12.0%
happy         5.0%
neutral      75.0%
sad           8.0%
```

Use `--stride` to control window spacing (default: 2 seconds):
```powershell
.venv\Scripts\python.exe Call_Analysis.py call.wav --stride 4
```

### 6. Split long calls for labeling

```powershell
.venv\Scripts\python.exe src\split_audio.py --input Voxonics_audio
```

Cuts every file in `Voxonics_audio/` into 4-second segments saved to `Cropped_audio/`.

---

## Adding New Training Data (VOXONICS)

The model improves automatically as you add more labeled telephonic clips.

**Folder structure required:**
```
data/raw_audio/VOXONICS/
    angry/    ← .wav or .mp3 clips labeled as angry
    happy/
    neutral/
    sad/
```

After adding clips:
1. Run `02_extract_features.py` (only new files are processed)
2. Retrain the notebook
3. New model is saved to `model_assets/`

---

## Datasets Used

| Dataset | Description | Clips |
|---------|-------------|-------|
| RAVDESS | Professional actors, studio | ~1,500 |
| CREMA-D | 91 actors, diverse demographics | ~7,400 |
| TESS | Female speakers, University of Toronto | ~2,800 |
| SAVEE | Male British speakers | ~480 |
| ASVP-ESD | Extended speech and sounds | ~1,000+ |
| MELD | Real TV dialogue (Friends) | ~1,400 |
| **VOXONICS** | **Real telephonic call-center recordings** | **~570** |

VOXONICS clips are weighted **5×** in training - reflecting their direct relevance to the production environment.

---

## Key Design Decisions

- **RMS normalization** - equalizes volume across speakers, reducing loudness-anger confusion
- **Sliding window** - clips longer than 2.5s generate multiple training samples (stride: 1.25s)
- **Speaker-aware split** - no speaker appears in more than one of train/val/test
- **Temporal smoothing** - isolated "angry" windows surrounded by neutral windows are corrected in `Call_Analysis.py`
- **Spectrogram caching** - `.npy` files prevent recomputing features across training runs

---

## Documentation

| File | Purpose |
|------|---------|
| `DOCUMENTATION.md` | Detailed explanation of every file, the audio pipeline, CNN architecture, and risks |
| `PROJECT_GUIDE.md` | Architecture rules, change procedures, file responsibilities |
| `AI_WORKFLOW.md` | Reusable prompts for working on this project with an AI assistant |
