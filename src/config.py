from __future__ import annotations

from pathlib import Path


TARGET_SR = 22050   # librosa default
MONO = True

# Normalization policy: RMS normalization limits real-world volume differences
NORMALIZE_AUDIO = True
NORMALIZE_TYPE = "rms"   # "rms" or "peak"
TARGET_RMS = 0.05        # Target loudness for RMS normalization (0.05 is standard for speech)
TRIM_SILENCE = True #deletes silence at the begining and end of the audio
TRIM_TOP_DB = 30 #silence is defined as sound below 30 dezibel



TARGET_DURATION_SEC = 2.5 #if the audio is longer than 2.5 seconds, it will be cropped
MIN_DURATION_SEC = 0.5   # if the audio is shorter than this after trimming, it will be rejected
CROP_MODE = "start"      # if the audio is longer than 2.5 seconds, it will be cropped from the start
PAD_MODE = "constant"    # if the audio is shorter than 2.5 seconds, it will be padded with zeros(constant)


N_MFCC = 40
N_FEATURES = 6 * N_MFCC + 12   # 252  (kept for 1D CNN compatibility)
EPSILON = 1e-10 # a very small number to prevent division by zero

# ── Log-Mel Spectrogram (2D CNN) ──────────────────────────────────────────────
N_MELS       = 128   # number of mel frequency bins (height of the spectrogram image)
HOP_LENGTH   = 512   # STFT hop — controls time resolution
FIXED_FRAMES = 128   # all spectrograms are cropped/padded to this many time steps (width)

# val = used for hyperparameter tuning / EarlyStopping.
# test = LOCKED — only touched for the final evaluation report.
VAL_SIZE  = 0.15   # 15% of speakers go to validation
TEST_SIZE = 0.15   # 15% of speakers go to test (locked holdout)


# ── Augmentation ──────────────────────────────────────────────────────────────
# Profile controls which augmentations are applied to the TRAINING set only.
#   "none"  → no augmentation  (baseline ablation)
#   "light" → noise + mild pitch shift only
#   "full"  → all four techniques with full ranges
AUGMENTATION_PROFILE = "full"

# Per-technique parameter ranges (used by augmentation.py)
NOISE_AMP_RANGE    = (0.005, 0.035)  # fraction of signal peak
STRETCH_RATE_RANGE = (0.80,  1.20)   # < 1 = slower,  > 1 = faster
SHIFT_MS_RANGE     = (-200,  200)    # time shift in milliseconds (±200ms)
PITCH_STEPS_RANGE  = (-2.0,  2.0)    # semitones (±2 = about one whole tone)

# ── MELD Dataset ──────────────────────────────────────────────────────────────
# Maps MELD's 7-class scheme to this project's 4 target emotions.
MELD_EMOTION_MAP = {
    "anger":   "angry",
    "joy":     "happy",
    "neutral": "neutral",
    "sadness": "sad",
    # "surprise" → dropped
    # "fear"     → dropped
    # "disgust"  → dropped
}

# ── Dataset Sample Weights ────────────────────────────────────────────────────
# Controls how much each dataset's samples influence the loss during training.
# Studio recordings (RAVDESS/CREMAD/TESS/SAVEE) = baseline weight 1.0
# ASVP-ESD = slightly upweighted (more phonetic diversity)
# MELD = highest weight (real conversational speech — closest to production)
DATASET_WEIGHTS = {
    "RAVDESS":   1.5,
    "CREMAD":    1.0,
    "TESS":      1.0,
    "SAVEE":     1.0,
    "ASVP-ESD":  2.0,
    "MELD":      2.5,
    "VOXONICS":  5.0,  # ← exact production domain (telephonic, real calls)
}



# ── CNN Architecture ──────────────────────────────────────────────────────────
L2_REG             = 0.001  # L2 regularization factor applied to Conv2D kernels
CNN_DROPOUT_CONV   = 0.4    # Dropout rate after conv blocks 2 and 3
CNN_DROPOUT_DENSE  = 0.5    # Dropout rate in the dense head
CNN_DENSE_UNITS    = 64     # Number of units in the Dense layer before softmax

# ── Training ──────────────────────────────────────────────────────────────────
LEARNING_RATE = 0.0005  # Initial Adam learning rate
EPOCHS        = 75    # Maximum number of training epochs

# ── Callbacks ─────────────────────────────────────────────────────────────────
EARLY_STOPPING_PATIENCE = 10   # Epochs without improvement before stopping
LR_REDUCE_PATIENCE      = 5    # Epochs without improvement before reducing LR
LR_REDUCE_FACTOR        = 0.5  # Factor to multiply LR by on plateau
LR_MIN                  = 1e-5 # Minimum allowed learning rate

# ── Sliding Window (maximises training samples from longer clips) ───────────────
# When enabled, clips longer than TARGET_DURATION_SEC are sliced into
# overlapping windows instead of cropping to just the first 2.5 seconds.
# Every window gets the same emotion label as the parent clip.
SLIDING_WINDOW_ENABLED    = True   # set False to revert to single-crop
SLIDING_WINDOW_STRIDE_SEC = 1.25   # 50% overlap: new window every 1.25 s

# ── Directory Layout ──────────────────────────────────────────────────────────
# Resolve project root dynamically so config.py can safely live in src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent if Path(__file__).parent.name == "src" else Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw_audio"
CLEAN_DIR = DATA_DIR / "clean_audio"
CACHE_DIR = DATA_DIR / "feature_cache"
CSV_DIR = DATA_DIR / "csv"

ASSETS_DIR = PROJECT_ROOT / "model_assets"

MODEL_PATH  = ASSETS_DIR / "final_emotion_model.keras"
SCALER_PATH = ASSETS_DIR / "scaler.pickle"
ENCODER_PATH = ASSETS_DIR / "encoder.pickle"

# Local checkpoint path OUTSIDE OneDrive — avoids sync delay during training
CHECKPOINT_DIR = Path("C:/model_checkpoints")
CHECKPOINT_PATH = CHECKPOINT_DIR / "best_emotion_model.keras"

# ── Fine-Tuning (Option B — domain adaptation on telephonic calls) ────────────
# Freeze the first N layers (CNN blocks 1-3, which extract general features).
# Only the last CNN block + dense head are retrained on Voxonics data.
#
# Layer map for the 4-block 2D CNN:
#   Block 1 : layers  0– 2  (Conv32,  BN, MaxPool)
#   Block 2 : layers  3– 6  (Conv64,  BN, MaxPool, Dropout)
#   Block 3 : layers  7–10  (Conv128, BN, MaxPool, Dropout)
#   Block 4 : layers 11–13  (Conv128, BN, GlobalAvgPool)  ← trainable
#   Dense   : layers 14–16  (Dense64, Dropout, Softmax)   ← trainable
FINETUNE_N_FREEZE_LAYERS  = 11      # freeze blocks 1-3; train block 4 + head
FINETUNE_LR               = 1e-4    # much lower than original 5e-4
FINETUNE_EPOCHS           = 30
FINETUNE_PATIENCE         = 7       # early-stopping patience
FINETUNE_VAL_SIZE         = 0.20    # 20% of Voxonics clips → validation
FINETUNE_AUGMENT_PROFILE  = "light" # lighter augmentation for small dataset

# Where to read labeled telephonic clips from (one sub-folder per emotion):
#   Voxonics_labeled/
#       angry/   ← your .wav / .mp3 clips
#       happy/
#       neutral/
#       sad/
VOXONICS_LABELED_DIR = "data/raw_audio/VOXONICS"

# Saved fine-tuned model (separate from base model)
FINETUNE_CHECKPOINT_PATH  = CHECKPOINT_DIR / "finetuned_emotion_model.keras"
FINETUNED_MODEL_PATH      = ASSETS_DIR / "finetuned_emotion_model.keras"
