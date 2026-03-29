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