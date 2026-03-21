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
N_FEATURES = 6 * N_MFCC + 12   # 252
EPSILON = 1e-10 # a very small number to prevent division by zero

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


ASSETS_DIR = Path("model_assets")

MODEL_PATH  = ASSETS_DIR / "final_emotion_model.keras"
SCALER_PATH = ASSETS_DIR / "scaler.pickle"
ENCODER_PATH = ASSETS_DIR / "encoder.pickle"


RAW_DIR   = "raw_audio"
CLEAN_DIR = "clean_audio"