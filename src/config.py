from __future__ import annotations

from pathlib import Path


TARGET_SR = 22050
MONO = True

NORMALIZE_AUDIO = True
NORMALIZE_TYPE = "rms"
TARGET_RMS = 0.05
TRIM_SILENCE = True
TRIM_TOP_DB = 30

TARGET_DURATION_SEC = 2.5
MIN_DURATION_SEC = 0.5
CROP_MODE = "start"
PAD_MODE = "constant"

N_MFCC = 40
N_FEATURES = 6 * N_MFCC + 12
EPSILON = 1e-10

N_MELS       = 128
HOP_LENGTH   = 512
FIXED_FRAMES = 128

VAL_SIZE  = 0.15
TEST_SIZE = 0.15

AUGMENTATION_PROFILE = "full"

NOISE_AMP_RANGE    = (0.005, 0.035)
STRETCH_RATE_RANGE = (0.80,  1.20)
SHIFT_MS_RANGE     = (-200,  200)
PITCH_STEPS_RANGE  = (-2.0,  2.0)

MELD_EMOTION_MAP = {
    "anger":   "angry",
    "joy":     "happy",
    "neutral": "neutral",
    "sadness": "sad",
}

DATASET_WEIGHTS = {
    "RAVDESS":   1.5,
    "CREMAD":    1.0,
    "TESS":      1.0,
    "SAVEE":     1.0,
    "ASVP-ESD":  2.0,
    "MELD":      2.5,
    "VOXONICS":  5.0,
}

L2_REG             = 0.001
CNN_DROPOUT_CONV   = 0.4
CNN_DROPOUT_DENSE  = 0.5
CNN_DENSE_UNITS    = 64

LEARNING_RATE = 0.0005
EPOCHS        = 75

EARLY_STOPPING_PATIENCE = 10
LR_REDUCE_PATIENCE      = 5
LR_REDUCE_FACTOR        = 0.5
LR_MIN                  = 1e-5

SLIDING_WINDOW_ENABLED    = True
SLIDING_WINDOW_STRIDE_SEC = 1.25

PROJECT_ROOT = Path(__file__).resolve().parent.parent if Path(__file__).parent.name == "src" else Path(__file__).resolve().parent

DATA_DIR  = PROJECT_ROOT / "data"
RAW_DIR   = DATA_DIR / "raw_audio"
CLEAN_DIR = DATA_DIR / "clean_audio"
CACHE_DIR = DATA_DIR / "feature_cache"
CSV_DIR   = DATA_DIR / "csv"

ASSETS_DIR = PROJECT_ROOT / "model_assets"

MODEL_PATH   = ASSETS_DIR / "final_emotion_model.keras"
ENCODER_PATH = ASSETS_DIR / "encoder.pickle"

CHECKPOINT_DIR  = Path("C:/model_checkpoints")
CHECKPOINT_PATH = CHECKPOINT_DIR / "best_emotion_model.keras"

FINETUNE_N_FREEZE_LAYERS = 11
FINETUNE_LR              = 1e-4
FINETUNE_EPOCHS          = 30
FINETUNE_PATIENCE        = 7
FINETUNE_VAL_SIZE        = 0.20
FINETUNE_AUGMENT_PROFILE = "light"

VOXONICS_LABELED_DIR = "data/raw_audio/VOXONICS"

FINETUNE_CHECKPOINT_PATH = CHECKPOINT_DIR / "finetuned_emotion_model.keras"
FINETUNED_MODEL_PATH     = ASSETS_DIR / "finetuned_emotion_model.keras"
