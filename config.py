from __future__ import annotations

from pathlib import Path

# =========================
# Audio loading / cleaning
# =========================
TARGET_SR = 16000
MONO = True

NORMALIZE_AUDIO = True
TRIM_SILENCE = True
TRIM_TOP_DB = 20

# =========================
# Fixed-length policy
# =========================
TARGET_DURATION_SEC = 2.5
CROP_MODE = "start"      # options: "start", "center"
PAD_MODE = "constant"    # options: "constant", "wrap"

# =========================
# Feature extraction
# =========================
N_MFCC = 30
EPSILON = 1e-10

# =========================
# Artifacts / model files
# =========================
ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ARTIFACTS_DIR / "final_emotion_model.keras"
SCALER_PATH = ARTIFACTS_DIR / "scaler.pickle"
ENCODER_PATH = ARTIFACTS_DIR / "encoder.pickle"

# Optional dataset arrays (only if you choose to save them)
X_TRAIN_PATH = ARTIFACTS_DIR / "x_traincnn.npy"
X_TEST_PATH = ARTIFACTS_DIR / "x_testcnn.npy"
Y_TRAIN_PATH = ARTIFACTS_DIR / "y_train.npy"
Y_TEST_PATH = ARTIFACTS_DIR / "y_test.npy"

# =========================
# Data directories
# =========================
RAW_DIR = "raw_audio"
CLEAN_DIR = "clean_audio"