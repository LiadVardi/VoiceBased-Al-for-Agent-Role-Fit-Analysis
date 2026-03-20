from __future__ import annotations

from pathlib import Path


TARGET_SR = 22050   # librosa default
MONO = True

# NORMALIZE_AUDIO must be False to match the training pipeline.Setting this to True will shift the MFCC distribution and break inference.
NORMALIZE_AUDIO = False #if its true the model might think happy is angry 
TRIM_SILENCE = True #deletes silence at the begining and end of the audio
TRIM_TOP_DB = 20 #silence is defined as sound below 20 dezibel



TARGET_DURATION_SEC = 2.5 #if the audio is longer than 2.5 seconds, it will be cropped
MIN_DURATION_SEC = 0.5   # if the audio is shorter than this after trimming, it will be rejected
CROP_MODE = "start"      # if the audio is longer than 2.5 seconds, it will be cropped from the start
PAD_MODE = "constant"    # if the audio is shorter than 2.5 seconds, it will be padded with zeros(constant)


N_MFCC = 40 # 80 features total: 40 MFCC means + 40 MFCC standard deviations
EPSILON = 1e-10 # a very small number to prevent division by zero


ASSETS_DIR = Path("model_assets")

MODEL_PATH  = ASSETS_DIR / "final_emotion_model.keras"
SCALER_PATH = ASSETS_DIR / "scaler.pickle"
ENCODER_PATH = ASSETS_DIR / "encoder.pickle"


RAW_DIR   = "raw_audio"
CLEAN_DIR = "clean_audio"