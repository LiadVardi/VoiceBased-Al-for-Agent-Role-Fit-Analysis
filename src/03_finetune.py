"""
03_finetune.py  —  Fine-tune the base model on telephonic Voxonics calls
=========================================================================
Strategy (Option B):
  - Load the pretrained model (base model trained on studio + MELD data)
  - Freeze CNN blocks 1-3  (general acoustic feature extractors)
  - Retrain CNN block 4 + dense head on your labeled telephonic clips
  - Save the fine-tuned model separately (base model is NOT overwritten)

HOW TO RUN:
-----------
  python src/03_finetune.py

DATA REQUIRED:
--------------
  Voxonics_labeled/
      angry/    ← .wav or .mp3 clips
      happy/
      neutral/
      sad/

OUTPUT:
-------
  C:/model_checkpoints/finetuned_emotion_model.keras  (best checkpoint)
  model_assets/finetuned_emotion_model.keras          (final copy)
"""

from __future__ import annotations

import os
import sys
import pickle
import shutil
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from config import (
    TARGET_SR, MONO,
    TARGET_DURATION_SEC, SLIDING_WINDOW_STRIDE_SEC,
    ENCODER_PATH, MODEL_PATH,
    FINETUNE_N_FREEZE_LAYERS, FINETUNE_LR, FINETUNE_EPOCHS,
    FINETUNE_PATIENCE, FINETUNE_VAL_SIZE,
    VOXONICS_LABELED_DIR, FINETUNE_CHECKPOINT_PATH, FINETUNED_MODEL_PATH,
    CHECKPOINT_DIR, LR_MIN,
)
from audio_pipeline import extract_spectrogram_from_audio_array
from augmentation import get_augmentations

SUPPORTED = {".wav", ".mp3", ".m4a", ".flac"}
EMOTIONS   = ["angry", "happy", "neutral", "sad"]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_voxonics_dataframe(labeled_dir) -> pd.DataFrame:
    """
    Scan Voxonics_labeled/{emotion}/ sub-folders and return a DataFrame
    with columns [path, emotion].
    """
    from pathlib import Path
    labeled_dir = Path(labeled_dir)

    if not labeled_dir.exists():
        raise FileNotFoundError(
            f"\n[ERROR] Labeled data folder not found: {labeled_dir}\n"
            f"Please create:\n"
            f"  {labeled_dir}/angry/\n"
            f"  {labeled_dir}/happy/\n"
            f"  {labeled_dir}/neutral/\n"
            f"  {labeled_dir}/sad/\n"
            f"and place your .wav / .mp3 clips in the correct sub-folder."
        )

    rows = []
    for emotion in EMOTIONS:
        emotion_dir = labeled_dir / emotion
        if not emotion_dir.exists():
            print(f"  [WARN] Sub-folder not found: {emotion_dir}")
            continue
        clips = [f for f in emotion_dir.iterdir() if f.suffix.lower() in SUPPORTED]
        for clip in clips:
            rows.append({"path": str(clip), "emotion": emotion})
        print(f"  {emotion:<10} : {len(clips)} clips")

    if not rows:
        raise ValueError("No audio clips found. Check your Voxonics_labeled/ folder.")

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Sliding-window spectrogram extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_windows(audio: np.ndarray, sr: int) -> list[np.ndarray]:
    """
    Extract overlapping 2.5-second windows from an audio clip.
    Windows that are mostly silence (trimmed below MIN_DURATION_SEC) are skipped.
    Returns a list of (128, 128) spectrograms.
    """
    window_samples = int(TARGET_DURATION_SEC * sr)
    stride_samples = int(SLIDING_WINDOW_STRIDE_SEC * sr)
    spectrograms   = []

    if len(audio) <= window_samples:
        # Short clip — single crop/pad, still wrapped in case of silence
        try:
            spectrograms.append(extract_spectrogram_from_audio_array(audio, sr))
        except Exception:
            pass
        return spectrograms

    start = 0
    while start + window_samples <= len(audio):
        window = audio[start : start + window_samples]
        try:
            spectrograms.append(extract_spectrogram_from_audio_array(window, sr))
        except Exception:
            pass   # silent/too-short window — skip it
        start += stride_samples

    return spectrograms


def build_spectrogram_dataset(
    df: pd.DataFrame,
    augment: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """
    Load all clips, extract sliding-window spectrograms (+ augmentations),
    return (specs_array, labels_list).
    """
    specs, labels = [], []
    augmentations = get_augmentations() if augment else {}

    for _, row in df.iterrows():
        try:
            audio, sr = librosa.load(row["path"], sr=TARGET_SR, mono=MONO)
        except Exception as exc:
            print(f"  [SKIP] {row['path']}: {exc}")
            continue

        # Original windows
        for spec in extract_windows(audio, sr):
            specs.append(spec)
            labels.append(row["emotion"])

        # Augmented versions (applied to full clip, then windowed)
        if augment:
            for aug_name, aug_fn in augmentations.items():
                try:
                    aug_audio = aug_fn(audio, sr=sr)
                    for spec in extract_windows(aug_audio, sr):
                        specs.append(spec)
                        labels.append(row["emotion"])
                except Exception:
                    pass

    return np.array(specs, dtype=np.float32), labels


# ─────────────────────────────────────────────────────────────────────────────
# 3. Main fine-tuning pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Voxonics Fine-Tuning  (Option B — Frozen CNN blocks 1-3)")
    print("=" * 60)

    # ── Load encoder (must match base model) ──────────────────────────────────
    if not ENCODER_PATH.exists():
        raise FileNotFoundError(
            f"Encoder not found at {ENCODER_PATH}.\n"
            "Run the notebook through the training cell first."
        )
    with open(ENCODER_PATH, "rb") as f:
        encoder = pickle.load(f)
    emotion_classes = list(encoder.categories_[0])
    print(f"\nClasses (from base encoder): {emotion_classes}")

    # ── Load Voxonics clips ────────────────────────────────────────────────────
    print(f"\n1. Loading labeled clips from: {VOXONICS_LABELED_DIR}")
    df = load_voxonics_dataframe(VOXONICS_LABELED_DIR)
    print(f"   Total clips: {len(df)}")

    # ── Train / val split (stratified by emotion) ─────────────────────────────
    train_df, val_df = train_test_split(
        df, test_size=FINETUNE_VAL_SIZE, stratify=df["emotion"], random_state=42
    )
    print(f"\n   Train: {len(train_df)} clips  |  Val: {len(val_df)} clips")

    # ── Extract spectrograms ───────────────────────────────────────────────────
    print("\n2. Extracting spectrograms (sliding window + augmentation)...")
    x_train_raw, y_train_labels = build_spectrogram_dataset(train_df, augment=True)
    x_val_raw,   y_val_labels   = build_spectrogram_dataset(val_df,   augment=False)

    print(f"   Train samples: {len(x_train_raw)}")
    print(f"   Val   samples: {len(x_val_raw)}")

    # Add channel dim → (N, 128, 128, 1)
    x_train = x_train_raw[..., np.newaxis]
    x_val   = x_val_raw[...,   np.newaxis]

    # One-hot encode labels using the BASE MODEL's encoder
    y_train = encoder.transform([[l] for l in y_train_labels])
    y_val   = encoder.transform([[l] for l in y_val_labels])

    # ── Load base model + freeze blocks 1-3 ───────────────────────────────────
    print(f"\n3. Loading base model from: {MODEL_PATH}")
    model = tf.keras.models.load_model(str(MODEL_PATH))

    print(f"   Freezing first {FINETUNE_N_FREEZE_LAYERS} layers (CNN blocks 1-3)...")
    for layer in model.layers[:FINETUNE_N_FREEZE_LAYERS]:
        layer.trainable = False
    for layer in model.layers[FINETUNE_N_FREEZE_LAYERS:]:
        layer.trainable = True

    trainable_count = sum(1 for l in model.layers if l.trainable)
    frozen_count    = sum(1 for l in model.layers if not l.trainable)
    print(f"   Frozen: {frozen_count} layers  |  Trainable: {trainable_count} layers")

    # ── Compile with lower learning rate ──────────────────────────────────────
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=FINETUNE_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    print(f"   Learning rate: {FINETUNE_LR}  (base was 5e-4)")

    # ── Callbacks ─────────────────────────────────────────────────────────────
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            str(FINETUNE_CHECKPOINT_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=FINETUNE_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_accuracy",
            patience=3,
            factor=0.5,
            min_lr=LR_MIN,
            verbose=1,
        ),
    ]

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\n4. Fine-tuning for up to {FINETUNE_EPOCHS} epochs...")
    print(f"   Checkpoint: {FINETUNE_CHECKPOINT_PATH}")

    history = model.fit(
        x_train, y_train,
        epochs=FINETUNE_EPOCHS,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        verbose=1,
    )

    # ── Evaluate on validation set ────────────────────────────────────────────
    print("\n5. Validation results:")
    pred_val   = model.predict(x_val, verbose=0)
    y_pred_lab = encoder.inverse_transform(pred_val)
    y_true_lab = encoder.inverse_transform(y_val)

    print(classification_report(y_true_lab, y_pred_lab, target_names=emotion_classes))

    # ── Copy best checkpoint → project assets ──────────────────────────────────
    FINETUNED_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(str(FINETUNE_CHECKPOINT_PATH), str(FINETUNED_MODEL_PATH))
    print(f"✓ Fine-tuned model saved to: {FINETUNED_MODEL_PATH}")
    print("\nTo use it for predictions, update Voxonics_Predictions.py:")
    print(f"  model = load_model(r'{FINETUNED_MODEL_PATH}')")


if __name__ == "__main__":
    main()
