"""
CLI tool for running emotion inference on one .wav file or a whole folder.

Exit codes
----------
  0  All files processed successfully (or with tolerated per-file errors).
  1  Fatal error (model not found, bad arguments, feature mismatch, etc.).

  An example of how to use the tool:
    python voxonics_predictions.py --input_dir test_audio/"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from audio_pipeline import (
    AudioEmptyError,
    AudioProcessingError,
    AudioTooShortError,
    extract_features_from_file,
)
from config import ENCODER_PATH, MODEL_PATH, N_FEATURES, SCALER_PATH



class ModelLoadError(RuntimeError):
    """Raised when the model, scaler, or encoder cannot be loaded from disk."""

class FeatureMismatchError(RuntimeError):
    """Raised when the extracted feature vector length doesn't match the scaler."""



def load_engine():
   
    from tensorflow.keras.models import load_model  # lazy import — keeps startup fast

    _check_asset(MODEL_PATH,   "Keras model")
    _check_asset(SCALER_PATH,  "StandardScaler")
    _check_asset(ENCODER_PATH, "OneHotEncoder")

    try:
        model = load_model(str(MODEL_PATH))
    except Exception as exc:
        raise ModelLoadError(f"Failed to load Keras model from '{MODEL_PATH}': {exc}") from exc

    try:
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
    except Exception as exc:
        raise ModelLoadError(f"Failed to load scaler from '{SCALER_PATH}': {exc}") from exc

    try:
        with open(ENCODER_PATH, "rb") as f:
            encoder = pickle.load(f)
    except Exception as exc:
        raise ModelLoadError(f"Failed to load encoder from '{ENCODER_PATH}': {exc}") from exc

    return model, scaler, encoder


def _check_asset(path: Path, label: str) -> None:
    if not Path(path).exists():
        raise ModelLoadError(
            f"{label} not found at '{path}'.\n"
            "  → Make sure you have trained the model and saved assets to model_assets/."
        )



def predict_file(
    audio_path: str | Path,
    model,
    scaler,
    encoder,
) -> dict[str, float]:
    
    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"File not found: '{audio_path}'")
    if not audio_path.is_file():
        raise FileNotFoundError(f"Path is not a file: '{audio_path}'")
    if audio_path.stat().st_size == 0:
        raise AudioEmptyError(f"File is empty (0 bytes): '{audio_path}'")

    features = extract_features_from_file(audio_path)

    if features.shape[0] != N_FEATURES:
        raise FeatureMismatchError(
            f"Feature vector has {features.shape[0]} dimensions, "
            f"but the scaler expects {N_FEATURES}.\n"
            "  → Re-extract features or retrain the model."
        )

    expected_scaler_feats = scaler.n_features_in_
    if features.shape[0] != expected_scaler_feats:
        raise FeatureMismatchError(
            f"Scaler was fitted on {expected_scaler_feats} features, "
            f"but extracted {features.shape[0]}.\n"
            "  → The scaler and audio_pipeline.py are out of sync. Retrain."
        )

    features_scaled = scaler.transform(features.reshape(1, -1))
    features_cnn    = np.expand_dims(features_scaled, axis=2)
    probs           = model.predict(features_cnn, verbose=0)[0]

    emotion_labels = encoder.categories_[0]
    return {label: float(prob) for label, prob in zip(emotion_labels, probs)}



def _print_result(filename: str, results: dict[str, float], top_k: int) -> None:
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    top = sorted_results[:top_k]

    print(f"\nResults for: {filename}")
    print("=" * 44)
    for emotion, prob in top:
        confidence = prob * 100
        bar = "█" * int(confidence / 5)
        print(f"  {emotion.ljust(14)} {confidence:6.2f}%  {bar}")
    print("=" * 44)


def _build_row(audio_path: Path, results: dict[str, float]) -> dict[str, Any]:
    """Build a flat dict suitable for CSV/JSON output."""
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    top_emotion, top_conf = sorted_results[0]
    row: dict[str, Any] = {
        "file":        str(audio_path),
        "top_emotion": top_emotion,
        "confidence":  round(top_conf * 100, 2),
    }
    for label, prob in results.items():
        row[label] = round(prob * 100, 2)
    return row


def _save_results(rows: list[dict[str, Any]], output_path: str) -> None:
    out = Path(output_path)
    suffix = out.suffix.lower()

    if suffix == ".json":
        with open(out, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {out}")

    elif suffix == ".csv":
        if not rows:
            print("No results to save.")
            return
        fieldnames = list(rows[0].keys())
        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nResults saved to: {out}")

    else:
        print(
            f"[WARNING] Unknown output extension '{suffix}'. "
            "Supported formats: .csv, .json"
        )



def run_single(args, model, scaler, encoder) -> int:
    """Returns exit code: 0 = success, 1 = fatal error."""
    try:
        results = predict_file(args.input, model, scaler, encoder)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    except AudioEmptyError as exc:
        print(f"[ERROR] Audio is silent or empty — {exc}", file=sys.stderr)
        return 1
    except AudioTooShortError as exc:
        print(f"[ERROR] Audio too short after silence trimming — {exc}", file=sys.stderr)
        return 1
    except AudioProcessingError as exc:
        print(f"[ERROR] Audio preprocessing failed — {exc}", file=sys.stderr)
        return 1
    except FeatureMismatchError as exc:
        print(f"[ERROR] Feature mismatch — {exc}", file=sys.stderr)
        return 1

    if not args.quiet:
        _print_result(Path(args.input).name, results, args.top_k)

    if args.output:
        _save_results([_build_row(Path(args.input), results)], args.output)

    return 0



def run_folder(args, model, scaler, encoder) -> int:
    """Recursively predicts on every .wav in the folder. Returns exit code."""
    folder = Path(args.input_dir)
    if not folder.exists():
        print(f"[ERROR] Folder not found: '{folder}'", file=sys.stderr)
        return 1
    if not folder.is_dir():
        print(f"[ERROR] Not a directory: '{folder}'", file=sys.stderr)
        return 1

    wav_files = sorted(folder.rglob("*.wav"))
    if not wav_files:
        print(f"[WARNING] No .wav files found in '{folder}'.")
        return 0

    print(f"Found {len(wav_files)} .wav file(s) in '{folder}'. Running inference...")

    rows: list[dict[str, Any]] = []
    failed: list[tuple[str, str]] = []

    for wav in wav_files:
        try:
            results = predict_file(wav, model, scaler, encoder)
            row = _build_row(wav, results)
            rows.append(row)
            if not args.quiet:
                _print_result(wav.name, results, args.top_k)
        except FileNotFoundError as exc:
            failed.append((str(wav), f"File not found: {exc}"))
        except AudioEmptyError as exc:
            failed.append((str(wav), f"Silent/empty audio: {exc}"))
        except AudioTooShortError as exc:
            failed.append((str(wav), f"Too short after trimming: {exc}"))
        except AudioProcessingError as exc:
            failed.append((str(wav), f"Preprocessing error: {exc}"))
        except FeatureMismatchError as exc:
            print(f"[ERROR] Feature mismatch — {exc}\n  Aborting folder run.", file=sys.stderr)
            return 1  # Fatal — all files would fail the same way

    print(f"\n{'─'*50}")
    print(f"  Processed:  {len(rows)}/{len(wav_files)} files")
    if failed:
        print(f"  Skipped:    {len(failed)} file(s) with errors:")
        for path, reason in failed:
            print(f"    • {Path(path).name}: {reason}")
    print(f"{'─'*50}")

    if args.output and rows:
        _save_results(rows, args.output)

    return 0



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voxonics_predictions",
        description="Voxonics Emotion Recognition — inference CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input", "-i",
        metavar="FILE",
        help="Path to a single .wav file.",
    )
    source.add_argument(
        "--input_dir", "-d",
        metavar="FOLDER",
        help="Path to a folder. All .wav files inside (recursive) are processed.",
    )

    parser.add_argument(
        "--output", "-o",
        metavar="PATH",
        help="Save results to this file. Supported formats: .csv, .json.",
    )
    parser.add_argument(
        "--top_k", "-k",
        type=int,
        default=3,
        metavar="N",
        help="Number of top emotions to display (default: 3).",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress per-file bar chart. Still prints the summary table.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top_k must be at least 1.")

    try:
        print("Loading Voxonics engine...")
        model, scaler, encoder = load_engine()
        print("Engine loaded.\n")
    except ModelLoadError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        sys.exit(1)

    if args.input:
        exit_code = run_single(args, model, scaler, encoder)
    else:
        exit_code = run_folder(args, model, scaler, encoder)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()