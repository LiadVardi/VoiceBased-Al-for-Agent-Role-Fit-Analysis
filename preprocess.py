from __future__ import annotations

from pathlib import Path

import soundfile as sf

from audio_pipeline import preprocess_audio_file
from config import RAW_DIR, CLEAN_DIR


def preprocess_directory(raw_dir: str = RAW_DIR, clean_dir: str = CLEAN_DIR) -> None:
    """
    Preprocess all .wav files from raw_dir and save them into clean_dir.
    """
    raw_path = Path(raw_dir)
    clean_path = Path(clean_dir)

    if not raw_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {raw_path}")

    clean_path.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(raw_path.glob("*.wav"))
    if not wav_files:
        print(f"No .wav files found in: {raw_path}")
        return

    processed_count = 0
    failed_count = 0

    for file_path in wav_files:
        try:
            processed_audio, sr = preprocess_audio_file(file_path)
            output_path = clean_path / file_path.name
            sf.write(output_path, processed_audio, sr)
            processed_count += 1
            print(f"[OK] {file_path.name} -> {output_path}")
        except Exception as e:
            failed_count += 1
            print(f"[ERROR] Failed to process {file_path.name}: {e}")

    print("-" * 50)
    print(f"Done preprocessing audio files.")
    print(f"Processed: {processed_count}")
    print(f"Failed:    {failed_count}")
    print("-" * 50)


if __name__ == "__main__":
    preprocess_directory()