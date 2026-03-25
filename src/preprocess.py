from __future__ import annotations

import collections
import logging
from pathlib import Path

import soundfile as sf

from audio_pipeline import preprocess_audio_file, AudioProcessingError
from config import RAW_DIR, CLEAN_DIR


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def preprocess_directory(raw_dir: str = RAW_DIR, clean_dir: str = CLEAN_DIR) -> None:
    """
    Preprocess all .wav files from raw_dir and save them into clean_dir,
    retaining the original subfolder structure.
    """
    raw_path = Path(raw_dir)
    clean_path = Path(clean_dir)

    if not raw_path.exists():
        logger.error(f"Input directory does not exist: {raw_path}")
        return

    clean_path.mkdir(parents=True, exist_ok=True)

    # Use rglob to find files in subdirectories too
    wav_files = sorted(raw_path.rglob("*.wav"))
    if not wav_files:
        logger.warning(f"No .wav files found in: {raw_path}")
        return

    logger.info(f"Found {len(wav_files)} .wav files to process.")

    processed_count = 0
    rejections = collections.Counter()

    for file_path in wav_files:
        # Recreate subfolder structure in the output directory
        rel_path = file_path.relative_to(raw_path)
        output_path = clean_path / rel_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            processed_audio, sr = preprocess_audio_file(file_path)
            sf.write(output_path, processed_audio, sr)
            processed_count += 1
            if processed_count % 100 == 0:
                logger.info(f"Processed {processed_count} files...")

        except AudioProcessingError as e:
            # Expected audio format or length issues
            reason = e.__class__.__name__
            rejections[reason] += 1
            logger.warning(f"Skipped {rel_path}: {e}")

        except Exception as e:
            # Unexpected errors (corrupted files, etc)
            reason = "UnexpectedError"
            rejections[reason] += 1
            logger.error(f"Failed to process {rel_path}: {e}")

    # ── Final Report ──────────────────────────────────────────────────────────
    logger.info("-" * 50)
    logger.info("Preprocessing Complete")
    logger.info(f"Total files found: {len(wav_files)}")
    logger.info(f"Successfully processed: {processed_count}")

    failed_count = sum(rejections.values())
    if failed_count > 0:
        logger.info(f"Rejected/Failed: {failed_count}")
        for reason, count in rejections.most_common():
            logger.info(f"  - {reason}: {count}")
    else:
        logger.info("No files were rejected. All good! ✓")
    logger.info("-" * 50)


if __name__ == "__main__":
    preprocess_directory()