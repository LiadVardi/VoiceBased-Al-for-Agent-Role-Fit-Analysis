"""
meld_upload.py  —  MELD Dataset → Azure Blob Storage
======================================================
Run ONCE to prepare and upload the MELD dataset.

STEP 1 — Download MELD from GitHub:
    https://github.com/declare-lab/MELD
    You need: MELD.Raw.tar.gz  (contains the mp4 audio clips + CSVs)

STEP 2 — Extract the archive so you have a folder like:
    <meld_dir>/
    ├── train_sent_emo.csv
    ├── dev_sent_emo.csv
    ├── test_sent_emo.csv
    ├── train/train_splits/         dia0_utt0.mp4, dia0_utt1.mp4 ...
    ├── dev/dev_splits_complete/    dia0_utt0.mp4 ...
    └── test/output_repeated_splits_test/  dia0_utt0.mp4 ...

STEP 3 — Install ffmpeg (needed to convert mp4 → wav):
    Windows: winget install ffmpeg    (or download from https://ffmpeg.org)

STEP 4 — Run this script:
    python src/meld_upload.py --meld-dir "C:/path/to/MELD.Raw"

What this script does:
  1. Combines train / dev / test CSVs → meld_labels.csv
  2. Keeps only the 4 target emotions (drops surprise/fear/disgust)
  3. Converts mp4 clips → wav (22 050 Hz, mono)
  4. Renames each file: {Speaker}_dia{Dialogue_ID}_utt{Utterance_ID}.wav
  5. Uploads everything to Azure under the "MELD/" prefix
"""

import argparse
import io
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

# ── Allow running from the project root ──────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from config import MELD_EMOTION_MAP

load_dotenv()

CONTAINER_NAME = "wav-files"
MELD_PREFIX    = "MELD/"

# Folders inside MELD.Raw that hold the mp4 clips
SPLIT_DIRS = {
    "train": "train/train_splits",
    "dev":   "dev/dev_splits_complete",
    "test":  "test/output_repeated_splits_test",
}

# CSV filenames inside MELD.Raw
SPLIT_CSVS = {
    "train": "train_sent_emo.csv",
    "dev":   "dev_sent_emo.csv",
    "test":  "test_sent_emo.csv",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_and_merge_csvs(meld_dir: Path) -> pd.DataFrame:
    """Combine train / dev / test label CSVs and filter to 4 emotions."""
    dfs = []
    for split, csv_name in SPLIT_CSVS.items():
        csv_path = meld_dir / csv_name
        if not csv_path.exists():
            print(f"  [WARN] CSV not found: {csv_path} — skipping {split}")
            continue
        df = pd.read_csv(csv_path)
        df["split"] = split
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    # Keep only the 4 target emotions
    before = len(combined)
    combined = combined[combined["Emotion"].isin(MELD_EMOTION_MAP)].copy()
    combined["Emotions"] = combined["Emotion"].map(MELD_EMOTION_MAP)
    dropped = before - len(combined)
    print(f"  CSV rows: {before} total → {len(combined)} kept ({dropped} dropped — not in 4-class scheme)")
    print(combined["Emotions"].value_counts().to_string())
    return combined


def build_blob_name(speaker: str, dialogue_id: int, utterance_id: int) -> str:
    """Return the canonical Azure blob name for a MELD clip."""
    safe_speaker = speaker.strip().replace(" ", "_")
    return f"{MELD_PREFIX}{safe_speaker}_dia{dialogue_id}_utt{utterance_id}.wav"


def convert_mp4_to_wav_bytes(mp4_path: Path, target_sr: int = 22050) -> bytes:
    """Use ffmpeg to convert mp4 → wav in memory.  Returns raw WAV bytes."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(mp4_path),
                "-ar", str(target_sr),
                "-ac", "1",          # mono
                "-vn",               # drop video stream
                tmp_path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Upload MELD dataset to Azure Blob Storage")
    parser.add_argument("--meld-dir", required=True, help="Path to the extracted MELD.Raw directory")
    parser.add_argument("--dry-run",  action="store_true", help="Print what would be uploaded without uploading")
    args = parser.parse_args()

    meld_dir = Path(args.meld_dir)
    if not meld_dir.exists():
        print(f"ERROR: MELD directory not found: {meld_dir}")
        sys.exit(1)

    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not connection_string:
        print("ERROR: AZURE_STORAGE_CONNECTION_STRING not set in .env")
        sys.exit(1)

    blob_service = BlobServiceClient.from_connection_string(connection_string)

    # ── 1. Load and merge label CSVs ─────────────────────────────────────────
    print("\n[1/3] Loading label CSVs …")
    label_df = load_and_merge_csvs(meld_dir)

    # ── 2. Upload the combined label CSV ─────────────────────────────────────
    print("\n[2/3] Uploading meld_labels.csv …")
    csv_bytes = label_df.to_csv(index=False).encode("utf-8")
    if not args.dry_run:
        blob_client = blob_service.get_blob_client(
            container=CONTAINER_NAME, blob=f"{MELD_PREFIX}meld_labels.csv"
        )
        blob_client.upload_blob(io.BytesIO(csv_bytes), overwrite=True)
        print(f"  Uploaded {MELD_PREFIX}meld_labels.csv  ({len(csv_bytes):,} bytes)")
    else:
        print(f"  [DRY RUN] Would upload {MELD_PREFIX}meld_labels.csv")

    # ── 3. Convert and upload audio clips ────────────────────────────────────
    print("\n[3/3] Converting mp4 → wav and uploading …")

    ok = skipped = errors = 0

    for _, row in label_df.iterrows():
        split    = row["split"]
        dia_id   = int(row["Dialogue_ID"])
        utt_id   = int(row["Utterance_ID"])
        speaker  = str(row["Speaker"])

        mp4_filename = f"dia{dia_id}_utt{utt_id}.mp4"
        mp4_path     = meld_dir / SPLIT_DIRS[split] / mp4_filename
        blob_name    = build_blob_name(speaker, dia_id, utt_id)

        if not mp4_path.exists():
            print(f"  [WARN] Audio not found: {mp4_path}")
            skipped += 1
            continue

        if args.dry_run:
            print(f"  [DRY RUN] Would upload {blob_name}")
            ok += 1
            continue

        try:
            wav_bytes  = convert_mp4_to_wav_bytes(mp4_path)
            blob_client = blob_service.get_blob_client(container=CONTAINER_NAME, blob=blob_name)
            blob_client.upload_blob(io.BytesIO(wav_bytes), overwrite=True)
            ok += 1
            if ok % 100 == 0:
                print(f"  … {ok} files uploaded so far")
        except Exception as exc:
            print(f"  [ERROR] {mp4_filename}: {exc}")
            errors += 1

    print(f"\n✓ Done.  Uploaded: {ok}  |  Skipped: {skipped}  |  Errors: {errors}")
    print(f"  All files are at:  {MELD_PREFIX}  in container '{CONTAINER_NAME}'")


if __name__ == "__main__":
    main()
