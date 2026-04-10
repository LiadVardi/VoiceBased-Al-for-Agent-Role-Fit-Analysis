"""
split_audio.py  —  Split audio files into fixed-length segments
================================================================
Cuts each MP3 / WAV file into segments of a fixed duration
(default: 4 seconds) and saves them to an output folder
(default: Cropped_audio/).

HOW TO RUN:
-----------
# Split every file in Voxonics_audio/ into 4-second segments:
python src/split_audio.py --input Voxonics_audio

# Custom segment length and output folder:
python src/split_audio.py --input Voxonics_audio --seconds 5 --output My_Cropped

NAMING CONVENTION:
------------------
Input:   393941H.mp3  (12s long)
Output:  Cropped_audio/393941H_part1.wav   (0–4s)
         Cropped_audio/393941H_part2.wav   (4–8s)
         Cropped_audio/393941H_part3.wav   (8–12s)

The original emotion letter in the filename is preserved so the
pipeline can still read the label from the filename.
"""

import argparse
import sys
from pathlib import Path

import librosa
import soundfile as sf

SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}
DEFAULT_SR          = 22050   # matches config.TARGET_SR
DEFAULT_SEGMENT_SEC = 4.0     # cut every 4 seconds
MIN_SEGMENT_SEC     = 1.0     # discard leftover shorter than this


def split_file(audio_path: Path, output_dir: Path,
               segment_sec: float, sr: int) -> int:
    """
    Split a single audio file into fixed-length segments.
    Returns the number of segments saved.
    """
    try:
        audio, _ = librosa.load(str(audio_path), sr=sr, mono=True)
    except Exception as exc:
        print(f"  [ERROR] Could not load {audio_path.name}: {exc}")
        return 0

    segment_samples = int(segment_sec * sr)
    min_samples     = int(MIN_SEGMENT_SEC * sr)
    total_samples   = len(audio)

    saved   = 0
    start   = 0
    part_no = 1

    while start < total_samples:
        end     = min(start + segment_samples, total_samples)
        segment = audio[start:end]

        if len(segment) < min_samples:
            break   # leftover too short — discard

        out_name = f"{audio_path.stem}_part{part_no}.wav"
        out_path = output_dir / out_name
        sf.write(str(out_path), segment, sr)

        saved   += 1
        part_no += 1
        start   += segment_samples

    return saved


def main():
    parser = argparse.ArgumentParser(
        description="Split audio files into fixed-length segments."
    )
    parser.add_argument(
        "--input",  "-i",
        default="Voxonics_audio",
        help="Folder containing the audio files to split (default: Voxonics_audio)",
    )
    parser.add_argument(
        "--output", "-o",
        default="Cropped_audio",
        help="Output folder for the segments (default: Cropped_audio)",
    )
    parser.add_argument(
        "--seconds", "-s",
        type=float,
        default=DEFAULT_SEGMENT_SEC,
        help=f"Length of each segment in seconds (default: {DEFAULT_SEGMENT_SEC})",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=DEFAULT_SR,
        help=f"Target sample rate in Hz (default: {DEFAULT_SR})",
    )
    args = parser.parse_args()

    input_dir  = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"ERROR: Input folder not found: {input_dir}")
        sys.exit(1)

    audio_files = [
        f for f in sorted(input_dir.iterdir())
        if f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not audio_files:
        print(f"No audio files found in '{input_dir}'")
        print(f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input    : {input_dir}  ({len(audio_files)} files)")
    print(f"Output   : {output_dir}")
    print(f"Segment  : {args.seconds}s per piece")
    print(f"SR       : {args.sr} Hz")
    print()

    total_saved = 0
    for audio_path in audio_files:
        duration = librosa.get_duration(path=str(audio_path))
        n = split_file(audio_path, output_dir,
                       segment_sec=args.seconds, sr=args.sr)
        total_saved += n
        print(f"  {audio_path.name:<30}  {duration:.1f}s  →  {n} segments")

    print()
    print(f"✓ Done.  {total_saved} segments saved to '{output_dir}/'")


if __name__ == "__main__":
    main()
