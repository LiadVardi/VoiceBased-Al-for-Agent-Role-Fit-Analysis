import os
import librosa
import soundfile as sf
import numpy as np

# Input and output directories
RAW_DIR = "raw_audio"
CLEAN_DIR = "clean_audio"

# Create output directory if it does not exist
os.makedirs(CLEAN_DIR, exist_ok=True)

# Target sample rate for all audio files
TARGET_SR = 16000

# Process each audio file in the input directory
for filename in os.listdir(RAW_DIR):
    if not filename.endswith(".wav"):
        continue  # Skip non-audio files

    filepath = os.path.join(RAW_DIR, filename)

    # Load audio file, convert to mono, resample to TARGET_SR
    audio, sr = librosa.load(filepath, sr=TARGET_SR, mono=True)

    # Normalize audio volume to range [-1, 1]
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val

    # Remove leading and trailing silence
    audio_trimmed, _ = librosa.effects.trim(audio, top_db=20)

    # Save processed audio file
    output_path = os.path.join(CLEAN_DIR, filename)
    sf.write(output_path, audio_trimmed, TARGET_SR)

print("Done preprocessing audio files.")
