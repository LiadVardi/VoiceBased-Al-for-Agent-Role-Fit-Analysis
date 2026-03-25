import json

nb_path = r"c:\Users\orbit_24ts2or\OneDrive\שולחן העבודה\Emotion-Recognition\VoiceBasedAi.ipynb"

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        source = cell.get("source", [])
        
        # 1. Update imports
        if len(source) > 0 and "import io\n" in source[0] or "import os\n" in source[0] or "from azure.storage.blob" in "".join(source):
            sys_path_line = "import sys\nsys.path.append('src')\n"
            if sys_path_line not in source:
                source.insert(0, sys_path_line)
                
            for i, line in enumerate(source):
                if line.startswith("from audio_pipeline import"):
                    source[i] = line.replace("from audio_pipeline", "from src.audio_pipeline")
                elif line.startswith("from config import"):
                    source[i] = line.replace("from config", "from src.config")
                elif line.startswith("from manifest_builder import"):
                    source[i] = line.replace("from manifest_builder", "from src.manifest_builder")
                elif line.startswith("from speaker_splitter import"):
                    source[i] = line.replace("from speaker_splitter", "from src.speaker_splitter")
                elif line.startswith("from augmentation import"):
                    source[i] = line.replace("from augmentation", "from src.augmentation")

        # 2. Update CSV reads
        for i, line in enumerate(source):
            if 'pd.read_csv("train_features_ready_for_model.csv")' in line:
                source[i] = line.replace('"train_features_ready_for_model.csv"', '"data/csv/train_features_ready_for_model.csv"')
            elif 'pd.read_csv("val_features_ready_for_model.csv")' in line:
                source[i] = line.replace('"val_features_ready_for_model.csv"', '"data/csv/val_features_ready_for_model.csv"')
            elif 'pd.read_csv("test_features_ready_for_model.csv")' in line:
                source[i] = line.replace('"test_features_ready_for_model.csv"', '"data/csv/test_features_ready_for_model.csv"')

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Updated Notebook successfully!")
