from datasets import Dataset, Audio, Features, Value
import pandas as pd
from pathlib import Path

root = Path("/home/aleph/tesis/frontend/public/assets/stimuli_norm")  # your folder
df   = pd.read_json(root / "metadata.jsonl", lines=True)

# Turn the file paths into absolute paths so push_to_hub can find the files
df["file_path"] = df["file_path"].apply(lambda p: str(root / p))

ds = Dataset.from_pandas(
    df,
    features=Features({
        "file_path": Audio(),            # decoded on the fly
        "speaker_id": Value("string"),
        "duration_ms": Value("float32"),
        "split": Value("string"),
        "rating": Value("float32"),
    }),
    preserve_index=False
)

# Split column → DatasetDict so the Hub viewer shows tabs
from datasets import DatasetDict

splits = {
    split: ds.filter(lambda ex, split=split: ex["split"] == split)
    for split in ("train", "val", "test")
}

datasetdict = DatasetDict(splits)

datasetdict.push_to_hub("asosawelford/es-TTS-subjective-naturalness", private=False)  # set private=True if desired
