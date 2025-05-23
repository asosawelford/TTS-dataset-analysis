#!/usr/bin/env python3
"""
make_manifest.py – build metadata.jsonl for a speech dataset
whose split/rating are defined in split/{train,val,test}.csv.

Expected CSV schema in split_dir
--------------------------------
stimuli,mos
spk1_0001.wav,4.8
spk2_0003.wav,2.0
...

Output JSONL columns
--------------------
file_path   – relative path from data_root
speaker_id  – parent folder name
duration_ms – float, duration in milliseconds
split       – "train", "val" or "test"
rating      – MOS 1–5 (float)

Example
-------
python make_manifest.py \
    --data_root /path/to/stimuli_norm \
    --split_dir /path/to/stimuli_norm/split \
    --out metadata.jsonl
"""

import argparse
import csv
import json
import sys
import wave
from pathlib import Path
from typing import Dict, Tuple

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}


# --------------------------------------------------------------------------- #
# Utility: compute duration (ms)
# --------------------------------------------------------------------------- #
def audio_duration_ms(path: Path) -> float:
    if path.suffix.lower() == ".wav":
        with wave.open(str(path), "rb") as w:
            frames = w.getnframes()
            sr = w.getframerate()
        return frames * 1000.0 / sr

    try:
        import soundfile as sf

        with sf.SoundFile(str(path)) as f:
            return len(f) * 1000.0 / f.samplerate
    except ImportError:
        raise RuntimeError(
            f"Cannot read {path.suffix}; install `pip install soundfile` or convert to WAV."
        )


# --------------------------------------------------------------------------- #
# Read split CSVs → mapping[file_path] = (split, rating)
# --------------------------------------------------------------------------- #
def read_split_tables(split_dir: Path) -> Dict[str, Tuple[str, float]]:
    mapping: Dict[str, Tuple[str, float]] = {}
    for split_name in ("train", "val", "test"):
        csv_path = split_dir / f"{split_name}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Expected {csv_path}")
        with csv_path.open(newline="", encoding="utf8") as f:
            reader = csv.DictReader(f)
            if {"stimuli", "mos"} - set(reader.fieldnames):
                raise ValueError(
                    f"{csv_path} must contain 'stimuli' and 'mos' columns."
                )
            for row in reader:
                rel_path = row["stimuli"].strip().replace("\\", "/")
                rating = float(row["mos"])
                mapping[rel_path] = (split_name, rating)
    return mapping


# --------------------------------------------------------------------------- #
# Generator that yields manifest rows
# --------------------------------------------------------------------------- #
def build_manifest(data_root: Path, split_map: Dict[str, Tuple[str, float]]):
    root = data_root.resolve()
    for audio in root.rglob("*"):
        if audio.suffix.lower() not in AUDIO_EXTS or not audio.is_file():
            continue

        rel_path = str(audio.relative_to(root)).replace("\\", "/")
        if rel_path not in split_map:
            print(f"⚠️  {rel_path} not in split tables – skipped", file=sys.stderr)
            continue

        split, rating = split_map[rel_path]
        speaker_id = audio.parent.name
        duration = audio_duration_ms(audio)

        yield {
            "file_path": rel_path,
            "speaker_id": speaker_id,
            "duration_ms": duration,
            "split": split,
            "rating": rating,
        }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Create JSONL manifest.")
    parser.add_argument("--data_root", required=True, type=Path)
    parser.add_argument("--split_dir", required=True, type=Path,
                        help="Folder with train/val/test CSVs")
    parser.add_argument("--out", default="metadata.jsonl", type=Path)
    args = parser.parse_args()

    split_map = read_split_tables(args.split_dir)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with args.out.open("w", encoding="utf8") as f:
        for row in build_manifest(args.data_root, split_map):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"Wrote {n_written} entries to {args.out} ✓", file=sys.stderr)


if __name__ == "__main__":
    main()
