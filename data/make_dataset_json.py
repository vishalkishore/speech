import argparse
import io
import json
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import snapshot_download


DEFAULT_DATASET = "JacobLinCool/VoiceBank-DEMAND-16k"
DEFAULT_VAL_SPEAKERS = ("p226", "p287")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download VoiceBank-DEMAND-16k parquet shards from Hugging Face, "
        "export wav files, and generate train/valid/test JSON manifests."
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"Hugging Face dataset id. Default: {DEFAULT_DATASET}",
    )
    parser.add_argument(
        "--output_root",
        default="data",
        help="Root directory where split folders and JSON manifests will be written.",
    )
    parser.add_argument(
        "--val_speakers",
        nargs="+",
        default=list(DEFAULT_VAL_SPEAKERS),
        help="Speaker IDs from the HF train split to reserve for validation.",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        help="Optional Hugging Face cache/snapshot directory.",
    )
    return parser.parse_args()


def ensure_split_dirs(root: Path, split: str) -> tuple[Path, Path]:
    clean_dir = root / split / "clean"
    noisy_dir = root / split / "noisy"
    clean_dir.mkdir(parents=True, exist_ok=True)
    noisy_dir.mkdir(parents=True, exist_ok=True)
    return clean_dir, noisy_dir


def write_pair(clean_audio, noisy_audio, sr: int, clean_path: Path, noisy_path: Path) -> None:
    sf.write(clean_path, clean_audio, sr)
    sf.write(noisy_path, noisy_audio, sr)


def write_json(path: Path, items: list[str]) -> None:
    with path.open("w") as f:
        json.dump(items, f, indent=2)


def decode_audio_field(field):
    """Decode an audio cell from parquet into (array, sampling_rate)."""
    if isinstance(field, dict):
        if "array" in field and "sampling_rate" in field:
            return field["array"], field["sampling_rate"]
        if field.get("bytes") is not None:
            return sf.read(io.BytesIO(field["bytes"]))
        if field.get("path"):
            return sf.read(field["path"])
    if isinstance(field, (bytes, bytearray)):
        return sf.read(io.BytesIO(field))
    raise ValueError(f"Unsupported audio field format: {type(field)}")


def iter_split_items(snapshot_dir: Path, split: str):
    split_files = sorted((snapshot_dir / "data").glob(f"{split}-*.parquet"))
    if not split_files:
        raise FileNotFoundError(f"No parquet shards found for split '{split}' in {snapshot_dir / 'data'}")

    for parquet_path in split_files:
        parquet_file = pq.ParquetFile(parquet_path)
        for batch in parquet_file.iter_batches(batch_size=64):
            for item in batch.to_pylist():
                yield item


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    val_speakers = set(args.val_speakers)
    snapshot_dir = Path(
        snapshot_download(
            repo_id=args.dataset,
            repo_type="dataset",
            allow_patterns=["data/*.parquet"],
            cache_dir=args.cache_dir,
        )
    )

    manifests = {
        "train_clean": [],
        "train_noisy": [],
        "valid_clean": [],
        "valid_noisy": [],
        "test_clean": [],
        "test_noisy": [],
    }

    # Process train split into train + valid.
    for item in iter_split_items(snapshot_dir, "train"):
        uid = item["id"]
        speaker = uid.split("_")[0]
        clean_audio, sr = decode_audio_field(item["clean"])
        noisy_audio, noisy_sr = decode_audio_field(item["noisy"])
        if sr != noisy_sr:
            raise ValueError(f"Sampling-rate mismatch for {uid}: clean={sr}, noisy={noisy_sr}")
        split = "valid" if speaker in val_speakers else "train"

        clean_dir, noisy_dir = ensure_split_dirs(output_root, split)
        clean_path = clean_dir / f"{uid}.wav"
        noisy_path = noisy_dir / f"{uid}.wav"

        write_pair(clean_audio, noisy_audio, sr, clean_path, noisy_path)

        manifests[f"{split}_clean"].append(str(clean_path))
        manifests[f"{split}_noisy"].append(str(noisy_path))

    # Process test split.
    clean_dir, noisy_dir = ensure_split_dirs(output_root, "test")
    for item in iter_split_items(snapshot_dir, "test"):
        uid = item["id"]
        clean_audio, sr = decode_audio_field(item["clean"])
        noisy_audio, noisy_sr = decode_audio_field(item["noisy"])
        if sr != noisy_sr:
            raise ValueError(f"Sampling-rate mismatch for {uid}: clean={sr}, noisy={noisy_sr}")
        clean_path = clean_dir / f"{uid}.wav"
        noisy_path = noisy_dir / f"{uid}.wav"

        write_pair(clean_audio, noisy_audio, sr, clean_path, noisy_path)

        manifests["test_clean"].append(str(clean_path))
        manifests["test_noisy"].append(str(noisy_path))

    for name, items in manifests.items():
        write_json(output_root / f"{name}.json", items)

    print(
        f"Train: {len(manifests['train_noisy'])}, "
        f"Valid: {len(manifests['valid_noisy'])}, "
        f"Test: {len(manifests['test_noisy'])}"
    )


if __name__ == "__main__":
    main()
