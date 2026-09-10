"""Create a leakage-safe 75/12.5/12.5 dataset split from the downloaded YOLO export.

The current Roboflow export has multiple `.rf.*` variants per source image.
This script keeps every variant of a source image in the same split.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "BOX" / "model-training" / "dataset"
TARGET = ROOT / "BOX" / "model-training" / "dataset-v8"
SOURCE_SUFFIX = re.compile(r"\.rf\.[^.]+\.jpg$", re.IGNORECASE)
SPLIT_RATIOS = {"train": 0.75, "valid": 0.125, "test": 0.125}


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare Smart Cart V8 data split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Replace only generated train/valid/test folders")
    return parser.parse_args()


def source_key(image: Path) -> str:
    return SOURCE_SUFFIX.sub("", image.name)


def label_classes(label: Path) -> list[int]:
    classes = []
    for line in label.read_text(encoding="utf-8").splitlines():
        if line.strip():
            classes.append(int(line.split()[0]))
    return classes


def copy_pair(image: Path, split: str):
    label = image.parent.parent / "labels" / f"{image.stem}.txt"
    image_target = TARGET / split / "images" / image.name
    label_target = TARGET / split / "labels" / label.name
    image_target.parent.mkdir(parents=True, exist_ok=True)
    label_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image, image_target)
    shutil.copy2(label, label_target)


def main():
    args = parse_args()
    if not (SOURCE / "data.yaml").is_file():
        raise SystemExit(f"원본 데이터셋을 찾을 수 없습니다: {SOURCE}")

    generated_paths = [TARGET / split for split in SPLIT_RATIOS]
    if any(path.exists() for path in generated_paths) and not args.force:
        raise SystemExit("V8 분할이 이미 있습니다. 다시 만들려면 --force를 사용하세요.")
    for path in generated_paths:
        if path.exists():
            shutil.rmtree(path)

    groups: dict[str, list[Path]] = defaultdict(list)
    for split in ("train", "valid", "test"):
        for image in (SOURCE / split / "images").glob("*.jpg"):
            label = image.parent.parent / "labels" / f"{image.stem}.txt"
            if label.is_file():
                groups[source_key(image)].append(image)

    # Stratify by primary class while keeping all variants of one source together.
    class_groups: dict[int, list[tuple[str, list[Path]]]] = defaultdict(list)
    for key, images in groups.items():
        classes = [class_id for image in images for class_id in label_classes(image.parent.parent / "labels" / f"{image.stem}.txt")]
        primary_class = min(classes) if classes else -1
        class_groups[primary_class].append((key, sorted(images)))

    rng = random.Random(args.seed)
    assignments = {split: [] for split in SPLIT_RATIOS}
    for records in class_groups.values():
        rng.shuffle(records)
        total = len(records)
        train_count = round(total * SPLIT_RATIOS["train"])
        valid_count = round(total * SPLIT_RATIOS["valid"])
        assignments["train"].extend(records[:train_count])
        assignments["valid"].extend(records[train_count:train_count + valid_count])
        assignments["test"].extend(records[train_count + valid_count:])

    # Train keeps all existing variants for stronger learning. Evaluation uses one representative per source.
    for split, records in assignments.items():
        for _, images in records:
            selected = images if split == "train" else [images[0]]
            for image in selected:
                copy_pair(image, split)

    names = [
        "apple", "bread", "carrot", "egg", "galic",
        "l_onion", "onion", "raw_pork", "shrimp", "sliced_ham",
    ]
    data_yaml = "\n".join([
        f"path: {TARGET.resolve()}",
        "train: train/images",
        "val: valid/images",
        "test: test/images",
        "", "nc: 10", f"names: {names}", "",
    ])
    (TARGET / "data.yaml").write_text(data_yaml, encoding="utf-8")
    summary = {
        "seed": args.seed,
        "source_groups": {split: len(records) for split, records in assignments.items()},
        "image_files": {
            split: len(list((TARGET / split / "images").glob("*.jpg")))
            for split in SPLIT_RATIOS
        },
        "ratios": SPLIT_RATIOS,
    }
    (TARGET / "split-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
