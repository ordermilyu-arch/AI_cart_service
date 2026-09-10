"""Evaluate the active local YOLO model on a chosen dataset split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="Evaluate Smart Cart local YOLO model")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=("val", "test"))
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", default="smart-cart-evaluation")
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "backend" / "models" / "smart_cart.pt",
        help="Model weights to evaluate",
    )
    args = parser.parse_args()
    data_yaml = args.data if args.data.is_absolute() else ROOT / args.data
    model_path = args.model if args.model.is_absolute() else ROOT / args.model
    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(data_yaml),
        split=args.split,
        device=args.device,
        imgsz=640,
        project=str(ROOT / "BOX" / "model-training" / "runs"),
        name=args.name,
        exist_ok=True,
        plots=False,
        verbose=False,
        workers=0,
    )
    print(json.dumps({
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
        "map50": round(float(metrics.box.map50), 4),
        "map50_95": round(float(metrics.box.map), 4),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
