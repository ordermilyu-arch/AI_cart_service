"""Train the Smart Cart YOLO model from the locally exported Roboflow dataset.

Run from the project root:
    python3 scripts/train_local_model.py
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "BOX" / "model-training" / "runs"
MODEL_OUTPUT = ROOT / "backend" / "models" / "smart_cart.pt"


def parse_args():
    parser = argparse.ArgumentParser(description="Train Smart Cart's local YOLO model")
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "BOX" / "model-training" / "dataset" / "data.yaml",
        help="Path to a Roboflow YOLOv8 data.yaml file",
    )
    parser.add_argument("--run-name", default="smart-cart", help="Name of the training result folder")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO input image size")
    parser.add_argument("--batch", type=int, default=-1, help="Batch size; -1 chooses automatically")
    parser.add_argument("--device", default=None, help="CUDA device such as 0, or cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_yaml = args.data if args.data.is_absolute() else ROOT / args.data
    if not dataset_yaml.is_file():
        raise SystemExit(f"데이터셋 설정 파일이 없습니다: {dataset_yaml}")
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit(
            "ultralytics가 필요합니다. python3 -m pip install -r requirements.txt 를 실행하세요."
        ) from error

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO("yolov8n.pt")
    train_args = {
        "data": str(dataset_yaml), "epochs": args.epochs, "imgsz": args.imgsz,
        "batch": args.batch, "project": str(RUNS_DIR), "name": args.run_name,
        "exist_ok": True, "pretrained": True,
        # Product-camera augmentation: no 90° rotation or shear.
        "fliplr": 0.5, "flipud": 0.0,
        "degrees": 10.0, "translate": 0.08, "scale": 0.20,
        "shear": 0.0, "perspective": 0.0005,
        "hsv_h": 0.015, "hsv_s": 0.5, "hsv_v": 0.3,
        "mosaic": 0.5, "close_mosaic": 10, "erasing": 0.1,
    }
    if args.device is not None:
        train_args["device"] = args.device
    model.train(**train_args)
    best_model = RUNS_DIR / args.run_name / "weights" / "best.pt"
    if not best_model.is_file():
        raise SystemExit(f"학습이 끝났지만 best.pt를 찾을 수 없습니다: {best_model}")
    shutil.copy2(best_model, MODEL_OUTPUT)
    print(f"로컬 모델 저장 완료: {MODEL_OUTPUT}")


if __name__ == "__main__":
    main()
