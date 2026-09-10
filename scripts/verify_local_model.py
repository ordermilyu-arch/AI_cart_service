"""Run the trained local model against one exported test image."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from local_detector import model_status, predict  # noqa: E402


def main():
    image_dir = ROOT / "BOX" / "model-training" / "dataset" / "test" / "images"
    image = next(image_dir.glob("*.jpg"), None)
    if image is None:
        raise SystemExit(f"테스트 이미지를 찾을 수 없습니다: {image_dir}")
    predictions = sorted(predict(image.read_bytes()), key=lambda item: item["confidence"], reverse=True)
    print(model_status())
    print(f"test image: {image.name}")
    for prediction in predictions[:3]:
        print(f"{prediction['class']}: {prediction['confidence']:.1%}")


if __name__ == "__main__":
    main()
