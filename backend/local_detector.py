"""Smart Cart의 로컬 YOLO 모델 추론 모듈."""

from __future__ import annotations

import io
import os
from functools import lru_cache
from pathlib import Path
from threading import Lock

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = ROOT / "backend" / "models" / "smart_cart.pt"
MODEL_PATH = Path(os.environ.get("SMART_CART_MODEL_PATH", DEFAULT_MODEL_PATH))
MODEL_ID = "local/smart-cart-yolov8"
_INFERENCE_LOCK = Lock()


class LocalModelError(RuntimeError):
    """Raised when the local model cannot be used yet."""


@lru_cache(maxsize=1)
def get_model():
    if not MODEL_PATH.is_file():
        raise LocalModelError(
            f"서비스 모델 파일이 없습니다: {MODEL_PATH}. "
            "backend/models/smart_cart.pt를 배치하세요."
        )
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise LocalModelError(
            "ultralytics가 설치되지 않았습니다. python3 -m pip install -r requirements.txt 를 실행하세요."
        ) from error
    return YOLO(str(MODEL_PATH))


def model_status() -> dict:
    return {
        "engine": "local-yolov8",
        "model_id": MODEL_ID,
        "model_path": str(MODEL_PATH),
        "model_exists": MODEL_PATH.is_file(),
    }


def predict(image_bytes: bytes, confidence: float = 0.01) -> list[dict]:
    """Return predictions in the same shape previously returned by Roboflow."""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as error:
        raise LocalModelError("카메라 이미지를 읽을 수 없습니다.") from error

    model = get_model()
    with _INFERENCE_LOCK:
        result = model.predict(image, conf=confidence, imgsz=640, verbose=False)[0]

    predictions = []
    names = result.names
    for box in result.boxes:
        class_id = int(box.cls.item())
        x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
        predictions.append({
            "class": str(names[class_id]),
            "class_id": class_id,
            "confidence": float(box.conf.item()),
            "x": (x1 + x2) / 2,
            "y": (y1 + y2) / 2,
            "width": x2 - x1,
            "height": y2 - y1,
        })
    return predictions
