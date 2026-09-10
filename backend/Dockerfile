# Cloud Run GPU에서 실행할 Smart Cart YOLOv8 API 이미지
# 공개 저장소의 최종 V7 가중치가 backend/models/smart_cart.pt에 포함된다.
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY backend ./backend
COPY ingredients.db recipes.db ./

ENV PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8080 \
    SMART_CART_MODEL_PATH=/tmp/smart_cart.pt

EXPOSE 8080
CMD ["python", "backend/app.py"]
