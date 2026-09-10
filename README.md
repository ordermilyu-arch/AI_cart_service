# AI Cart Service

카메라로 식재료를 인식해 장바구니에 담고, 보유 재료 기반의 레시피를 추천하는 스마트 카트 웹 서비스입니다.

![기술 구성](https://img.shields.io/badge/Frontend-React%20%2B%20Vite-61DAFB) ![기술 구성](https://img.shields.io/badge/Backend-Python-3776AB) ![기술 구성](https://img.shields.io/badge/AI-YOLOv8-00FFFF)

## 핵심 기능

- 브라우저 카메라를 상시 표시하고 약 1.8초 간격으로 프레임을 YOLOv8에 전달
- 신뢰도 75% 이상이고 동일 상품이 2회 연속 확인될 때만 장바구니에 추가
- 장바구니 수량·삭제·포장봉투·바코드 결제 완료 흐름
- 구매한 재료가 가장 많이 포함된 레시피 우선 추천
- React/Vite 프론트엔드와 Python/SQLite/YOLOv8 백엔드 분리

## 서비스 구조

```text
웹 브라우저 카메라
  → React/Vite 화면
  → Python API /api/detect
  → 직접 학습한 V7 YOLOv8 모델
  → 상품명·신뢰도 반환
  → 2회 연속 확인 후 장바구니 반영
```

배포 환경에서는 GitHub push가 Vercel 프론트엔드와 Cloud Run GPU 백엔드를 각각 갱신합니다. 최종 V7 가중치는 이 저장소의 `backend/models/smart_cart.pt`에 포함되어 Docker 이미지와 함께 배포됩니다.

## 로컬 실행

### 1. 백엔드

V7 모델 파일을 `backend/models/smart_cart.pt`에 준비한 뒤 실행합니다.

```bash
cd /mnt/c/folder/AI_cart_service
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 backend/app.py
```

### 2. 프론트엔드

별도 터미널에서 실행합니다.

```bash
cd /mnt/c/folder/AI_cart_service/frontend
npm install
npm run dev -- --host 0.0.0.0
```

`http://127.0.0.1:5173`을 열고 카메라 권한을 허용합니다.

## 배포

배포에 필요한 파일은 이미 포함되어 있습니다.

- `backend/Dockerfile`: CUDA 기반 YOLOv8 API 컨테이너
- `cloudbuild.yaml`: GitHub → Cloud Build → Cloud Run GPU 자동 배포
- `frontend`를 Vercel에 연결하고 `VITE_API_BASE_URL` 환경변수에 Cloud Run의 `/api` 주소를 설정

계정 설정부터 배포 확인까지의 전체 순서는 [웹배포 실시간 카메라 가이드](docs/웹배포_실시간카메라_가이드.md)를 참고합니다.

## 학습 모델

V7은 YOLOv8n 기반의 식재료 10종 인식 모델입니다. 학습 과정과 검증 성능은 [최종 모델 기록](docs/로컬_YOLOv8_최종_모델_기록.md)에 정리했습니다.

학습·평가 재현용 스크립트는 `scripts/`에 포함했습니다. 학습 데이터셋과 중간 실험 가중치는 제외하고, 최종 서비스 V7 가중치 `backend/models/smart_cart.pt` 하나만 포함합니다.

## 공개 저장소 보안 원칙

- `BOX/`, 학습 데이터셋, `.venv/`, `node_modules/` 제외
- 최종 서비스 V7 가중치 하나만 공개하고, 중간 실험 가중치는 제외
- API 키·개인 환경 파일 제외
- Roboflow API 키와 Cloud Storage 접근 키를 사용하지 않음

## 기술 스택

| 구분 | 사용 기술 |
| --- | --- |
| Frontend | React, Vite, Lucide React |
| Backend | Python, SQLite, 표준 HTTP 서버 |
| AI | Ultralytics YOLOv8n, PyTorch CUDA |
| 배포 | Vercel, Google Cloud Run GPU, Cloud Build, Cloud Storage |
