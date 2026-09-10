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

최종 V7 가중치가 `backend/models/smart_cart.pt`에 포함되어 있으므로, Roboflow API 키나 별도의 모델 다운로드 없이 바로 로컬 테스트를 할 수 있습니다.

## 처음 실행하기

### 실행 전 설치할 것

- [Git](https://git-scm.com/downloads)
- Python 3.10 이상
- Node.js LTS (npm 포함)
- 카메라가 연결된 컴퓨터

NVIDIA GPU가 없어도 실행은 가능합니다. 다만 YOLO가 CPU로 동작하므로 상품 인식 속도는 더 느릴 수 있습니다.

### 1. 프로젝트 받기

터미널에서 한 번만 실행합니다.

```bash
git clone https://github.com/ordermilyu-arch/AI_cart_service.git
cd AI_cart_service
```

### 2. 백엔드 실행

터미널 창 1에서 아래 명령을 순서대로 실행하고, 이 창은 켜 둡니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 backend/app.py
```

마지막 줄에 아래처럼 보이면 정상입니다.

```text
Smart Cart API: http://127.0.0.1:8000
```

Windows PowerShell에서는 가상환경 활성화 명령만 아래처럼 다릅니다.

```powershell
.venv\Scripts\Activate.ps1
```

### 3. 프론트엔드 실행

새 터미널 창 2에서 실행합니다. 백엔드 터미널은 종료하지 않습니다.

```bash
cd /mnt/c/folder/AI_cart_service/frontend
npm ci
npm run dev
```

### 4. 카메라 테스트

1. 브라우저에서 `http://127.0.0.1:5173`을 엽니다.
2. 카메라 권한 요청이 나오면 `허용`을 누릅니다.
3. 상품을 카메라에 비춥니다.
4. 신뢰도 75% 이상이고 같은 상품이 2번 연속 감지되면 장바구니에 추가됩니다.

카메라가 보이지 않으면 브라우저 주소창 왼쪽의 카메라 권한을 `허용`으로 바꾼 뒤 새로고침합니다. `localhost`와 `127.0.0.1`은 로컬 카메라 테스트가 가능한 안전한 주소입니다.

## 학습 모델

V7은 YOLOv8n 기반의 식재료 10종 인식 모델입니다. 학습 과정과 검증 성능은 [최종 모델 기록](docs/로컬_YOLOv8_최종_모델_기록.md)에 정리했습니다.

학습·평가 재현용 스크립트는 `scripts/`에 포함했습니다. 학습 데이터셋과 중간 실험 가중치는 제외하고, 최종 서비스 V7 가중치 `backend/models/smart_cart.pt` 하나만 포함합니다.

## 기술 스택

| 구분 | 사용 기술 |
| --- | --- |
| Frontend | React, Vite, Lucide React |
| Backend | Python, SQLite, 표준 HTTP 서버 |
| AI | Ultralytics YOLOv8n, PyTorch CUDA |
| 실행 환경 | 로컬 브라우저, Python API |
