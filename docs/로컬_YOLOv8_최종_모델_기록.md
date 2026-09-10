# Smart Cart 로컬 YOLOv8 최종 모델 기록

갱신일: 2026-09-10
최종 적용 버전: **V7**

## 최종 서비스 적용 상태

- 서비스 모델 파일: `backend/models/smart_cart.pt`
- 공개 포트폴리오 저장소에는 V7 서비스 모델 하나만 포함한다.
- 원본 작업 폴더의 V7 백업과 서비스 파일의 SHA-256 값이 일치하는 것을 확인했다.
- 백엔드의 `/api/detect` API는 `backend/local_detector.py`를 통해 위의 로컬 모델을 사용한다.
- Roboflow API나 API 키는 서비스 실행에 사용하지 않는다.

## V7 학습 정보

- 모델: YOLOv8n
- 학습 장치: NVIDIA GeForce RTX 4060 Laptop GPU (CUDA)
- 학습 설정: 50 epochs, 이미지 크기 640 x 640
- 데이터 위치: `BOX/model-training/dataset` (GitHub 업로드 제외)
- 클래스 10개: `apple`, `bread`, `carrot`, `egg`, `galic`, `l_onion`, `onion`, `raw_pork`, `shrimp`, `sliced_ham`
- 원본 이미지 그룹 분할: Train 620 / Validation 78 / Test 78 (약 80 / 10 / 10)

### 데이터 전처리와 증강

Roboflow에서 자동 방향 보정 후 640 x 640으로 변환했다. 각 원본 이미지에 대해 다음 증강을 적용한 변형본 3개가 생성되어 학습 데이터로 사용되었다.

- 좌우 반전 50%
- 90도 회전(없음/시계/반시계 동일 확률)
- -15도 ~ +15도 회전
- 가로·세로 -10도 ~ +10도 shear

### V7 검증 성능

| 기준 | Epoch | Precision | Recall | mAP@50 | mAP@50:95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| mAP@50 최고 | 44 | 78.26% | 81.41% | **83.96%** | 71.13% |
| mAP@50:95 최고 | 49 | 73.53% | 86.73% | 83.27% | **72.06%** |

`best.pt` 기반의 V7 모델을 최종 서비스 모델로 선택했다. 로컬 확인 이미지에서도 계란 클래스가 70.8%로 검출되는 것을 확인했다.

## V8 실험 기록과 최종 선택 근거

V8은 원본 이미지 그룹이 겹치지 않도록 75 / 12.5 / 12.5 비율로 다시 나눠 실험했다. 독립 Test 96장 중 라벨 형식 오류 이미지 1장을 제외한 95장(257 객체) 평가에서 Precision 71.56%, Recall 70.16%, mAP@50 72.78%, mAP@50:95 49.80%였다.

V7을 V8의 Test 세트에 다시 적용하면 mAP@50 94.39%, mAP@50:95 82.75%가 나왔지만, V7은 해당 원본 데이터 전체로 이미 학습한 이력이 있어 일반화 성능 비교 지표로 사용하지 않는다. 최종 서비스는 사용 중 확인된 안정성과 기존 V7 검증 성능을 기준으로 V7을 채택했다. V8 학습 결과와 데이터는 `BOX/model-training`에 보관하며 서비스에는 연결하지 않는다.

## 실행 방법

이미 서버가 실행 중이었다면 모델을 다시 읽도록 백엔드를 한 번 종료한 뒤 재실행한다.

```bash
cd /mnt/c/folder/AUTO_CART
source .venv/bin/activate
python3 backend/app.py
```

프론트엔드는 별도 터미널에서 실행한다.

```bash
cd /mnt/c/folder/AUTO_CART/frontend
npm run dev -- --host 0.0.0.0
```

브라우저에서 `http://127.0.0.1:5173`을 연다. 백엔드가 정상이라면 `http://127.0.0.1:8000/api/health`에서 로컬 모델 상태를 확인할 수 있다.

## GitHub 공개 저장소 주의

`BOX/`, 학습 데이터, 중간 실험 모델, `.venv/`는 `.gitignore`로 제외되어 있다. 단, 포트폴리오 재현을 위해 최종 V7 서비스 모델 `backend/models/smart_cart.pt` 하나만 GitHub와 Docker 이미지에 포함한다.
