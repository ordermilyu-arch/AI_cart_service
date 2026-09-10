# 모델 학습·평가 스크립트

- `train_local_model.py`: YOLOv8 모델 학습
- `prepare_v8_dataset.py`: 원본 그룹 기준 데이터셋 분할 준비
- `evaluate_local_model.py`: Test 세트 성능 평가
- `verify_local_model.py`: 이미지 한 장으로 추론 확인

학습 데이터와 중간 실험 가중치는 공개 배포본에 포함하지 않습니다. 최종 서비스 V7 가중치 하나는 `backend/models/smart_cart.pt`에 포함합니다.
