# VAST-ReID
**A Low-Light Benchmark Dataset for Person Re-Identification with Visual and Attribute-Rich Semantic Tracking**

## Overview
VAST-ReID는 야간·저조도 환경에서의 사람 재식별(Person Re-ID) 성능을 평가하기 위한 벤치마크 데이터셋 및 평가 파이프라인입니다.
기존 ReID 데이터셋은 대부분 밝은 환경에서 촬영되어 실제 야간 CCTV 상황을 반영하지 못합니다.
본 프로젝트는 조도 단계별 이미지, 의복·소지품 속성 라벨, 트래킹 기반 시퀀스 정보를 함께 제공하여
저조도 ReID 연구의 기준점을 마련하는 것을 목표로 합니다.

## Goals
- 조도 등급(Normal / Low / Extreme-Low)별 ReID 데이터셋 구축
- 의복·소지품 중심의 속성(Attribute) 및 시맨틱 캡션 제공
- 기존 ReID 모델의 저조도 성능 저하 정량 분석
- 저조도 개선 기법 및 속성 융합을 통한 성능 회복 검증

## Dataset Structure
```
VAST-ReID/
├── train/
├── query/
├── gallery/
└── annotations/
    ├── train.json
    ├── query.json
    └── gallery.json
```
각 샘플은 person_id, camera_id, track_id, illumination, bbox, attributes, caption 정보를 포함합니다.

## Pipeline
1. 영상 수집 (실촬영 + 공개 데이터셋 기반 저조도 합성)
2. 전처리 (프레임 추출, 조도 분류, 얼굴 블러)
3. 검출·트래킹 (YOLO + ByteTrack)
4. 어노테이션 (ID 검수, VLM 기반 속성 라벨 + 사람 검수)
5. 벤치마크 평가 (OSNet, TransReID, CLIP-ReID)
6. 데모 (Gradio)

## Evaluation Metrics
- ReID: Rank-1, Rank-5, mAP, mINP
- Tracking: IDF1, HOTA
- Attribute: 속성별 정확도, mA

## Tech Stack
Python, PyTorch, torchreid, Ultralytics YOLO, ByteTrack, CLIP, Zero-DCE, Retinexformer, Gradio

## Ethics & Privacy
- 실촬영 데이터는 참여자 동의를 받아 수집
- 모든 얼굴 영역 블러 처리
- 성별·나이·인종 등 민감 속성은 라벨에서 제외
- 연구·교육 목적에 한해 사용

## Schedule
| 주차 | 기간 | 내용 |
|---|---|---|
| 1 | 10/14 ~ 10/20 | 기획, 환경 구축 |
| 2 | 10/21 ~ 10/27 | 데이터 수집·전처리 |
| 3 | 10/28 ~ 11/03 | 검출·트래킹·어노테이션 |
| 4 | 11/04 ~ 11/10 | 벤치마크 실험 |
| 5 | 11/11 ~ 11/17 | 데모·발표 준비 |
| - | 11/18 | 최종 발표 |

## Getting Started
```bash
git clone https://github.com/<user>/VAST-ReID.git
cd VAST-ReID
pip install -r requirements.txt
python tools/synthesize_lowlight.py --src data/market1501 --out data/vast
python tools/evaluate.py --model osnet --split low
python app.py   # 데모 실행
```

## Author
- 송민호 (동양미래대학교)
