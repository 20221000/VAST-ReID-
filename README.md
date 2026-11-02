# VAST-ReID
**A Low-Light Benchmark Dataset for Person Re-Identification with Visual and Attribute-Rich Semantic Tracking**

## Overview
VAST-ReID는 야간·저조도 환경에서 사람 재식별(Person Re-ID) 성능을 평가하기 위한 벤치마크 데이터셋과 평가 파이프라인입니다.
기존 ReID 데이터셋은 대부분 밝은 환경에서 촬영되어 실제 야간 CCTV 상황을 반영하지 못합니다.
이 프로젝트는 공개 데이터셋(Market-1501)에 물리 기반 저조도를 합성해 **조도 3단계 데이터셋**을 만들고,
**저조도에서 ReID가 왜 무너지는지 분석**하며, **학습 없이 성능을 회복하는 방법**을 검증합니다.

![low-light grid](docs/figures/lowlight_grid.jpg)

## 핵심 결과 (요약)
> 모든 수치는 실제 실행 결과이며 `results/*.csv`에 있습니다. OSNet(Market-1501 사전학습)으로 추론만 했습니다.

| 조건 | Rank-1 | mAP |
|---|---|---|
| 밝은 query → 밝은 gallery (Normal) | 94.5 | 82.6 |
| 어두운 query → 어두운 gallery (Low) | 48.5 | 22.4 |
| **어두운 query → 밝은 gallery (야간 CCTV → 주간 기록)** | **23.0** | **18.7** |
| + 저조도 개선 전처리 (Zero-DCE 등) | 개선 없음 | 개선 없음 |
| + 속성 재순위화 (목격자 진술 + 예측 속성) | 34.3 | 28.3 |
| + hubness 완화 (CSLS) | 43.1 | 33.8 |
| **+ CSLS + 속성 재순위화** | **52.0** | **40.7** |

- 어두워지면 Rank-1이 94.5 → 48.5로 거의 절반이 됩니다.
- 사진을 밝게 만드는 전처리는 노이즈까지 증폭시켜 오히려 성능을 떨어뜨렸습니다.
- 원인을 분석해 보니, 노이즈를 옷의 잔무늬로 착각해 **서로 다른 사람의 어두운 query가 같은 몇몇 gallery 이미지로 몰리고**(hubness) 있었습니다.
- hub 이미지의 점수를 깎는 CSLS와 속성 정보를 더하면, 야간→주간 검색 Rank-1이 **23.0 → 52.0**으로 회복됩니다.
- 표의 "어두운 → 밝은" 아래 행들은 재순위화를 위해 코사인 유사도 기준입니다. 유클리드 거리 기준 베이스라인은 25.5 / 20.3입니다.

---

## 원래 계획과 바뀐 점

| | 원래 계획 | 현재 |
|---|---|---|
| 데이터 | 실촬영 + 합성 | **공개 데이터셋(Market-1501) + 저조도 합성만** (실촬영 생략) |
| 실행 환경 | Google Colab | 로컬 PC (WSL2 Ubuntu, RTX 5080, PyTorch cu128) |
| 모델 | 일부 직접 학습 | **딥러닝 학습 없음**: 사전학습 모델 추론 + 속성용 작은 선형 분류기만 |
| 속성 라벨 | VLM 라벨링 + 사람 검수 | 공개 속성 라벨(Market-1501 Attribute)에서 6개 선택 |
| 개선 방법 | 저조도 개선(Zero-DCE) 중심 | Zero-DCE는 실패 확인 → **속성 재순위화 + hubness 완화** |
| AI API | 라벨링 도구 | 호출 수가 적은 곳에만 사용 (아래 참고) |
| 트래킹 | YOLO + ByteTrack 데모 | MOT17 저조도 합성으로 정량 평가 예정 (진행 전) |

---

## 시스템 구조

```
Market-1501 원본 사진
   │ ① 저조도 합성 (빛 감소 + 센서 노이즈)            tools/synthesize_lowlight.py
   ▼
normal / low / extreme 3단계 데이터셋                    data/vast/{level}/{train,query,gallery}
   │
   ├─② OSNet (사전학습) → 외형 특징 벡터              vastreid/reid.py, tools/evaluate.py
   │
   ├─③ CLIP (사전학습) → 이미지 특징 → 선형 분류기 → 속성 확률
   │                                                     vastreid/clip_attr.py, tools/attr_probe.py
   │
   └─④ 최종 점수 = 외형 유사도 − β·(hub 점수) + λ·(속성 일치도)
                                                         tools/rerank.py, tools/hubness.py
                     │
                     ▼
        Rank-1 / mAP / mINP → results/*.csv → Gradio 데모 (demo/app.py)
```

### 각 단계의 원리

**① 저조도 합성**: 사진을 단순히 어둡게 편집하지 않고, 카메라가 어두운 곳에서 찍을 때처럼 만듭니다.
선형(빛의 양) 공간에서 노출을 줄이고(Low ×0.15~0.30, Extreme ×0.03~0.08), 빛이 적을 때 생기는 센서 노이즈(Poisson + Gaussian),
약한 블러, 색 틀어짐, JPEG 압축을 더합니다. 난수 시드가 고정되어 있어 몇 번을 실행해도 같은 결과가 나옵니다.

**② 외형 특징 (OSNet)**: 사람 사진 한 장을 512개의 숫자(벡터)로 바꾸는 모델입니다. 같은 사람이면 벡터가 비슷하게 나오도록 이미 학습되어 있습니다.
검색은 query 벡터와 가장 가까운 gallery 벡터를 순서대로 찾는 일입니다.

**③ 속성 예측 (CLIP + 선형 분류기)**: CLIP은 이미지와 문장을 같은 공간의 벡터로 바꿉니다.
"흰 셔츠를 입은 사람"이라는 문장과 이미지가 가까우면 흰 셔츠일 가능성이 높다고 봅니다(zero-shot).
정확도를 높이려고 CLIP 특징 위에 속성마다 로지스틱 회귀(선형 분류기)를 하나씩 붙였습니다. CLIP 자체는 학습하지 않습니다.
속성 6개: 상의 색, 하의 색, 소매, 하의 종류, 가방, 모자 (**성별·나이·머리 길이는 제외**).

**④ 점수 조합**
- **외형 유사도**: OSNet이 본 생김새
- **속성 일치도**: 상의 색, 가방 등이 얼마나 같은가. 어두워져도 색과 형태는 일부 남아 외형 정보를 보완합니다.
- **hub 점수 (CSLS)**: 그 gallery 사진이 어두운 사진들 전반과 두루 비슷한가. 두루 비슷하면 특정 사람이 아니라 노이즈에 반응한 것이므로 점수를 깎습니다.
- **λ, β 선택**: 테스트셋은 보지 않고 train 이미지로 만든 검증 분할에서만 고릅니다.

### 저조도에서 ReID가 무너지는 이유: hubness
어두운 query 3,368장의 검색 결과(top-10)를 분석했습니다 (`docs/DEVLOG.md`).

| query 조도 | top-10에 나온 서로 다른 gallery 이미지 | 자주 나온 상위 20장의 점유율 |
|---|---|---|
| Normal | 12,100 | 0.6% |
| Low | 6,564 | 22.2% |
| Extreme | 176 | 96.1% |

Low에서는 이미지 한 장이 query 938개의 top-10에 등장했습니다. 이런 hub 이미지는 대부분 땡땡이·체크·꽃무늬처럼
**잔무늬가 촘촘한 옷**이었습니다. 모델이 센서 노이즈를 옷의 무늬로 착각한 것입니다. CSLS를 적용하면 Low→Normal에서
상위 20장 점유율이 22.2% → 3.9%로 줄고 Rank-1이 23.0 → 43.1로 오릅니다.

![hub images](docs/figures/hub_gallery.jpg)

---

## AI API 활용 (Grok, GPT)

수업에서 지원받은 AI API는 반 전체가 함께 쓰는 키이므로, **호출 수가 적은 곳에만** 사용했습니다.

| 용도 | 하는 일 | 호출량 |
|---|---|---|
| 비전 LLM 저조도 실험 | 사진을 보여주고 6개 속성을 JSON으로 답하게 해 조도별 정확도를 비교 | 50명 × 3조도 × 2모델 = 300회 |
| 데모 텍스트 검색 (선택, 로컬 전용) | "흰 반팔에 백팩 멘 사람" → 속성 JSON | 검색 1번에 1회 |

**ReID 검색 자체를 API로 하지 않은 이유**: query 3,368장을 gallery 15,913장과 모두 비교하면 수천만 번 호출이 필요해
비현실적이고, 표준 지표(Rank-1, mAP)로 비교할 수도 없습니다. 대량 계산은 로컬 모델이, 사람의 말을 이해하는 일과
소규모 비교 실험은 API가 맡도록 나눴습니다.

**토큰 절약 장치** (`vastreid/llm.py`): 같은 요청은 캐시에서 재사용, 실행당 호출 상한, 호출 없이 횟수만 세는 dry-run,
사용량 로그(`outputs/api_usage.csv`). API 키는 로컬 `.env`에만 두며 레포와 공개 데모에는 포함하지 않습니다.

### 비전 LLM도 어두운 사진은 못 알아본다
속성 평균 정확도(%), 테스트 50명 × 3조도 (`results/attr_acc_vlm.csv`). "모르겠다(unknown)" 답변은 오답 처리.

| 모델 | Normal | Low | Extreme |
|---|---|---|---|
| Grok (`grok-4.20-0309-non-reasoning`) | **85.6** | 75.7 | 36.7 |
| GPT (`gpt-4.1-mini`) | 83.2 | 69.9 | 32.5 |
| CLIP + 선형 분류기 | 78.5 | **78.6** | **63.8** |
| CLIP zero-shot | 52.4 | 44.5 | 21.2 |

밝을 때는 비전 LLM이 가장 정확하지만, 극저조도에서는 크게 무너집니다. 선형 분류기는 흔한 값(예: 모자 안 씀)을 잘 맞히는
효과도 포함되어 있어 해석에 주의가 필요합니다. 표본이 작아 오차 범위가 큽니다.

---

## 상세 결과

### 저조도 개선 전처리 — 효과 없음 (`results/enhance.csv`, `results/enhance_v2.csv`)
Rank-1 (%), 같은 조도끼리 검색. 게이팅 = 어두운 이미지(평균 밝기 < 0.25)에만 적용.

| 방법 | Normal | Low | Extreme |
|---|---|---|---|
| 없음 | **94.5** | **48.5** | 2.5 |
| 감마 (전부 적용) | 90.1 | 43.1 | 2.9 |
| CLAHE (전부 적용) | 85.8 | 42.8 | 3.5 |
| Zero-DCE (전부 적용) | 62.2 | 32.4 | 2.5 |
| Zero-DCE (게이팅) | 91.8 | 31.5 | 2.5 |
| 노이즈 제거 + Zero-DCE (게이팅) | 91.4 | 27.1 | 2.9 |

밝기를 올리면 센서 노이즈도 함께 증폭됩니다 (`docs/figures/enhance_grid.jpg`).

### 재순위화 조합 (`results/hubness.csv`)
Rank-1 / mAP (%), 코사인 유사도 기준.

| 방법 | Low → Low | Low → Normal | Extreme → Normal |
|---|---|---|---|
| 외형만 | 48.1 / 22.6 | 23.0 / 18.7 | 0.0 / 0.2 |
| + 예측 속성 | 48.8 / 23.9 | 31.6 / 25.4 | 0.0 / 0.4 |
| + 목격자 속성 | 49.7 / 25.2 | 34.3 / 28.3 | 1.7 / 3.0 |
| + CSLS | 50.7 / 25.5 | 43.1 / 33.8 | 0.4 / 0.6 |
| + CSLS + 예측 속성 | 52.2 / 26.9 | 49.2 / 38.2 | 1.3 / 1.3 |
| **+ CSLS + 목격자 속성** | **53.7 / 28.4** | **52.0 / 40.7** | **9.3 / 6.5** |

- 예측 속성: query·gallery 모두 CLIP + 선형 분류기 예측 (완전 자동)
- 목격자 속성: query는 사람이 진술한 인상착의(정답 속성으로 대체), gallery는 예측. 목격자 진술 + CCTV 기록 검색 시나리오
- 참고 상한선(정답 속성을 양쪽에 사용): Low→Normal 61.0 / 58.4 (`results/rerank_oracle.csv`)

---

## 데모
`python demo/app.py` → http://127.0.0.1:7860

1. **이미지 검색**: 저조도 query를 고르거나 업로드 → 주간 gallery Top-10, 정답/오답 표시. 목격자 속성을 넣으면 재순위화.
2. **텍스트 검색**: 인상착의 문장 → 속성 추출 → 속성이 일치하는 사람 검색.
3. **벤치마크**: `results/*.csv`를 읽어 그래프로 표시 (수치 하드코딩 없음).

---

## Getting Started
```bash
git clone https://github.com/20221000/VAST-ReID-.git
cd VAST-ReID-
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements.txt
# torchreid는 공식 레포에서 설치 (C 컴파일러 필요: sudo apt install build-essential)
uv pip install --no-build-isolation git+https://github.com/KaiyangZhou/deep-person-reid.git

# 데이터 (Kaggle 토큰은 환경변수로만 전달)
export KAGGLE_API_TOKEN=<your_token>
kaggle datasets download iiierie/market-1501-v15-09-15 -p data/raw --unzip

# 저조도 합성: data/vast/{normal,low,extreme}/{train,query,gallery}
python tools/synthesize_lowlight.py --src data/raw/Market-1501-v15.09.15 --out data/vast \
    --grid docs/figures/lowlight_grid.jpg

# 실험 (사전학습 모델 추론 + 속성용 선형 분류기만, 딥러닝 학습 없음)
python tools/evaluate.py --out results/baseline.csv
python tools/evaluate.py --enhance none gamma clahe zerodce --out results/enhance.csv
python tools/build_annotations.py
python tools/clip_attributes.py && python tools/attr_probe.py
python tools/rerank.py --attr-source witness-probe
python tools/hubness.py

# 데모
python demo/app.py
```
가중치: OSNet은 torchreid Model Zoo의 Market-1501 학습 가중치(`weights/osnet_x1_0_market1501.pth`),
Zero-DCE는 공식 레포의 `Epoch99.pth`(`weights/zerodce_epoch99.pth`)를 사용합니다.

## 레포 구조
```
configs/      저조도 합성·API 설정
tools/        실행 스크립트 (합성, 평가, 속성, 재순위화, hubness, VLM 실험)
vastreid/     공용 모듈 (데이터, 지표, 특징 추출, 개선 전처리, CLIP 속성, LLM 래퍼)
annotations/  ID별 속성 6개 + 캡션 (person_attributes.json)
results/      실험 결과 CSV
demo/         Gradio 데모
docs/         개발 일지(DEVLOG.md), 그림
```

## 평가 지표
- ReID: Rank-1, Rank-5, mAP, mINP (Market-1501 표준 프로토콜: 같은 사람·같은 카메라 제외, junk 제외)
- 속성: 속성별 정확도, 평균 정확도
- 트래킹(예정): IDF1, HOTA

## 한계
- 저조도는 합성입니다. 실제 야간 CCTV는 카메라 내부에서 노이즈를 제거하므로, 합성 노이즈가 실제보다 거칠 수 있습니다.
- Extreme 등급은 현재 설정에서 대부분의 방법이 무작위 수준에 가까워, 합성 강도 재조정을 검토 중입니다.
- λ, β 검증에 쓴 train 이미지는 OSNet이 학습한 이미지라 검증 점수가 과대평가됩니다(보수적인 λ 선택 쪽으로 치우침).
- 비전 LLM 실험은 50명 샘플이라 오차 범위가 큽니다.

## Ethics & Privacy
- 실촬영 없이 공개 연구용 데이터셋(Market-1501)만 사용
- 성별·나이·인종과, 성별을 간접적으로 드러낼 수 있는 머리 길이는 라벨과 모델에서 제외
- 비전 LLM에는 성별·나이를 판단하지 말라고 명시
- 연구·교육 목적에 한해 사용

## Development Log
개발 과정의 결정, 시행착오, 실험 결과, 배운 점은 [docs/DEVLOG.md](docs/DEVLOG.md)에 기록합니다.

## Author
- 송민호 (동양미래대학교)
