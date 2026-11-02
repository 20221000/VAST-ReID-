# VAST-ReID 인수인계 문서 (HANDOFF)

> 작성일: 2026-11-02 (월) · 최종 발표: **2026-11-18 (수)** · 남은 기간 약 16일
> 이 문서는 새 Claude Code 세션에서 프로젝트를 이어서 진행하기 위한 인수인계 자료입니다.
> 새 세션에서는 이 파일을 레포 루트에 두고 "HANDOFF.md 읽고 이어서 진행해줘"라고 시작하면 됩니다.

---

## 0. 새 세션 Claude에게 (작업 원칙)

- 사용자: 송민호 (동양미래대학교 학생). 수업 과제이며 **1인 진행**, 결과물은 노션 계획서 + Public GitHub 레포 + 11/18 발표·시연.
- 남은 시간이 짧으므로 **"돌아가는 최소 버전 먼저, 확장은 나중"** 원칙으로 진행할 것.
- 실험 수치는 반드시 실제 실행 결과만 사용할 것. 디자인 시안에 들어간 수치(예: Rank-1 88/61/34)는 **화면 예시용 가상값**이므로 보고서·README·발표에 쓰면 안 됨.
- GPU가 없을 가능성이 높음 → 학습은 Google Colab 기준으로 설계하고, 로컬은 CPU로도 돌아가는 추론·평가 위주.
- 연구 윤리 원칙 유지: 성별·나이·인종 등 민감 속성은 라벨/모델에서 제외, 실촬영 시 얼굴 블러 + 동의.

---

## 1. 프로젝트 개요

**VAST-ReID: A Low-Light Benchmark Dataset for Person Re-Identification with Visual and Attribute-Rich Semantic Tracking**

- **문제:** 기존 Person Re-ID 데이터셋(Market-1501, MSMT17 등)은 주간·밝은 환경 위주라 야간·저조도 CCTV 상황에서 모델 성능이 크게 떨어짐.
- **목표:**
  1. 조도 등급(Normal / Low / Extreme-Low)별 ReID 벤치마크 데이터셋 구축
  2. 의복·소지품 중심 속성(Attribute) 라벨 + 시맨틱 캡션 제공
  3. 기존 ReID 모델의 저조도 성능 저하를 정량 측정
  4. 저조도 개선(Zero-DCE) 및 속성 융합으로 성능 회복 가능성 검증
  5. Gradio 데모 웹으로 시연

---

## 2. 지금까지 한 일 (완료)

| 항목 | 내용 | 위치 |
|---|---|---|
| 기획·계획서 | 시스템 구조, UI/UX 예시, AI 활용 방안, 주간 일정, README 초안 작성 | 사용자 노션 (제출 완료) |
| 시스템 설계 | 파이프라인, 기술 스택, JSON 스키마, 윤리 기준 확정 | 본 문서 §4 |
| UI/UX 디자인 | Claude Design으로 데모 화면 3종 제작 (흰색 테마로 재디자인 완료) | https://claude.ai/artifact/2xcnHyxe1PpNsR7qkg6Emp |
| 진행상황 보고 | 10/28 기준 진행상황 보고서 작성·제출 | 사용자 노션 |

**아직 안 한 것 (코드·데이터 작업 전부):** GitHub 레포 생성, 개발 환경, 데이터 다운로드·합성, 실험, 데모 구현. 즉 **구현은 0%** 상태에서 시작.

---

## 3. 확정된 의사결정 (10/28 진행보고에서 변경된 사항 포함)

1. **실촬영은 생략**하고 공개 데이터셋(Market-1501)에 저조도를 **합성**해서 데이터셋을 만든다.
2. 속성은 핵심 6개로 제한: `upper_color`, `lower_color`, `upper_type`(또는 sleeve), `sleeve`, `bag`, `hat`.
3. 실험은 **OSNet 중심** (torchreid, 경량). CLIP-ReID / TransReID는 시간 남으면.
4. 트래킹(YOLO + ByteTrack)은 **축소**: 데모용 짧은 영상에서 시연하는 정도로만. 정량 지표(IDF1/HOTA)는 선택 사항.
5. 데모는 Gradio, 화면 구성은 디자인 시안 3화면을 따른다.

### 새 세션에서 추가로 검토할 제안 (사용자 확인 필요)

- **Market-1501 Attribute 라벨 활용:** Market-1501에는 공개된 속성 어노테이션(Lin et al., "Market-1501_Attribute", ID 단위 27개 속성)이 있음. 여기서 의복·소지품 속성만 골라 쓰면 VLM 라벨링 없이 속성 라벨을 확보할 수 있어 일정 단축에 매우 유리. **gender, age 속성은 반드시 제외.** VLM(Gemini 등)은 시맨틱 캡션 생성과 일부 검증에만 사용.
- **DukeMTMC-reID는 사용하지 말 것** (원저자가 공개 철회한 데이터셋).
- Market-1501은 정지 이미지 데이터셋이라 "트래킹 시퀀스"가 없음 → 발표 때 한계로 명시하고, 트래킹은 데모 영상에서만 보여주는 것으로 범위를 정리.

---

## 4. 시스템 설계

### 4-1. 파이프라인

```
Market-1501 (원본, Normal)
   │
   ▼
[저조도 합성] tools/synthesize_lowlight.py
   ├─ Normal (원본)
   ├─ Low          (노출 배율 0.15~0.30)
   └─ Extreme-Low  (노출 배율 0.03~0.08)
   │   * 선형공간 변환 → 밝기 스케일 → Poisson-Gaussian 노이즈 → 감마 복원 → 약한 블러/색 틀어짐
   ▼
[메타데이터] tools/build_annotations.py
   ├─ person_id, camera_id, illumination, mean_luminance
   ├─ 속성 6개 (Market-1501 Attribute에서 매핑)
   └─ caption (템플릿 기반 생성, 선택적으로 VLM 보강)
   ▼
[벤치마크] tools/evaluate.py
   ├─ OSNet (ImageNet/Market 사전학습)
   ├─ + Zero-DCE 전처리
   ├─ + 속성 융합 (재순위화)
   └─ Rank-1, Rank-5, mAP, mINP  ×  조도 등급별
   ▼
[데모] app.py (Gradio, 3탭)
```

### 4-2. 평가 프로토콜 (제안)

- **Protocol A (핵심):** Normal train set으로 학습한 모델을 query/gallery를 각 조도 등급으로 합성한 테스트셋에서 평가 → "저조도 성능 저하" 측정.
- **Protocol B (선택):** train set에 조도 등급을 섞어 학습(augmentation) 후 동일하게 평가 → "저조도 데이터로 학습하면 얼마나 회복되나" 측정.
- 합성은 시드를 고정해 재현 가능하게 하고, 각 이미지의 노출 배율·노이즈 파라미터를 메타데이터에 기록.

### 4-3. 속성 융합 (가장 간단한 버전부터)

1. **재순위화(우선 구현):** `score = cos_sim(f_q, f_g) + λ · attr_agreement(q, g)`. attr_agreement는 일치하는 속성 비율. λ는 val에서 0~0.5 탐색.
2. (시간 남으면) OSNet 위에 속성 분류 헤드를 붙여 멀티태스크 학습.

### 4-4. 기술 스택

Python 3.10+, PyTorch, torchreid (OSNet), OpenCV, NumPy, Zero-DCE/Zero-DCE++ (사전학습 가중치), Ultralytics YOLO + ByteTrack(데모용), CLIP(open_clip, 텍스트 검색), Gradio, Google Colab.

### 4-5. 데이터 스키마 (annotations/*.json 항목 1개)

```json
{
  "image_id": "0002_c1s1_000451_03__low",
  "source_image": "bounding_box_test/0002_c1s1_000451_03.jpg",
  "person_id": 2,
  "camera_id": 1,
  "split": "query",
  "illumination": "low",
  "exposure_scale": 0.22,
  "noise_sigma": 0.03,
  "mean_luminance": 0.11,
  "attributes": {
    "upper_color": "black",
    "lower_color": "blue",
    "upper_type": "long_sleeve",
    "sleeve": "long",
    "bag": "backpack",
    "hat": false
  },
  "caption": "검은 긴팔 상의와 파란 하의를 입고 백팩을 멘 사람"
}
```

> 합성 데이터이므로 계획서의 `lux_estimate` 대신 `mean_luminance`(0~1)와 합성 파라미터를 기록. `track_id`, `frame`, `bbox`는 Market-1501에 없으므로 제외(데모 영상 트래킹 결과에만 사용).

---

## 5. 권장 레포 구조

```
VAST-ReID/
├── README.md
├── HANDOFF.md              # 이 문서
├── requirements.txt
├── configs/
│   └── lowlight.yaml       # 조도 등급별 합성 파라미터
├── data/                   # .gitignore (원본·생성 데이터는 커밋 금지)
│   ├── market1501/
│   └── vast/{normal,low,extreme}/{train,query,gallery}/
├── annotations/            # 생성된 JSON (용량 작으면 커밋)
├── tools/
│   ├── synthesize_lowlight.py
│   ├── build_annotations.py
│   ├── evaluate.py
│   ├── enhance.py          # Zero-DCE 적용
│   └── make_figures.py     # 결과 그래프
├── vastreid/               # 공용 모듈 (데이터셋 로더, 지표, 속성 재순위화)
├── notebooks/
│   └── colab_train_eval.ipynb
├── results/                # 실험 결과 CSV·그래프
├── demo/
│   └── app.py              # Gradio 3탭
└── docs/
    └── design/             # 디자인 시안 캡처 이미지
```

---

## 6. 남은 일정 (11/2 기준 재조정)

| 기간 | 목표 | 완료 기준 |
|---|---|---|
| **11/2(월) ~ 11/4(수)** | 레포·환경·데이터 | Public 레포 생성 + README 업로드, Market-1501 확보, 저조도 합성 스크립트 완성, 3등급 데이터셋 v0.1 생성, 샘플 이미지 그리드 확인 |
| **11/5(목) ~ 11/7(토)** | 메타데이터·베이스라인 | 속성 매핑 + JSON 생성, OSNet으로 조도별 Rank-1/mAP 측정 (Protocol A) → **첫 결과표 확보** |
| **11/8(일) ~ 11/11(수)** | 개선 실험 | Zero-DCE 전처리 전후 비교, 속성 재순위화 실험, (선택) Protocol B, 결과 CSV·그래프 |
| **11/12(목) ~ 11/14(토)** | 데모 | Gradio 3탭(이미지 검색 / 텍스트 검색 / 벤치마크) 구현, (선택) YOLO+ByteTrack 데모 영상 |
| **11/15(일) ~ 11/17(화)** | 마무리 | README 최종화(실제 결과 반영), 발표 자료, 실패 사례 정리, 리허설 |
| **11/18(수)** | 최종 발표 | 시연 + 결과 발표 |

**최소 성공 기준 (이것만 되면 발표 가능):** 저조도 합성 데이터셋 + OSNet 조도별 성능표 + Zero-DCE 비교 1개 + Gradio 이미지 검색 탭.

---

## 7. 바로 시작할 작업 체크리스트

- [ ] GitHub Public 레포 `VAST-ReID` 생성 (사용자 GitHub 아이디 확인 필요)
- [ ] `.gitignore`에 `data/`, 가중치 파일(`*.pth`, `*.pt`) 추가
- [ ] `requirements.txt` 작성, 가상환경 또는 Colab 세팅
- [ ] Market-1501 다운로드 (공식 배포처에서 사용자가 직접 받아야 할 수 있음 → 경로만 인자로 받게 작성)
- [ ] Market-1501 Attribute 어노테이션 확보 (GitHub `vana77/Market-1501_Attribute`)
- [ ] `synthesize_lowlight.py` 작성: 입력 폴더 → 3등급 출력, 시드 고정, 파라미터 로그
- [ ] 합성 결과를 눈으로 확인할 샘플 그리드 이미지 생성 (발표 자료에도 사용)
- [ ] `build_annotations.py`: 속성 매핑(민감 속성 제외) + 템플릿 캡션
- [ ] `evaluate.py`: torchreid OSNet 사전학습 가중치로 조도별 평가 → `results/baseline.csv`

---

## 8. 데모 화면 명세 (디자인 시안 기준)

디자인 시안: https://claude.ai/artifact/2xcnHyxe1PpNsR7qkg6Emp
스타일: 흰색/밝은 회색 바탕, 메인색 짙은 남색 `#1B2A4A`, 강조색 앰버 `#F5B547`, 폰트 IBM Plex Sans KR. 인물 썸네일은 저조도 느낌을 주는 어두운 박스.

1. **이미지 검색 탭**
   - 좌측: 쿼리 이미지 업로드, 조도 필터(전체/Low/Extreme), 속성 필터(상의색·가방), 모델 선택, 저조도 개선 토글, 검색 버튼
   - 우측: Top-10 결과 그리드 (유사도, 카메라, 조도, 정답/오답 배지), 오답 실패 원인 표시
2. **텍스트 검색 탭**
   - 자연어 입력("흰색 상의, 검은 바지, 백팩을 멘 사람") → 속성 칩 추출 → CLIP 유사도 순 결과
   - 결과 선택 시 우측에 시맨틱 캡션 + 속성 상세
3. **벤치마크 탭**
   - Rank-1/mAP 전환, 모델 × 조도 막대그래프, 핵심 지표 카드(하락폭·개선 효과), 실패 원인 분포
   - 수치는 `results/*.csv`에서 읽어 표시 (하드코딩 금지)

Gradio는 시안을 100% 재현할 필요 없음. `gr.Tabs` 3개 + `gr.Gallery` + 커스텀 CSS로 색·폰트만 맞추면 충분.

---

## 9. 리스크와 대응

| 리스크 | 대응 |
|---|---|
| Market-1501 다운로드 지연 | 사용자에게 수동 다운로드 요청, 그동안 합성 스크립트를 샘플 이미지로 개발 |
| GPU 부족 | 사전학습 OSNet으로 추론·평가만 진행(학습 생략 가능), 학습은 Colab |
| Zero-DCE 가중치/코드 호환 문제 | 대안으로 CLAHE·감마 보정 같은 고전 기법을 비교군으로 사용 |
| 시간 부족 | §6 "최소 성공 기준"만 먼저 완성, 텍스트 검색·트래킹은 선택 |
| 합성 저조도의 현실성 비판 | 발표에서 한계로 명시, 실제 야간 이미지 몇 장으로 정성 비교 |

---

## 10. 참고: README 초안 핵심 (노션 계획서에 작성한 내용)

- Overview / Goals / Dataset Structure / Pipeline / Evaluation Metrics / Tech Stack / Ethics & Privacy / Schedule / Getting Started / Author(송민호, 동양미래대학교)
- 계획서 대비 바뀐 점(실촬영 생략, 합성 기반, 트래킹 축소, `lux_estimate` → `mean_luminance`)을 README에 반영할 것.
- Getting Started 예시 명령:
  ```bash
  pip install -r requirements.txt
  python tools/synthesize_lowlight.py --src data/market1501 --out data/vast --seed 42
  python tools/build_annotations.py --vast data/vast --attr data/market_attribute.mat
  python tools/evaluate.py --model osnet_x1_0 --level all --out results/baseline.csv
  python tools/evaluate.py --model osnet_x1_0 --level all --enhance zerodce --out results/zerodce.csv
  python demo/app.py
  ```

---

## 11. 세션 2 기록 (2026-11-02) — 위 내용보다 우선함

### 변경된 결정
- **실행 환경:** Colab 미사용. 로컬 PC(WSL2 Ubuntu, RTX 5080, CUDA 13 드라이버)에서 전부 진행. PyTorch는 **cu128 이상** 빌드 필요 (`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`).
- **Python:** 시스템 Python 3.14에는 pip/venv가 없어 `uv`(~/.local/bin)로 **Python 3.12 `.venv`** 구성. 패키지 설치는 `uv pip install --python .venv/bin/python <pkg>`.
- **데이터:** Kaggle API로 다운로드. 토큰은 환경변수 `KAGGLE_API_TOKEN`으로만 전달 (코드·레포·파일에 저장 금지).
  - Market-1501: `iiierie/market-1501-v15-09-15` → `data/raw/Market-1501-v15.09.15/` (train 12,936 / query 3,368 / gallery 19,732, 표준과 일치)
  - 속성: github `vana77/Market-1501_Attribute` (`market_attribute.mat`), gender/age 제외
  - MOT17: `wenhoujinjust/mot-17` (약 5GB, 트래킹 단계에서 2~3개 시퀀스만 사용)
- **트래킹 정량 평가 추가 (§3-4 변경):** MOT17에 같은 저조도 합성 적용 → ByteTrack vs BoT-SORT+OSNet vs +Zero-DCE, 조도별 HOTA/IDF1 (TrackEval).
- **데모:** 공개용은 Hugging Face Spaces(샘플 갤러리만), 발표 시연은 로컬.
- **작업 원칙:** 집 PC이므로 작업당 1시간 이내, 처음엔 작은 샘플로 시간·온도 측정 후 전체 실행, 밤샘 실행 금지.

### 완료
- 레포 뼈대: `.gitignore`, `requirements.txt`, `configs/lowlight.yaml`
- `tools/synthesize_lowlight.py`: reid/mot 두 모드, 시드 고정(결정적 재현 확인), 이미지별 파라미터·`mean_luminance`를 `synthesis_log.csv`에 기록. MOT 모드는 시퀀스 단위로 노출 고정, 노이즈만 프레임별로 생성하며 gt/seqinfo도 복사.
- 저조도 합성 v0.1 생성: `data/vast/{normal,low,extreme}/{train,query,gallery}` (36,036장 × 3등급, 13초, 442MB)
  - mean_luminance 평균: normal 0.396 / low 0.181 / extreme 0.081
  - 샘플 그리드: `docs/figures/lowlight_grid.jpg`

### 다음 작업
1. PyTorch cu128 + torchreid 설치 → OSNet 사전학습 가중치로 조도별 Rank-1/mAP/mINP (Protocol A) → `results/baseline.csv`
2. `build_annotations.py`: market_attribute.mat 매핑(6개 속성) + 템플릿 캡션
3. Zero-DCE 전처리 비교, 속성 재순위화
4. MOT17 2~3개 시퀀스 다운로드 → 합성 → 트래커 비교

### 결정: 혼합안 (11/2 추가)
- **직접 학습은 하지 않음.** Protocol B와 속성 분류 헤드 학습은 제외. 사전학습 OSNet, Zero-DCE, YOLO는 **추론만** 사용 (로컬 GPU).
- **AI API는 호출량이 적은 곳에만 사용** (반 공용 키이므로 아껴 씀):
  1. 데모 텍스트 검색: 자연어 → 속성 JSON 추출
  2. 시맨틱 캡션: 속성 → 자연스러운 문장 (여러 ID를 한 번에 묶어 호출)
  3. 신규 실험: **비전 LLM의 조도별 속성 인식 정확도** (샘플 약 50장 × 3등급, Market Attribute 정답과 비교)
- API 키는 `.env`(gitignore, 권한 600)에만 저장. `.git/hooks/pre-commit`이 키 패턴과 `.env` 커밋을 차단 (로컬 훅이라 clone 시 다시 설치해야 함). **키 커밋·푸시 절대 금지.**
- `vastreid/llm.py`: 디스크 캐시, 실행당 호출 상한(`configs/api.yaml`의 `max_calls_per_run`), dry-run, `outputs/api_usage.csv`에 토큰 사용량 기록.
- Provider: **xAI Grok** (`grok-4.20-0309-non-reasoning`) 동작 확인. OpenAI(`gpt-4.1-mini`)도 새 키로 동작 확인.
- torchreid: PyPI판은 제3자 재배포라 사용하지 않음. 공식 레포를 사용자가 직접 설치하기로 함 (진행 대기).

### 베이스라인 결과 (11/2, `results/baseline.csv`, 실제 실행값)
- 환경: torch 2.11.0+cu128, torchreid 1.4.0(공식 레포, 사용자가 직접 설치, build-essential 필요), 가중치는 공식 Model Zoo의 `osnet_x1_0` Market-1501 → `weights/osnet_x1_0_market1501.pth` (gitignore, gdown id `1vduhq5DpN2q1g4fYEZfPI17MJeh9qyrA`)
- 검증: normal/normal R1 94.5 / mAP 82.6 ↔ 공식 94.2 / 82.6 → 평가 파이프라인 정상
- `tools/evaluate.py`: same(q·g 같은 조도), cross(q 저조도 → g normal) 두 프로토콜. 전체 실행 46초, GPU 최고 63°C

| protocol | query | gallery | R1 | mAP | mINP |
|---|---|---|---|---|---|
| same | normal | normal | 94.5 | 82.6 | 51.0 |
| same | low | low | 48.5 | 22.4 | 2.6 |
| cross | low | normal | 25.5 | 20.3 | 7.6 |
| same | extreme | extreme | 2.5 | 1.0 | 0.1 |
| cross | extreme | normal | 0.0 | 0.3 | 0.2 |

- 관찰: extreme은 사실상 무작위 수준으로 붕괴. Zero-DCE 결과를 보고 extreme 합성 강도를 재조정할지 결정 (조정 시 버전을 v0.2로 올리고 기록).

### 저조도 개선 비교 (11/2, `results/enhance.csv`, 실제 실행값) — 개선 효과 없음
- `vastreid/enhance.py`: gamma(적응형), clahe, zerodce(공식 Epoch99.pth, 79,416 params, strict 로드). 모든 query·gallery에 동일 적용.
- Zero-DCE는 32장씩 나눠 적용 (256장을 한 번에 넣으면 VRAM 16GB 초과 → WSL sysmem fallback으로 사실상 정지).
- 전체 실행 202초, GPU 최고 약 60°C.

| enhance | normal/normal R1 | low/low R1 | low→normal R1 | extreme/extreme R1 |
|---|---|---|---|---|
| none | 94.5 | 48.5 | 25.5 | 2.5 |
| gamma | 90.1 | 43.1 | 14.9 | 2.9 |
| clahe | 85.8 | 42.8 | 17.7 | 3.5 |
| zerodce | 62.2 | 32.4 | 10.7 | 2.5 |

- 원인 (`docs/figures/enhance_grid.jpg`로 육안 확인):
  1. normal 이미지에도 적용하면 과노출·탈색 (Zero-DCE normal R1 94.5 → 62.2)
  2. 밝기를 올리면 합성 센서 노이즈가 함께 증폭되어 컬러 반점이 생김. 노이즈 제거 없이 밝기만 올리면 ReID에 역효과.
- 후보 대응: (a) 밝기 게이팅: mean_luminance가 임계값 미만인 이미지에만 개선 적용, (b) 노이즈 제거 후 개선, (c) 합성 v0.2: 실제 CCTV ISP처럼 노이즈 제거된 출력을 반영 (extreme 무작위 수준 문제와 함께 검토)

### 게이팅·노이즈 제거 실험 (11/2, `results/enhance_v2.csv`, 실제 실행값)
- 게이트 τ=0.25: 성능을 보지 않고 합성 로그의 밝기 분포로만 정함 (normal 하위 5% = 0.257). 적용 비율은 extreme 100%, low 약 92%, normal 약 4%.
- 노이즈 제거: OpenCV NLM (h=10), 밝기 개선 전에 적용. 전체 실행은 게이팅만 151초, NLM 포함 443초.

| 설정 (R1 %) | normal | low/low | low→normal | extreme/extreme |
|---|---|---|---|---|
| none (기준) | 94.5 | **48.5** | **25.5** | 2.5 |
| gate+gamma | 92.6 | 41.6 | 16.2 | 2.9 |
| gate+clahe | 93.6 | 42.2 | 16.7 | 3.5 |
| gate+zerodce | 91.8 | 31.5 | 8.5 | 2.5 |
| gate+nlm | 93.9 | 38.3 | 24.4 | 2.6 |
| gate+nlm+gamma | 92.0 | 32.4 | 10.2 | 3.1 |
| gate+nlm+clahe | 93.0 | 42.3 | 23.4 | 3.1 |
| gate+nlm+zerodce | 91.4 | 27.1 | 9.6 | 2.9 |

- 결론: 게이팅으로 normal 손상은 해결됨 (zerodce normal 62.2 → 91.8). 하지만 low·extreme에서는 어떤 조합도 개선 없음 기준을 넘지 못함 (픽셀 수준 개선의 한계). NLM 노이즈 제거만 해도 low/low 성능이 떨어짐 (디테일 손실).
- 해석: 밝기를 되살려도 노이즈와 정보 손실이 ReID 특징에 남음. 사전학습 모델 + 픽셀 개선만으로는 회복 불가 → 발표에서 "부정적 결과"로 명확히 제시.
- 다음 후보: 합성 v0.2 (실제 CCTV처럼 ISP 노이즈 제거된 출력, extreme 강도 완화) 검토, 속성 재순위화(조도에 강한 색·의복 정보) 실험.

### 개발 일지
- `docs/DEVLOG.md`: 제출용 개발·학습 일지 (시간순 결정, 시행착오, 결과, 배운 점). **작업이 끝날 때마다 갱신할 것.**

### 속성·재순위화 (11/2)
- `tools/build_annotations.py` → `annotations/person_attributes.json` (6개 속성, gender/age/hair 제외), `data/vast/annotations/*.json`
- `vastreid/reid.py`: 특징 추출 공용 모듈 (evaluate.py와 rerank.py가 함께 사용). 특징 캐시는 `outputs/features/`.
- `tools/rerank.py --attr-source oracle` → `results/rerank_oracle.csv`: 상한선. low/low R1 48.1 → 76.1, cross low R1 23.0 → 61.0.
- 주의: train 기반 검증은 OSNet이 train으로 학습되어 편향됨 (DEVLOG 시행착오 6).
- **다음: CLIP zero-shot 속성 예측 → 현실적인 재순위화** (`--attr-source clip`), 이어서 VLM 저조도 속성 실험(API, 샘플).

### CLIP 속성·현실적 재순위화 (11/2)
- `open_clip_torch` 3.3.0, ViT-B-16/openai. 속성 확률은 `outputs/attr_probs/*.npz` (gitignore).
- `results/attr_acc_clip.csv`: zero-shot 평균 정확도 normal 52.8 / low 46.2 / extreme 26.1. 소매·가방·모자는 다수 클래스보다 낮음.
- `results/rerank_clip.csv`, `results/rerank_witness.csv`: **low→normal R1 23.0 → 27.8(clip) / 29.2(witness)**. 처음으로 확인된 개선. same-level은 +1%p 이하.
- 다음 후보: (1) 속성 예측 정확도 개선 (CLIP 특징 위 선형 분류기 = 수 초짜리 로지스틱 회귀. 딥러닝 학습은 아님, 사용자 확인 필요), (2) VLM(Grok/GPT) 저조도 속성 정확도 비교 실험(샘플, API), (3) 데모.

### A: 선형 분류기 속성 (11/2)
- `tools/attr_probe.py` (CLIP 특징 + 로지스틱 회귀, ID 기준 5-fold 교차 예측) → `outputs/attr_probs/probe__*.npz`, `results/attr_acc_probe.csv` (평균 정확도 78.3 / 75.3 / 65.1)
- `tools/rerank.py --attr-source probe | witness-probe` → low→normal R1 **29.6 / 34.3** (외형만 23.0). λ가 0.5(범위 끝)에서 선택됨 → 범위 확장 검토.

### B: 비전 LLM 저조도 속성 실험 (11/2)
- `tools/vlm_attributes.py` → `results/attr_acc_vlm.csv` (50명 × 3조도, 모델당 150회, 캐시는 `outputs/api_cache/`)
- 평균 정확도 normal/low/extreme: Grok 85.6/75.7/36.7, GPT 83.2/69.9/32.5, 선형 분류기 78.5/78.6/63.8, zero-shot 52.4/44.5/21.2
- API 누적 사용량: 약 314회, 약 12만 토큰 (`outputs/api_usage.csv`)
- **다음: C. Gradio 데모** (이미지 검색 / 텍스트 검색 / 벤치마크 3탭)

### C: Gradio 데모 (11/2)
- `demo/app.py`: 실행 `python demo/app.py` → http://127.0.0.1:7860 (WSL에서 실행해도 Windows 브라우저에서 열림)
- 필요 파일(로컬 생성): `data/vast/`, `outputs/features/osnet_x1_0__none__*.pt`, `outputs/attr_probs/probe__*.npz`
- 텍스트 검색 LLM 옵션은 로컬 전용. **HF Spaces에는 키를 절대 넣지 않음** → 규칙 기반 + 샘플 갤러리만.
- 남은 일: HF Spaces 공개판(샘플 갤러리 데이터 팩), 데모 화면 캡처를 docs/에 추가, MOT17 트래킹 비교, 발표 자료.

### hubness 완화 CSLS (11/2) — 현재 최고 결과
- `tools/hubness.py` → `results/hubness.csv`. 참조 집합 = query 조도의 train 이미지, k·β·λ는 train 검증에서 선택.
- **Low→Normal R1: 외형만 23.0 → CSLS 43.1 → CSLS+목격자 속성 52.0** (mAP 18.7 → 33.8 → 40.7). Low→Low 48.1 → 53.7. Extreme→Normal 0.0 → 9.3.
- β가 1.0(범위 끝)에서 선택됨 → 범위 확장 검토. 데모에 CSLS 옵션 추가 필요.
- README 전면 개편 완료 (구조·원리·API 활용·결과·한계).

### 대기 중: 노션 기획서 수정 (11/2 요청)
- 페이지: https://app.notion.com/p/VAST-ReID-3f9f73de03a580f58953f32f671a65b0 (로그인 필요, Notion 커넥터로 접근)
- 원칙: **항목 추가·삭제 금지, 바뀐 내용만 수정.** 수치는 results/*.csv의 실제 값만 사용.
- 반영할 변경 사항:
  1. 실촬영 생략 → 공개 데이터셋(Market-1501) + 물리 기반 저조도 합성만
  2. Colab → 로컬 PC (WSL2, RTX 5080, PyTorch cu128)
  3. 딥러닝 학습 없음 (사전학습 OSNet·CLIP·Zero-DCE 추론 + 속성용 선형 분류기)
  4. VLM 라벨링 → 공개 속성 라벨(Market-1501 Attribute) 6개, gender/age/hair 제외
  5. `lux_estimate` → `mean_luminance`, track_id/bbox 없음
  6. 개선 방법: Zero-DCE 등 전처리는 실패(노이즈 증폭) → 속성 재순위화 + hubness 완화(CSLS). Low→Normal R1 23.0 → 52.0
  7. AI API(Grok, GPT): 비전 LLM 저조도 속성 실험(300회), 데모 텍스트 검색(선택). ReID 자체는 로컬 모델
  8. 트래킹: YOLO+ByteTrack 데모 → MOT17 저조도 합성 정량 평가(HOTA/IDF1) 예정
  9. 데모: Gradio 3탭 완료(로컬), 공개는 HF Spaces 예정 (키 미포함)
  10. 평가 모델: OSNet 중심 (TransReID, CLIP-ReID는 미진행)

### AdaBN + 확장 CSLS (11/2) — 현재 최고 결과
- `tools/adabn.py` → `results/adabn.csv`, 특징 캐시 `outputs/features/osnet_x1_0__adabn-{low,extreme}__*`
- `tools/hubness.py --feat adabn` → `results/hubness_adabn.csv`; 원래 특징 확장 탐색 → `results/hubness_wide.csv`
- `vastreid/metrics.py: evaluate_fast` (GPU 벡터화, evaluate와 동일 결과, 50배 빠름)
- **Low→Normal R1: 23.0 → AdaBN 85.3 → AdaBN+CSLS+목격자 86.8 / Extreme→Normal 0.0 → 35.6**
- 데모: AdaBN·CSLS 옵션, 벤치마크 종합 그래프. README·DEVLOG 갱신.
- 한계: 합성 저조도는 AdaBN과 잘 맞아 효과가 과대평가됐을 수 있음.
- 다음: MOT17 트래킹 (ByteTrack vs BoT-SORT+OSNet vs +AdaBN), HF Spaces 공개판, 발표 자료, 노션 기획서 반영(사용자가 붙여넣기).

### MOT17 트래킹 (11/2)
- 데이터: `data/raw/mot17/MOT17/train/MOT17-{02,04,09}-FRCNN` (zip은 `data/raw/mot17_zip/`, 5.9GB, 필요 없으면 삭제 가능), 저조도 `data/vast_mot/{level}/{seq}`
- `tools/track_mot.py`: YOLO11m(`weights/yolo11m.pt`) 검출 캐시 `outputs/mot_dets/`, 결과 `outputs/mot_results/{level}/{tracker}/{seq}.txt`, `--no-eval` / `--eval-only`
- 패키지: ultralytics 8.4, boxmot 26(외형 특징은 `Detections(embeddings=...)`로 전달), TrackEval(공식 레포 설치, `np.float` 별칭 패치 포함)
- `results/tracking.csv`: Low는 HOTA 45.9 → 43.7로 거의 유지, Extreme은 검출 붕괴(DetA 7.7). AssA는 ByteTrack 31.2 → BoT-SORT+OSNet 37.9 → AdaBN 40.3.
- 다음: (선택) YOLO AdaBN, HF Spaces 공개판, 발표 자료, 노션 기획서 반영

### YOLO AdaBN (실패) / HF Spaces (11/2)
- `tools/track_mot.py --det-adabn` (적응용 `data/vast_mot_adapt/{level}/MOT17-11-FRCNN`) → `results/tracking_yolo_adabn.csv`: 모든 조도에서 악화 (장면 차이 때문). DEVLOG에 부정적 결과로 기록.
- `tools/build_space.py` → `space/data/` (git 제외), `space/app.py` (numpy 전용 Gradio). HF 계정 `krskulk`, 토큰은 `.env`의 HF_TOKEN.
- **Gradio Space는 PRO 필요(402)** → 사용자 결정 대기: A 정적 HTML/JS Space(권장, 무료) / B Gradio-Lite / C PRO 구독. 공개 범위도 결정 필요(비공개 권장).

### 웹 데모 (11/2) — 디자인 시안 구현
- 실행: `python demo/server.py` → http://127.0.0.1:7860 (화면 `demo/web/`, 로직 `vastreid/demo_core.py`)
- `tools/failure_analysis.py` → `results/failure_causes.csv`, `outputs/demo/hub_freq__*.npy` (데모 실패 원인 칩에 필요)
- Gradio 데모(`demo/app.py`)는 이전 버전으로 유지. 배포(HF)는 보류.
- 남은 작업: 발표용 그림(`tools/make_figures.py`), 발표 자료, README 데모 캡처, 리허설

### k-reciprocal + 데모 조정 (11/2)
- `tools/kreciprocal.py` → `results/kreciprocal.csv` (표준값 k1=20 k2=6 λ=0.3, **원래 거리에 먼저 적용 후 CSLS·속성**). AdaBN + k-reciprocal + CSLS + 목격자 속성: **Low→Normal 90.3 / mAP 88.2**, Extreme→Normal 34.4 / 37.3.
- AdaBN 없이 k-reciprocal은 손해(23.0 → 19.9).
- 데모: `tools/build_demo_cache.py` → `outputs/demo/kr__*.npy` (재순위화 옵션용). 화면은 "시안 1440px + 비율 확대(zoom)" 버전으로 롤백 상태, 텍스트 검색 오른쪽 카드만 높이 맞춤.
- 서버 재시작이 필요한 경우: results CSV 갱신, demo_core.py 수정 후 (lru_cache).

### 저조도 미세조정 + AI 설명 기능 (11/2) — 현재 최고 결과
- `tools/finetune.py` (train 3조도, 12 에폭, 7분) → `weights/osnet_x1_0_vast_lowlight_ft.pth`(git 제외), 특징 캐시 `outputs/features/osnet_x1_0-ft__*`, `results/finetune.csv`.
- 조합: `hubness.py/kreciprocal.py --prefix osnet_x1_0-ft` → `results/hubness_ft*.csv`, `results/kreciprocal_ft.csv`.
  **미세조정 + k-reciprocal + CSLS + 속성(AdaBN 없음): Low→Normal 92.9 / 89.1, Extreme→Normal 79.2 / 76.9.** Normal→Normal은 93.5(사전학습 95.5).
- 미세조정 모델에는 AdaBN이 해로움 → 데모에서 미세조정 백본 선택 시 AdaBN 자동 off.
- 데모: 백본 선택(기본 미세조정), `tools/build_demo_cache.py --prefix osnet_x1_0-ft` → `outputs/demo/kr__osnet_x1_0-ft__*.npy`. 벤치마크에 미세조정 2행 추가.
- AI(로컬 전용, OpenAI gpt-4.1-mini): `/api/ai_explain`(오답 설명), `/api/ai_caption`(시맨틱 캡션), 서버 실행당 120회 상한.
- 미세조정 백본의 hub 빈도(`hub_freq`)는 없음 → 실패 원인 칩에서 hub 항목만 생략됨.
