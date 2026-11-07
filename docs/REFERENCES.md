# 참고 문헌 · 출처 (발표용)

프로젝트에서 사용하거나 비교한 데이터셋·모델·기법의 원 논문과 출처입니다. 발표 자료의 "참고 문헌" 슬라이드와 각 결과 슬라이드 하단 출처 표기에 그대로 쓸 수 있습니다.
"어디에 썼나"는 이 저장소에서의 역할입니다.

## 1. 데이터셋

| 이름 | 논문 / 출처 | 어디에 썼나 |
|---|---|---|
| **Market-1501** | L. Zheng, L. Shen, L. Tian, S. Wang, J. Wang, Q. Tian. *Scalable Person Re-identification: A Benchmark.* **ICCV 2015** | 저조도 3단계 합성의 원본 (1,501명, 36,036장 × 3조도) |
| **Market-1501 Attribute** | Y. Lin, L. Zheng, Z. Zheng, Y. Wu, Z. Hu, C. Yan, Y. Yang. *Improving Person Re-identification by Attribute and Identity Learning.* **Pattern Recognition 2019** · [github.com/vana77/Market-1501_Attribute](https://github.com/vana77/Market-1501_Attribute) | 의복·소지품 속성 6개 (성별·나이·머리 제외) |
| **MSMT17** | L. Wei, S. Zhang, W. Gao, Q. Tian. *Person Transfer GAN to Bridge Domain Gap for Person Re-Identification.* **CVPR 2018** | 추가 학습(1,041명) · 다른 장소 일반화 시험. Kaggle 비공식 사본 `ouassimaazzouzi/msmt17` 사용(공식은 동의서 신청) |
| **Night600** | A. Lu, Z. Zhang, Y. Huang, Y. Zhang, C. Li, J. Tang, L. Wang. *Illumination Distillation Framework for Nighttime Person Re-Identification and A New Benchmark.* **IEEE Transactions on Multimedia, vol. 26, pp. 406–419, 2024** · [arXiv 2308.16486](https://arxiv.org/abs/2308.16486) · [github.com/Alexadlu/IDF](https://github.com/Alexadlu/IDF) | 실제 야간 CCTV 검증(test 300명) + 최종 모델 학습(train 300명). 비교 수치 BoT 11.2 / AGW 12.5 / TransReID 16.0 / IDF 17.2 (Rank-1, 논문 Table II) |
| **MOT17** | A. Milan, L. Leal-Taixé, I. Reid, S. Roth, K. Schindler. *MOT16: A Benchmark for Multi-Object Tracking.* **arXiv 1603.00831, 2016** | 저조도 트래킹 시험 (3개 시퀀스 합성) |
| NightReID (사용 안 함, 향후 검증 후보) | Y. Zhao, W. Ruan, H. Li, M. Ye. *NightReID: A Large-Scale Nighttime Person Re-Identification Benchmark.* **AAAI 2025**, 39(10):10519–10527 · [DOI 10.1609/aaai.v39i10.33142](https://ojs.aaai.org/index.php/AAAI/article/view/33142) | 1,500명·53,000장 이상 실제 야간. 동의서 신청 필요 |
| DukeMTMC-reID (사용 안 함) | 원저자가 공개 철회 | 윤리 문제로 제외 |

> **이름이 같은 다른 연구 (발표 때 꼭 구분)**: *VAST-ReID: A Low-Light Benchmark Dataset for Person Re-Identification with Visual and Attribute-Rich Semantic Tracking*, Khan et al., **WACV 2026 (oral)** · [wacv.thecvf.com/virtual/2026/oral/1192](https://wacv.thecvf.com/virtual/2026/oral/1192).
> 제목이 이 프로젝트와 **완전히 같지만** 별개의 연구입니다. 실제 저조도 감시 영상 1,211개·21개 장소·169명을 직접 촬영한 데이터셋이고, 이 프로젝트는 공개 데이터셋에 저조도를 합성하고 Night600으로 검증합니다.

## 2. ReID 모델 · 백본

| 이름 | 논문 / 출처 | 어디에 썼나 |
|---|---|---|
| **OSNet** | K. Zhou, Y. Yang, A. Cavallaro, T. Xiang. *Omni-Scale Feature Learning for Person Re-Identification.* **ICCV 2019** | 베이스라인 백본 (Market-1501 사전학습 가중치) |
| **Torchreid** | K. Zhou, T. Xiang. *Torchreid: A Library for Deep Learning Person Re-Identification in Pytorch.* **arXiv 1910.10093, 2019** · [github.com/KaiyangZhou/deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid) | OSNet 구현·가중치, 평가 코드 |
| **CLIP** | A. Radford et al. *Learning Transferable Visual Models From Natural Language Supervision.* **ICML 2021** | 최종 백본(ViT-B/16 이미지 인코더), 속성 예측 특징 |
| OpenCLIP | G. Ilharco et al. *OpenCLIP* (2021) · [github.com/mlfoundations/open_clip](https://github.com/mlfoundations/open_clip) | CLIP 가중치 로드(`openai` ViT-B/16) |
| CLIP-ReID (비슷한 접근) | S. Li, L. Sun, Q. Li. *CLIP-ReID: Exploiting Vision-Language Model for Image Re-Identification without Concrete Text Labels.* **AAAI 2023** | CLIP 이미지 인코더를 ReID로 미세조정하는 방향의 선행 연구 (이 프로젝트는 텍스트 프롬프트 학습 없이 이미지 인코더만 사용) |
| **BNNeck / Strong Baseline (BoT)** | H. Luo, Y. Gu, X. Liao, S. Lai, W. Jiang. *Bag of Tricks and A Strong Baseline for Deep Person Re-identification.* **CVPR Workshops 2019** | CLIP 백본의 BNNeck, Night600 비교 모델 BoT |
| **AGW · mINP** | M. Ye, J. Shen, G. Lin, T. Xiang, L. Shao, S. C. H. Hoi. *Deep Learning for Person Re-identification: A Survey and Outlook.* **IEEE TPAMI 2021** | 평가 지표 mINP, Night600 비교 모델 AGW |
| TransReID | S. He, H. Luo, P. Wang, F. Wang, H. Li, W. Jiang. *TransReID: Transformer-based Object Re-Identification.* **ICCV 2021** | Night600 비교 모델 |

## 3. 학습 기법

| 이름 | 논문 / 출처 | 어디에 썼나 |
|---|---|---|
| **Triplet 손실 + PK 샘플링** | A. Hermans, L. Beyer, B. Leibe. *In Defense of the Triplet Loss for Person Re-Identification.* **arXiv 1703.07737, 2017** | 짝 학습(같은 사람은 가깝게, 다른 사람은 멀게, 16명 × 4장) |
| 라벨 스무딩 | C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, Z. Wojna. *Rethinking the Inception Architecture for Computer Vision.* **CVPR 2016** | 분류 손실 |
| Random Erasing | Z. Zhong, L. Zheng, G. Kang, S. Li, Y. Yang. *Random Erasing Data Augmentation.* **AAAI 2020** | 데이터 증강 |

## 4. 저조도 보정 · 검색 보정

| 이름 | 논문 / 출처 | 어디에 썼나 |
|---|---|---|
| **Zero-DCE** | C. Guo, C. Li, J. Guo, C. C. Loy, J. Hou, S. Kwong, R. Cong. *Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement.* **CVPR 2020** · [github.com/Li-Chongyi/Zero-DCE](https://github.com/Li-Chongyi/Zero-DCE) | 저조도 개선 전처리 비교 → **효과 없음**(노이즈 증폭) |
| CLAHE | K. Zuiderveld. *Contrast Limited Adaptive Histogram Equalization.* **Graphics Gems IV, 1994** | 전처리 비교 |
| **AdaBN** | Y. Li, N. Wang, J. Shi, J. Liu, X. Hou. *Revisiting Batch Normalization For Practical Domain Adaptation.* **arXiv 1603.04779, 2016 (ICLR 2017 Workshop)** | 학습 없는 저조도 적응 (BN 통계 재추정) |
| **Hubness** | M. Radovanović, A. Nanopoulos, M. Ivanović. *Hubs in Space: Popular Nearest Neighbors in High-Dimensional Data.* **JMLR 11, 2010** | 저조도 실패 원인 분석의 개념 |
| **CSLS** | A. Conneau, G. Lample, M. Ranzato, L. Denoyer, H. Jégou. *Word Translation Without Parallel Data.* **ICLR 2018** | hub 감점 (원래 단어 번역용 → ReID에 적용) |
| **k-reciprocal 재순위화** | Z. Zhong, L. Zheng, D. Cao, S. Li. *Re-ranking Person Re-identification with k-reciprocal Encoding.* **CVPR 2017** | 재순위화 (논문 표준값 k1=20, k2=6, λ=0.3) |

## 5. 트래킹

| 이름 | 논문 / 출처 | 어디에 썼나 |
|---|---|---|
| YOLO11 | G. Jocher, J. Qiu. *Ultralytics YOLO11* (2024) · [github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | 사람 검출 (`yolo11m`) |
| ByteTrack | Y. Zhang et al. *ByteTrack: Multi-Object Tracking by Associating Every Detection Box.* **ECCV 2022** | 트래커 비교 |
| BoT-SORT | N. Aharon, R. Orfaig, B.-Z. Bobrovsky. *BoT-SORT: Robust Associations Multi-Pedestrian Tracking.* **arXiv 2206.14651, 2022** | 트래커 (+ OSNet 외형 특징) |
| HOTA · TrackEval | J. Luiten, A. Ošep, P. Dendorfer, P. Torr, A. Geiger, L. Leal-Taixé, B. Leibe. *HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking.* **IJCV 2021** · [github.com/JonathonLuiten/TrackEval](https://github.com/JonathonLuiten/TrackEval) | 트래킹 지표 HOTA·IDF1·MOTA |

## 6. AI API (생성형 모델)

| 이름 | 출처 | 어디에 썼나 |
|---|---|---|
| GPT-4.1 mini | OpenAI API (`gpt-4.1-mini`) | 데모 AI 오답 설명·시맨틱 캡션, 텍스트 검색 속성 추출(선택), 비전 LLM 속성 실험 |
| Grok | xAI API (`grok-4.20-0309-non-reasoning`) | 같은 용도의 대체 제공자 (`configs/api.yaml`) |

## 발표 슬라이드용 짧은 표기

- 데이터: Market-1501 (Zheng et al., ICCV 2015) · MSMT17 (Wei et al., CVPR 2018) · Night600 (Lu et al., IEEE TMM 2024)
- 모델: OSNet (Zhou et al., ICCV 2019) · CLIP ViT-B/16 (Radford et al., ICML 2021) · BNNeck (Luo et al., CVPRW 2019)
- 보정: AdaBN (Li et al., 2016) · CSLS (Conneau et al., ICLR 2018) · k-reciprocal (Zhong et al., CVPR 2017) · Zero-DCE (Guo et al., CVPR 2020)
- 실제 야간 비교 수치 출처: Lu et al., IEEE TMM 2024, Table II
