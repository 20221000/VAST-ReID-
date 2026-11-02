---
title: VAST-ReID
emoji: 🌙
colorFrom: indigo
colorTo: yellow
sdk: gradio
sdk_version: 6.31.0
app_file: app.py
pinned: false
license: other
short_description: Low-light person re-identification benchmark demo
---

# VAST-ReID 공개 데모

저조도 사람 재식별(Person Re-ID) 벤치마크의 공개 데모입니다. 코드와 전체 결과는 [GitHub](https://github.com/20221000/VAST-ReID-)에 있습니다.

- 이미지는 Market-1501(연구용 공개 데이터셋)에 저조도를 합성한 것으로, **소형 샘플 갤러리**만 포함합니다.
- 특징·속성 확률은 미리 계산되어 있어 모델 가중치나 API 키 없이 동작합니다.
- 성별·나이 등 민감 속성은 다루지 않으며, 연구·교육 목적에 한해 사용합니다.
- 이미지 출처: Market-1501 (Zheng et al., ICCV 2015). 속성 라벨: Market-1501_Attribute (Lin et al.).
