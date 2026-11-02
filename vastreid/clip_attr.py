"""CLIP zero-shot 속성 예측 (학습 없음).

- 속성 값마다 여러 문장 템플릿으로 텍스트 임베딩을 만들고 평균 (prompt ensembling)
- 이미지는 잘라내지 않고 정사각형으로 여백을 채움 (letterbox). 기본 CenterCrop은 세로로 긴 사람의 머리·발을 자름.
- 출력: 속성별 확률 벡터 (softmax, CLIP logit scale 사용)
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

ATTR_VALUES = {
    "upper_color": ["black", "white", "red", "purple", "yellow", "gray", "blue", "green"],
    "lower_color": ["black", "white", "pink", "purple", "yellow", "gray", "blue", "green", "brown"],
    "sleeve": ["long", "short"],
    "lower_type": ["pants", "shorts", "skirt"],
    "bag": ["backpack", "handbag", "bag", "none"],
    "hat": ["False", "True"],
}

# 속성 값 → 사람을 묘사하는 구절
PHRASES = {
    "upper_color": lambda v: f"a {v} shirt",
    "lower_color": lambda v: f"{v} pants or a {v} skirt",
    "sleeve": lambda v: {"long": "a long-sleeved top", "short": "a short-sleeved top"}[v],
    "lower_type": lambda v: {"pants": "long pants", "shorts": "shorts", "skirt": "a skirt or a dress"}[v],
    "bag": lambda v: {"backpack": "a backpack", "handbag": "a handbag", "bag": "a shoulder bag",
                      "none": "no bag"}[v],
    "hat": lambda v: {"True": "a hat", "False": "no hat"}[v],
}
VERB = {"upper_color": "wearing", "lower_color": "wearing", "sleeve": "wearing", "lower_type": "wearing",
        "bag": "carrying", "hat": "wearing"}
TEMPLATES = [
    "a photo of a person {verb} {p}.",
    "a surveillance camera photo of a pedestrian {verb} {p}.",
    "a low resolution photo of a person {verb} {p}.",
    "a cropped photo of a person {verb} {p}.",
]

MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)


def letterbox(bgr: np.ndarray, size: int = 224) -> np.ndarray:
    h, w = bgr.shape[:2]
    s = size / max(h, w)
    img = cv2.resize(bgr, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((size, size, 3), 127, np.uint8)
    y0, x0 = (size - img.shape[0]) // 2, (size - img.shape[1]) // 2
    canvas[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img
    return canvas


class _DS(torch.utils.data.Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = letterbox(cv2.imread(str(self.paths[i]), cv2.IMREAD_COLOR))
        x = (img[..., ::-1].astype(np.float32) / 255.0 - MEAN) / STD
        return torch.from_numpy(x.transpose(2, 0, 1).copy())


class ClipAttr:
    def __init__(self, arch: str = "ViT-B-16", pretrained: str = "openai", device: str = "cuda"):
        import open_clip
        self.device = device
        self.model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=pretrained, device=device)
        self.model.eval()
        tok = open_clip.get_tokenizer(arch)
        self.text = {}
        with torch.no_grad():
            for attr, values in ATTR_VALUES.items():
                embs = []
                for v in values:
                    p = PHRASES[attr](v)
                    t = tok([tpl.format(verb=VERB[attr], p=p) for tpl in TEMPLATES]).to(device)
                    e = torch.nn.functional.normalize(self.model.encode_text(t).float(), dim=-1).mean(0)
                    embs.append(torch.nn.functional.normalize(e, dim=-1))
                self.text[attr] = torch.stack(embs)  # (V, D)
        self.scale = float(self.model.logit_scale.detach().exp())

    @torch.no_grad()
    def predict(self, paths: list[Path], batch: int = 256, workers: int = 8) -> dict[str, np.ndarray]:
        dl = torch.utils.data.DataLoader(_DS(paths), batch_size=batch, num_workers=workers, pin_memory=True)
        out = {a: [] for a in ATTR_VALUES}
        for x in dl:
            with torch.autocast("cuda", dtype=torch.float16):
                f = self.model.encode_image(x.to(self.device, non_blocking=True))
            f = torch.nn.functional.normalize(f.float(), dim=-1)
            for a, t in self.text.items():
                out[a].append((self.scale * f @ t.T).softmax(-1).cpu())
        return {a: torch.cat(v).numpy() for a, v in out.items()}
