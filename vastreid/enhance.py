"""저조도 개선 전처리 (모두 사전학습 또는 고전 기법, 학습 없음).

- gamma  : 적응형 감마. 평균 밝기가 target(0.4)이 되도록 감마를 이미지마다 계산
- clahe  : LAB의 L 채널에 CLAHE
- zerodce: Zero-DCE (Guo et al., CVPR 2020) 공식 사전학습 가중치 (Epoch99.pth)
           구조는 공식 레포 Li-Chongyi/Zero-DCE의 enhance_net_nopool 과 동일

CPU 기법(gamma, clahe)은 BGR uint8 → BGR uint8 함수,
zerodce는 GPU 배치용 torch 모듈 (RGB float 0~1 → RGB float 0~1).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ 고전 기법
def adaptive_gamma(bgr: np.ndarray, target: float = 0.4) -> np.ndarray:
    x = bgr.astype(np.float32) / 255.0
    m = float(np.clip(x.mean(), 1e-3, 0.99))
    g = np.log(target) / np.log(m)
    return np.round(np.power(x, g) * 255.0).clip(0, 255).astype(np.uint8)


_CLAHE = None


def clahe(bgr: np.ndarray) -> np.ndarray:
    global _CLAHE
    if _CLAHE is None:
        _CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 8))
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab[..., 0] = _CLAHE.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


CPU_ENHANCERS = {"gamma": adaptive_gamma, "clahe": clahe}


# ------------------------------------------------------------------ 게이팅 / 노이즈 제거
def mean_luma(bgr: np.ndarray) -> float:
    """BT.709 luma 평균 (0~1). synthesize_lowlight.py 의 mean_luminance 와 같은 정의."""
    b, g, r = (bgr[..., i].astype(np.float32) / 255.0 for i in range(3))
    return float((0.2126 * r + 0.7152 * g + 0.0722 * b).mean())


def nlm_denoise(bgr: np.ndarray, h: float = 10.0) -> np.ndarray:
    """Non-Local Means 컬러 노이즈 제거 (밝기 개선 전에 적용)."""
    return cv2.fastNlMeansDenoisingColored(bgr, None, h, h, 7, 21)


DENOISERS = {"none": None, "nlm": nlm_denoise}


# ------------------------------------------------------------------ Zero-DCE
class ZeroDCE(nn.Module):
    def __init__(self, nf: int = 32):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.e_conv1 = nn.Conv2d(3, nf, 3, 1, 1)
        self.e_conv2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.e_conv3 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.e_conv4 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.e_conv5 = nn.Conv2d(nf * 2, nf, 3, 1, 1)
        self.e_conv6 = nn.Conv2d(nf * 2, nf, 3, 1, 1)
        self.e_conv7 = nn.Conv2d(nf * 2, 24, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        for ri in torch.split(r, 3, dim=1):  # 8회 반복 light-enhancement curve
            x = x + ri * (x * x - x)
        return x.clamp(0, 1)


def load_zerodce(weights: Path = ROOT / "weights/zerodce_epoch99.pth", device: str = "cuda") -> ZeroDCE:
    m = ZeroDCE()
    sd = torch.load(weights, map_location="cpu", weights_only=True)
    sd = {k.removeprefix("module."): v for k, v in sd.items()}
    m.load_state_dict(sd, strict=True)
    return m.eval().to(device)


GPU_ENHANCERS = {"zerodce": load_zerodce}
ALL = ["none", *CPU_ENHANCERS, *GPU_ENHANCERS]
