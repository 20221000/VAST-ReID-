"""ReID 특징 추출 공용 코드 (사전학습 torchreid 모델, 추론 전용)."""
from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch

from vastreid.enhance import mean_luma

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def gpu_temp() -> str:
    try:
        return subprocess.check_output(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
                                       text=True).strip()
    except Exception:
        return "?"


def load_model(name: str, weights: Path, device: str):
    import torchreid
    model = torchreid.models.build_model(name=name, num_classes=1, pretrained=False)
    sd = torch.load(weights, map_location="cpu", weights_only=True)
    sd = sd.get("state_dict", sd)
    sd = {k.removeprefix("module."): v for k, v in sd.items() if not k.removeprefix("module.").startswith("classifier.")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.startswith("classifier.")]
    if missing or unexpected:
        raise RuntimeError(f"가중치 불일치 missing={missing[:5]} unexpected={unexpected[:5]}")
    return model.eval().to(device)


class ImgDS(torch.utils.data.Dataset):
    def __init__(self, paths, size=(128, 256), cpu_enhance=None, denoise=None, gate=None):
        self.paths, self.size, self.cpu_enhance = paths, size, cpu_enhance
        self.denoise, self.gate = denoise, gate

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = cv2.imread(str(self.paths[i]), cv2.IMREAD_COLOR)
        apply = self.gate is None or mean_luma(img) < self.gate
        if apply and self.denoise is not None:
            img = self.denoise(img)
        if apply and self.cpu_enhance is not None:
            img = self.cpu_enhance(img)
        img = cv2.resize(img, self.size, interpolation=cv2.INTER_LINEAR)
        return torch.from_numpy(img[..., ::-1].transpose(2, 0, 1).copy()), apply  # RGB uint8 CHW, 적용 여부


@torch.no_grad()
def extract(model, paths, device, batch=256, workers=8, cpu_enhance=None, gpu_enhance=None,
            denoise=None, gate=None) -> tuple[torch.Tensor, float]:
    """특징과 개선이 적용된 이미지 비율을 반환."""
    dl = torch.utils.data.DataLoader(ImgDS(paths, cpu_enhance=cpu_enhance, denoise=denoise, gate=gate), batch_size=batch,
                                     num_workers=workers, pin_memory=True)
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    out, n_applied = [], 0
    for x, applied in dl:
        x = x.to(device, non_blocking=True).float() / 255.0
        applied = applied.to(device)
        n_applied += int(applied.sum())
        if gpu_enhance is not None and applied.any():
            # Zero-DCE는 전 해상도 32ch 활성값을 쓰므로 작은 묶음으로 나눠 VRAM 초과(WSL sysmem fallback) 방지
            sel = x[applied]
            x[applied] = torch.cat([gpu_enhance(c) for c in sel.split(32)])
        out.append(model((x - mean) / std).float())
    return torch.cat(out), n_applied / max(1, len(paths))
