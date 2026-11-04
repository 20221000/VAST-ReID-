"""CLIP ViT-B/16 이미지 인코더 기반 ReID 백본 (CLIP-ReID의 기본 구조: 이미지 인코더 미세조정 + BNNeck).

- 입력은 다른 모델과 같은 256×128, ImageNet 정규화 텐서 → 모델 안에서 CLIP 정규화로 바꿈 (extract·학습 코드 공용).
- 위치 임베딩을 14×14 → 16×8 패치 격자로 보간 (stride 16, 128 패치).
- train: 분류 로짓 (loss="triplet"이면 (로짓, BNNeck 앞 특징)), eval: BNNeck 뒤 512차원 특징.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

IMNET_MEAN, IMNET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
CLIP_MEAN, CLIP_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)


class ClipReID(torch.nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True, grid=(16, 8)):
        super().__init__()
        import open_clip
        self.visual = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai" if pretrained else None)[0].visual
        pe = self.visual.positional_embedding.data
        n = int((pe.shape[0] - 1) ** 0.5)
        g = pe[1:].reshape(1, n, n, -1).permute(0, 3, 1, 2)
        g = F.interpolate(g, size=grid, mode="bicubic", align_corners=False).permute(0, 2, 3, 1).reshape(-1, pe.shape[1])
        self.visual.positional_embedding = torch.nn.Parameter(torch.cat([pe[:1], g]))
        self.visual.grid_size = grid
        dim = self.visual.output_dim
        self.bnneck = torch.nn.BatchNorm1d(dim)
        self.bnneck.bias.requires_grad_(False)
        self.classifier = torch.nn.Linear(dim, num_classes, bias=False)
        torch.nn.init.normal_(self.classifier.weight, std=0.001)
        self.loss = "softmax"
        a = torch.tensor(IMNET_STD) / torch.tensor(CLIP_STD)
        b = (torch.tensor(IMNET_MEAN) - torch.tensor(CLIP_MEAN)) / torch.tensor(CLIP_STD)
        self.register_buffer("norm_a", a.view(1, 3, 1, 1), persistent=False)
        self.register_buffer("norm_b", b.view(1, 3, 1, 1), persistent=False)

    def forward(self, x):
        f = self.visual(x * self.norm_a + self.norm_b)
        fb = self.bnneck(f)
        if not self.training:
            return fb
        logits = self.classifier(fb)
        return (logits, f) if self.loss == "triplet" else logits
