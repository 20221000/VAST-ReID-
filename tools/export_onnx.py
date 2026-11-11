"""브라우저 영상 검색용 ONNX 변환: 증류 OSNet(ReID) + YOLO11n(사람 검출).

- ReID: 입력 RGB 0~1 float [N,3,256,128] → ImageNet 정규화와 L2 정규화를 모델 안에 넣어 512차원 단위 벡터 출력.
- 검출: Ultralytics YOLO11n을 imgsz 고정, NMS 포함으로 변환 → 출력 [1,300,6] (x1,y1,x2,y2,score,cls).
- 검증: 같은 이미지에서 PyTorch와 onnxruntime 결과를 비교해 results/onnx_parity.csv에 기록.
사용: python tools/export_onnx.py [--imgsz 480]
"""
from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vastreid.reid import MEAN, STD, load_model  # noqa: E402

OUT = ROOT / "demo/static/models"


class ReIDWrap(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))

    def forward(self, x):
        f = self.net((x - self.mean) / self.std)
        return torch.nn.functional.normalize(f, dim=1)


def load_crop(p: Path) -> np.ndarray:
    img = cv2.imread(str(p))[..., ::-1]
    img = cv2.resize(img, (128, 256), interpolation=cv2.INTER_LINEAR)
    return (img.transpose(2, 0, 1).astype(np.float32) / 255.0)


def export_reid(weights: Path) -> Path:
    net = load_model("osnet_x1_0", weights, "cpu")
    m = ReIDWrap(net).eval()
    dst = OUT / "reid_osnet.onnx"
    torch.onnx.export(m, torch.rand(2, 3, 256, 128), str(dst), input_names=["images"], output_names=["feats"],
                      dynamic_axes={"images": {0: "n"}, "feats": {0: "n"}}, opset_version=17, dynamo=False)
    return dst


def check_reid(weights: Path, onnx_path: Path, n: int = 64) -> dict:
    import onnxruntime as ort
    paths = sorted((ROOT / "data").rglob("*.jpg"))
    random.Random(0).shuffle(paths)
    x = np.stack([load_crop(p) for p in paths[:n]])
    m = ReIDWrap(load_model("osnet_x1_0", weights, "cpu")).eval()
    with torch.no_grad():
        ft = m(torch.from_numpy(x)).numpy()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    t = time.perf_counter()
    fo = sess.run(None, {"images": x})[0]
    ms = (time.perf_counter() - t) * 1000 / n
    cos = (ft * fo).sum(1)
    rank_same = float(((ft @ ft.T).argsort(1)[:, -5:] == (fo @ fo.T).argsort(1)[:, -5:]).mean())
    return {"model": "reid_osnet", "n": n, "min_cos": float(cos.min()), "mean_cos": float(cos.mean()),
            "top5_order_same": rank_same, "ms_per_img_ort_cpu": round(ms, 2)}


def export_det(imgsz: int) -> Path:
    from ultralytics import YOLO
    det = YOLO("yolo11n.pt")
    f = det.export(format="onnx", imgsz=imgsz, nms=True, simplify=True, opset=17, dynamic=False)
    dst = OUT / "det_yolo11n.onnx"
    shutil.move(f, dst)
    pt = Path("yolo11n.pt")
    if pt.exists():
        shutil.move(pt, ROOT / "weights/yolo11n.pt")
    return dst


def check_det(onnx_path: Path, imgsz: int) -> dict:
    import onnxruntime as ort
    from ultralytics import YOLO
    frames = sorted((ROOT / "data").rglob("img1/*.jpg"))[:8]
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    yolo = YOLO(str(ROOT / "weights/yolo11n.pt"))
    ious, counts, ms = [], [], []
    for p in frames:
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        s = imgsz / max(h, w)
        nh, nw = round(h * s), round(w * s)
        pad = np.full((imgsz, imgsz, 3), 114, np.uint8)
        pad[:nh, :nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        x = pad[..., ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        t = time.perf_counter()
        out = sess.run(None, {sess.get_inputs()[0].name: x})[0][0]
        ms.append((time.perf_counter() - t) * 1000)
        mine = out[(out[:, 4] > 0.3) & (out[:, 5] == 0), :4] / s
        ref = yolo.predict(str(p), imgsz=imgsz, conf=0.3, classes=[0], verbose=False)[0].boxes.xyxy.cpu().numpy()
        counts.append((len(mine), len(ref)))
        for b in ref:
            if len(mine) == 0:
                ious.append(0.0)
                continue
            x1 = np.maximum(mine[:, 0], b[0]); y1 = np.maximum(mine[:, 1], b[1])
            x2 = np.minimum(mine[:, 2], b[2]); y2 = np.minimum(mine[:, 3], b[3])
            inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
            a = (mine[:, 2] - mine[:, 0]) * (mine[:, 3] - mine[:, 1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
            ious.append(float((inter / a).max()))
    return {"model": "det_yolo11n", "n": len(frames), "frames": [str(p.relative_to(ROOT)) for p in frames[:1]],
            "boxes_onnx_vs_ultra": counts, "mean_best_iou": float(np.mean(ious)) if ious else None,
            "ms_per_frame_ort_cpu": round(float(np.median(ms)), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=ROOT / "weights/osnet_x1_0_distill_clipnight.pth")
    ap.add_argument("--imgsz", type=int, default=480)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    r = export_reid(args.weights)
    rows = [check_reid(args.weights, r)]
    print(rows[-1])
    d = export_det(args.imgsz)
    rows.append(check_det(d, args.imgsz))
    print(rows[-1])
    for p in (r, d):
        print(p.name, round(p.stat().st_size / 1e6, 1), "MB")
    with open(ROOT / "results/onnx_parity.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "key", "value"])
        for row in rows:
            for k, v in row.items():
                if k != "model":
                    w.writerow([row["model"], k, v])


if __name__ == "__main__":
    main()
