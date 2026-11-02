"""data/vast/{level}/{split}/ 폴더의 Market-1501 형식 이미지 목록.

파일명 규칙: {pid}_c{cam}s{seq}_{frame}_{bbox}.jpg
  - pid == -1 : junk (표준 프로토콜대로 제외)
  - pid == 0  : distractor (gallery에 유지)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_PAT = re.compile(r"^(-?\d+)_c(\d)")


@dataclass(frozen=True)
class Item:
    path: Path
    pid: int
    camid: int  # 0-based


def parse_name(name: str) -> tuple[int, int]:
    m = _PAT.match(name)
    if not m:
        raise ValueError(f"Market-1501 형식이 아닌 파일명: {name}")
    return int(m.group(1)), int(m.group(2)) - 1


def list_split(root: Path, level: str, split: str, drop_junk: bool = True) -> list[Item]:
    d = Path(root) / level / split
    if not d.is_dir():
        raise FileNotFoundError(d)
    items = []
    for f in sorted(d.glob("*.jpg")):
        pid, cam = parse_name(f.name)
        if drop_junk and pid == -1:
            continue
        items.append(Item(f, pid, cam))
    return items
