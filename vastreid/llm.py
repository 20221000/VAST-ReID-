"""반 공용 AI API를 아껴 쓰기 위한 얇은 래퍼.

- 키는 .env / 환경변수에서만 읽음 (코드·레포에 저장 금지)
- 디스크 캐시: 같은 요청은 다시 호출하지 않음
- 호출 상한: 실행당 max_calls_per_run 초과 시 BudgetExceeded
- dry_run=True 이면 호출하지 않고 횟수만 셈 (비용 사전 추정용)
- 사용 토큰은 outputs/api_usage.csv 에 누적 기록

xAI(Grok)와 OpenAI 모두 OpenAI 호환 Chat Completions 를 사용.
"""
from __future__ import annotations

import base64
import csv
import hashlib
import json
import os
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


class BudgetExceeded(RuntimeError):
    pass


class LLM:
    def __init__(self, config: str | Path = ROOT / "configs/api.yaml", provider: str | None = None,
                 dry_run: bool = False, max_calls: int | None = None):
        cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
        self.provider = provider or cfg["provider"]
        self.pcfg = cfg["providers"][self.provider]
        self.max_calls = max_calls if max_calls is not None else cfg["max_calls_per_run"]
        self.max_tokens = cfg["max_output_tokens"]
        self.cache_dir = ROOT / cfg["cache_dir"]
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.usage_log = ROOT / "outputs/api_usage.csv"
        self.dry_run = dry_run
        self.calls = 0          # 실제(또는 dry-run 가상) 호출 수
        self.cache_hits = 0
        self._client = None

    # ------------------------------------------------------------ 내부
    def _get_client(self):
        if self._client is None:
            from dotenv import load_dotenv
            from openai import OpenAI
            load_dotenv(ROOT / ".env")
            key = os.environ.get(self.pcfg["key_env"])
            if not key:
                raise RuntimeError(f"환경변수 {self.pcfg['key_env']} 가 없습니다 (.env 확인).")
            kw = {"api_key": key}
            if self.pcfg.get("base_url"):
                kw["base_url"] = self.pcfg["base_url"]
            self._client = OpenAI(**kw)
        return self._client

    def _cache_path(self, payload: dict) -> Path:
        h = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        return self.cache_dir / f"{h[:2]}/{h}.json"

    def _log_usage(self, model: str, usage, tag: str):
        new = not self.usage_log.exists()
        self.usage_log.parent.mkdir(parents=True, exist_ok=True)
        with open(self.usage_log, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["time", "provider", "model", "tag", "prompt_tokens", "completion_tokens"])
            w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), self.provider, model, tag,
                        getattr(usage, "prompt_tokens", ""), getattr(usage, "completion_tokens", "")])

    # ------------------------------------------------------------ 공개 API
    def chat(self, system: str, user: str, images: list[str | Path] | None = None,
             json_mode: bool = False, tag: str = "") -> str | None:
        """텍스트(+선택적 이미지) 질의. dry_run 이면 None 반환."""
        model = self.pcfg["vision_model" if images else "text_model"]
        content: list[dict] | str = user
        if images:
            content = [{"type": "text", "text": user}]
            for p in images:
                b64 = base64.b64encode(Path(p).read_bytes()).decode()
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        messages = [{"role": "system", "content": system}, {"role": "user", "content": content}]
        payload = {"provider": self.provider, "model": model, "messages": messages, "json": json_mode,
                   "max_tokens": self.max_tokens}

        cp = self._cache_path(payload)
        if cp.exists():
            self.cache_hits += 1
            return json.loads(cp.read_text(encoding="utf-8"))["text"]

        if self.calls >= self.max_calls:
            raise BudgetExceeded(f"호출 상한 {self.max_calls}회 도달 (configs/api.yaml max_calls_per_run).")
        self.calls += 1
        if self.dry_run:
            return None

        kw = dict(model=model, messages=messages, max_tokens=self.max_tokens, temperature=0)
        if json_mode:
            kw["response_format"] = {"type": "json_object"}
        resp = self._get_client().chat.completions.create(**kw)
        text = resp.choices[0].message.content or ""
        self._log_usage(model, resp.usage, tag)
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_text(json.dumps({"text": text, "model": model}, ensure_ascii=False), encoding="utf-8")
        return text

    def summary(self) -> str:
        mode = "DRY-RUN(가상)" if self.dry_run else "실제"
        return f"[api] provider={self.provider} {mode} 호출 {self.calls}회 / 상한 {self.max_calls}, 캐시 적중 {self.cache_hits}회"
