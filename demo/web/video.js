/* 영상에서 찾기 — 브라우저 안에서 사람 검출(YOLO11n) + ReID(증류 OSNet)를 ONNX Runtime Web으로 실행.
   사진·영상은 서버로 보내지 않는다. 절차는 tools/build_video_sample.py(Python 재현·정확도 측정)와 같다:
   초당 fps장 추출 → 검출(conf ≥ 0.35, 높이 ≥ 프레임의 3.7%) → 인접 프레임 IoU ≥ 0.2로 같은 사람 묶기(트랙)
   → 트랙마다 짝수 번째 등장에서 최대 6장만 특징 계산 → 트랙 점수 = 상위 3장 유사도 평균 → 점수 순 후보. */
(() => {
  const ORT_VER = "1.20.1";
  const ORT_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VER}/dist/`;
  const IMGSZ = 480, CONF = 0.35, MIN_H = 0.037, LINK_IOU = 0.2, MAX_EMB = 6, TOPK = 3, MAX_W = 1280, SHOW = 8;
  const $ = (s) => document.querySelector(s);
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const mmss = (t) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;
  const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
  const canvas = (w, h) => Object.assign(document.createElement("canvas"), { width: w, height: h });

  const V = {
    inited: false, sample: null, det: null, reid: null, ep: "",
    q: null,        // {name, img(canvas 원본), boxes:[{b,score}], sel: 인덱스 또는 -1(사진 전체)}
    video: null,    // {url, name, duration, w, h, sample:bool}
    fps: 2, busy: false, R: null, thr: 0.5,
  };

  /* ---------- 모델 ---------- */
  function loadScript(src) {
    return new Promise((ok, ng) => {
      const s = document.createElement("script");
      s.src = src; s.crossOrigin = "anonymous"; s.onload = ok; s.onerror = () => ng(new Error("ONNX Runtime을 불러오지 못했습니다(인터넷 연결 확인)"));
      document.head.appendChild(s);
    });
  }

  let modelsP = null;
  function loadModels(onStep) {
    if (modelsP) return modelsP;
    modelsP = (async () => {
      onStep?.("실행 엔진 불러오는 중");
      if (!window.ort) await loadScript(`${ORT_BASE}ort.webgpu.min.js`);
      ort.env.wasm.wasmPaths = ORT_BASE;
      ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
      const eps = navigator.gpu && !/[?&]wasm=1/.test(location.search) ? ["webgpu", "wasm"] : ["wasm"];  // ?wasm=1: CPU 강제(속도 비교용)
      const make = async (url) => {
        try { return { s: await ort.InferenceSession.create(url, { executionProviders: eps }), ep: eps[0] }; }
        catch (e) { if (eps[0] === "wasm") throw e; return { s: await ort.InferenceSession.create(url, { executionProviders: ["wasm"] }), ep: "wasm" }; }
      };
      onStep?.("사람 검출 모델 받는 중 (11MB, 처음 한 번)");
      const d = await make("models/det_yolo11n.onnx");
      onStep?.("ReID 모델 받는 중 (9MB, 처음 한 번)");
      const r = await make("models/reid_osnet.onnx");
      V.det = d.s; V.reid = r.s;
      V.ep = d.ep === "webgpu" || r.ep === "webgpu" ? "WebGPU(그래픽카드)" : "WASM(CPU)";
      $("#v-ep").textContent = `실행 방식: ${V.ep}.`;
    })();
    modelsP.catch(() => { modelsP = null; });
    return modelsP;
  }

  /* ---------- 검출·특징 ---------- */
  async function detect(src, w, h) {
    const s = IMGSZ / Math.max(w, h), nw = Math.round(w * s), nh = Math.round(h * s);
    const c = canvas(IMGSZ, IMGSZ), g = c.getContext("2d", { willReadFrequently: true });
    g.fillStyle = "rgb(114,114,114)"; g.fillRect(0, 0, IMGSZ, IMGSZ);
    g.drawImage(src, 0, 0, w, h, 0, 0, nw, nh);
    const d = g.getImageData(0, 0, IMGSZ, IMGSZ).data, n = IMGSZ * IMGSZ, x = new Float32Array(3 * n);
    for (let i = 0; i < n; i++) { x[i] = d[4 * i] / 255; x[n + i] = d[4 * i + 1] / 255; x[2 * n + i] = d[4 * i + 2] / 255; }
    const out = await V.det.run({ [V.det.inputNames[0]]: new ort.Tensor("float32", x, [1, 3, IMGSZ, IMGSZ]) });
    const o = out[V.det.outputNames[0]].data, boxes = [];
    for (let i = 0; i < o.length; i += 6) {
      if (o[i + 4] < CONF || Math.round(o[i + 5]) !== 0) continue;
      const b = [clamp(o[i] / s, 0, w), clamp(o[i + 1] / s, 0, h), clamp(o[i + 2] / s, 0, w), clamp(o[i + 3] / s, 0, h)];
      if (b[3] - b[1] >= MIN_H * h && b[2] - b[0] >= 4) boxes.push({ b, score: o[i + 4] });
    }
    return boxes;
  }

  const P = 256 * 128;
  async function embed(src, boxes) {
    const out = [];
    for (let k0 = 0; k0 < boxes.length; k0 += 16) {
      const part = boxes.slice(k0, k0 + 16), n = part.length, x = new Float32Array(n * 3 * P);
      const c = canvas(128, 256), g = c.getContext("2d", { willReadFrequently: true });
      part.forEach((b, k) => {
        const [x1, y1, x2, y2] = b.map(Math.floor);
        g.drawImage(src, x1, y1, Math.max(1, x2 - x1), Math.max(1, y2 - y1), 0, 0, 128, 256);
        const d = g.getImageData(0, 0, 128, 256).data, o = k * 3 * P;
        for (let i = 0; i < P; i++) { x[o + i] = d[4 * i] / 255; x[o + P + i] = d[4 * i + 1] / 255; x[o + 2 * P + i] = d[4 * i + 2] / 255; }
      });
      const r = await V.reid.run({ [V.reid.inputNames[0]]: new ort.Tensor("float32", x, [n, 3, 256, 128]) });
      const f = r[V.reid.outputNames[0]].data;
      for (let k = 0; k < n; k++) out.push(f.slice(k * 512, (k + 1) * 512));
    }
    return out;
  }
  const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
  function iou(a, b) {
    const w = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0])), h = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
    const i = w * h; return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9);
  }
  function thumb(src, b, w = 128) {
    const h = w * 2, c = canvas(w, h), g = c.getContext("2d");
    g.fillStyle = "#1E2533"; g.fillRect(0, 0, w, h);
    const bw = b[2] - b[0], bh = b[3] - b[1], s = Math.min(w / bw, h / bh);
    g.drawImage(src, b[0], b[1], bw, bh, (w - bw * s) / 2, (h - bh * s) / 2, bw * s, bh * s);
    return c;
  }

  /* ---------- 입력: 찾을 사람 ---------- */
  function imgFrom(url) {
    return new Promise((ok, ng) => { const im = new Image(); im.onload = () => ok(im); im.onerror = () => ng(new Error("사진을 열 수 없습니다")); im.src = url; });
  }
  async function setQuery(url, name) {
    const im = await imgFrom(url);
    const s = Math.min(1, 1280 / Math.max(im.naturalWidth, im.naturalHeight));
    const c = canvas(Math.round(im.naturalWidth * s), Math.round(im.naturalHeight * s));
    c.getContext("2d").drawImage(im, 0, 0, c.width, c.height);
    V.q = { name, img: c, boxes: [], sel: -1 };
    $("#vq-empty").classList.add("hidden"); $("#vq-prev").classList.remove("hidden");
    drawQuery();
    $("#vq-stat").textContent = "사진 속 사람을 찾는 중…";
    $("#vq-boxes").classList.add("hidden");
    refresh();
    try {
      await loadModels((t) => { $("#vq-stat").textContent = t; });
      const q = V.q;
      const boxes = (await detect(q.img, q.img.width, q.img.height)).sort((a, b) => (b.b[2] - b.b[0]) * (b.b[3] - b.b[1]) - (a.b[2] - a.b[0]) * (a.b[3] - a.b[1]));
      if (q !== V.q) return;
      // 이미 사람만 잘라 낸 사진(상자가 사진 대부분)이면 사진 전체를 쓴다
      const area = q.img.width * q.img.height;
      const big = boxes.length && (boxes[0].b[2] - boxes[0].b[0]) * (boxes[0].b[3] - boxes[0].b[1]) > 0.55 * area;
      q.boxes = big ? [] : boxes.slice(0, 6);
      q.sel = q.boxes.length ? 0 : -1;
      drawQuery(); renderQBoxes();
      $("#vq-stat").textContent = big || !boxes.length ? "사진 전체를 찾을 사람으로 사용합니다" :
        boxes.length === 1 ? "사람 1명 검출됨" : `사람 ${boxes.length}명 검출됨 · 아래에서 찾을 사람을 고르세요`;
    } catch (e) { $("#vq-stat").textContent = `오류: ${e.message}`; }
    refresh();
  }
  function drawQuery(target = $("#vq-canvas")) {
    const q = V.q; if (!q) return;
    target.width = q.img.width; target.height = q.img.height;
    const g = target.getContext("2d");
    g.drawImage(q.img, 0, 0);
    const lw = Math.max(2, q.img.width / 160);
    q.boxes.forEach((o, i) => {
      g.lineWidth = i === q.sel ? lw * 1.6 : lw; g.strokeStyle = i === q.sel ? "#F5B547" : "rgba(255,255,255,.75)";
      g.strokeRect(o.b[0], o.b[1], o.b[2] - o.b[0], o.b[3] - o.b[1]);
    });
  }
  function renderQBoxes() {
    const q = V.q, el = $("#vq-boxes");
    if (!q || q.boxes.length < 2) { el.classList.add("hidden"); return; }
    el.innerHTML = "";
    q.boxes.forEach((o, i) => {
      const bt = document.createElement("button");
      bt.type = "button"; bt.className = i === q.sel ? "on" : ""; bt.setAttribute("aria-pressed", String(i === q.sel));
      bt.setAttribute("aria-label", `사진 속 사람 ${i + 1}`);
      bt.appendChild(thumb(q.img, o.b, 48));
      bt.onclick = () => { q.sel = i; drawQuery(); renderQBoxes(); };
      el.appendChild(bt);
    });
    el.classList.remove("hidden");
  }
  function queryBox() { const q = V.q; return q.sel >= 0 ? q.boxes[q.sel].b : [0, 0, q.img.width, q.img.height]; }

  /* ---------- 입력: 영상 ---------- */
  function loadVideoEl(el, url) {
    return new Promise((ok, ng) => {
      el.onloadedmetadata = () => ok(el); el.onerror = () => ng(new Error("이 브라우저에서 열 수 없는 영상 형식입니다(MP4 H.264 권장)"));
      el.src = url; el.load();
    });
  }
  async function setVideo(url, name, isSample) {
    const el = $("#vv-prev");
    $("#vv-stat").textContent = "영상 여는 중…";
    try {
      await loadVideoEl(el, url);
      V.video = { url, name, duration: el.duration, w: el.videoWidth, h: el.videoHeight, sample: !!isSample };
      el.classList.remove("hidden"); $("#vv-empty").classList.add("hidden");
      el.currentTime = Math.min(1, el.duration / 2);
      $("#vv-stat").textContent = `${name} · ${mmss(el.duration)} · ${el.videoWidth}×${el.videoHeight}` + (el.duration > 180 ? " · 3분이 넘어 오래 걸릴 수 있습니다" : "");
    } catch (e) { V.video = null; $("#vv-stat").textContent = `오류: ${e.message}`; }
    refresh();
  }

  function nFrames() { return V.video ? Math.max(1, Math.floor(V.video.duration * V.fps)) : 0; }
  function refresh() {
    const ok = V.q && V.video && !V.busy;
    $("#v-go").disabled = !ok;
    $("#v-est").textContent = V.video ? `${nFrames()}장 · ${mmss(V.video.duration)} 영상, 초당 ${V.fps}장` : "사진과 영상을 넣어 주세요";
  }

  /* ---------- 분석 ---------- */
  function seek(el, t) {
    return new Promise((ok) => { const f = () => { el.removeEventListener("seeked", f); ok(); }; el.addEventListener("seeked", f); el.currentTime = t; });
  }
  function progress(frac, text) {
    $("#v-prog").classList.remove("hidden");
    $("#v-bar").style.width = `${Math.round(frac * 100)}%`;
    $("#v-ptext").textContent = text;
  }

  async function run() {
    if (V.busy) return;
    V.busy = true; refresh();
    const t0 = performance.now();
    try {
      await loadModels((t) => progress(0.02, t));
      const qb = queryBox(), qf = (await embed(V.q.img, [qb]))[0];
      const el = document.createElement("video");
      el.muted = true; el.playsInline = true; el.preload = "auto";
      await loadVideoEl(el, V.video.url);
      const W = el.videoWidth, H = el.videoHeight, s = Math.min(1, MAX_W / W);
      const fw = Math.round(W * s), fh = Math.round(H * s);
      const fc = canvas(fw, fh), fg = fc.getContext("2d", { willReadFrequently: true });
      const N = nFrames(), frames = [], tracks = [];
      let prev = [], nDet = 0, nEmb = 0;
      for (let k = 0; k < N; k++) {
        const t = Math.min(el.duration - 0.05, k / V.fps);
        await seek(el, t);
        fg.drawImage(el, 0, 0, fw, fh);
        const dets = await detect(fc, fw, fh);
        nDet += dets.length;
        // 인접 프레임 연결(탐욕, IoU 큰 순)
        const pairs = [];
        dets.forEach((d, j) => prev.forEach((p, i) => { const v = iou(d.b, p.b); if (v >= LINK_IOU) pairs.push([v, j, i]); }));
        pairs.sort((a, b) => b[0] - a[0]);
        const tid = new Array(dets.length).fill(-1), usedP = new Set();
        for (const [, j, i] of pairs) if (tid[j] < 0 && !usedP.has(i)) { tid[j] = prev[i].tid; usedP.add(i); }
        const want = [];
        dets.forEach((d, j) => {
          if (tid[j] < 0) { tid[j] = tracks.length; tracks.push({ n: 0, emb: [], t0: t, t1: t }); }
          const tr = tracks[tid[j]];
          if (tr.n % 2 === 0 && tr.emb.length < MAX_EMB) want.push(j);
          tr.n += 1; tr.t1 = t;
        });
        const feats = want.length ? await embed(fc, want.map((j) => dets[j].b)) : [];
        nEmb += want.length;
        want.forEach((j, m) => {
          tracks[tid[j]].emb.push({ sim: dot(feats[m], qf), t, b: dets[j].b, frame: k, th: thumb(fc, dets[j].b) });
        });
        frames.push({ t, dets: dets.map((d, j) => ({ b: d.b, tid: tid[j] })) });
        prev = dets.map((d, j) => ({ b: d.b, tid: tid[j] }));
        progress(0.05 + 0.95 * (k + 1) / N, `프레임 분석 ${k + 1}/${N} · 사람 ${nDet}건 검출`);
      }
      const cands = tracks.map((tr, id) => {
        if (!tr.emb.length) return null;
        const top = [...tr.emb].sort((a, b) => b.sim - a.sim);
        const score = top.slice(0, TOPK).reduce((a, e) => a + e.sim, 0) / Math.min(TOPK, top.length);
        return { id, score, best: top[0], t0: tr.t0, t1: tr.t1, n: tr.n };
      }).filter(Boolean).sort((a, b) => b.score - a.score);
      cands.forEach((c, i) => { c.rank = i + 1; });
      V.R = { frames, cands, fw, fh, scale: s, W, H, N, nDet, nEmb, nTracks: tracks.length, sec: (performance.now() - t0) / 1000, duration: el.duration };
      el.removeAttribute("src"); el.load();
      showResults();
    } catch (e) {
      progress(0, `오류: ${e.message}`);
    } finally {
      V.busy = false; refresh();
    }
  }

  /* ---------- 결과 ---------- */
  function shown() { return V.R.cands.filter((c) => c.score >= V.thr).slice(0, SHOW); }

  function showResults() {
    const R = V.R;
    $("#vu").classList.add("hidden"); $("#vr").classList.remove("hidden");
    const qc = $("#vr-q"), qb = queryBox();
    const th = thumb(V.q.img, qb, 92); qc.width = th.width; qc.height = th.height; qc.getContext("2d").drawImage(th, 0, 0);
    $("#vr-qname").textContent = V.q.name;
    $("#vr-qinfo").textContent = V.q.sel >= 0 ? `사진 속 ${V.q.boxes.length}명 중 1명` : "사진 전체 사용";
    $("#vr-vname").textContent = V.video.name;
    $("#vr-dl").innerHTML = [["길이", mmss(R.duration)], ["분석 프레임", `${R.N}장`], ["검출된 사람", `${R.nDet}건`],
      ["사람 묶음(트랙)", `${R.nTracks}개`], ["처리 시간", `${R.sec.toFixed(1)}초`], ["실행 방식", V.ep.replace(/\(.*\)/, "")]]
      .map(([k, v]) => `<dt>${k}</dt><dd class="mono">${esc(v)}</dd>`).join("");
    const ev = V.sample?.eval;
    $("#vr-evalnote").textContent = V.video.sample && ev ? `참고: 이 예제 영상에서 같은 방식으로 ${ev.n_query}명을 찾아본 결과 ${Math.round(ev.rank1 * 100)}%를 1위로 찾았습니다(주간 사진 → 저조도 영상, 같은 순간 ±2초 제외).` : "";
    const vid = $("#vr-video");
    $("#vr-screen").style.aspectRatio = `${R.W} / ${R.H}`;
    if (vid.dataset.url !== V.video.url) { vid.src = V.video.url; vid.dataset.url = V.video.url; }
    const ov = $("#vr-overlay"); ov.width = R.fw; ov.height = R.fh;
    $("#vr-ticks").innerHTML = [0, 1, 2, 3].map((i) => `<span>${mmss(R.duration * i / 3)}</span>`).join("");
    renderCands();
    const top = shown()[0];
    if (top) goTo(top, false); else drawOverlay();
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function renderCands() {
    const list = shown(), R = V.R;
    $("#vr-count").textContent = `후보 ${list.length}개` + (R.cands.length > list.length ? ` / 전체 ${R.cands.length}개` : "");
    const tl = $("#vr-tl");
    tl.innerHTML = "";
    list.forEach((c) => {
      const b = document.createElement("button");
      b.type = "button"; b.className = `mk ${c.rank <= 3 ? "hi" : "mid"}`;
      b.style.left = `${(c.t0 / R.duration) * 100}%`;
      b.style.width = `max(8px, ${((c.t1 - c.t0) / R.duration) * 100}%)`;
      b.setAttribute("aria-label", `후보 ${c.rank}위 · ${mmss(c.best.t)}`);
      b.onclick = () => goTo(c, true);
      tl.appendChild(b);
    });
    const ph = document.createElement("span"); ph.className = "ph"; ph.id = "vr-ph"; tl.appendChild(ph);
    const box = $("#vr-cands");
    box.innerHTML = list.length ? "" : `<p class="vempty">최소 유사도 ${V.thr.toFixed(2)} 이상인 후보가 없습니다. 왼쪽 슬라이더를 낮춰 보세요.</p>`;
    list.forEach((c) => {
      const b = document.createElement("button");
      b.type = "button"; b.className = `vcand${c.rank === 1 ? " first" : ""}`;
      b.setAttribute("aria-label", `후보 ${c.rank}위, ${mmss(c.best.t)}, 유사도 ${c.score.toFixed(2)}`);
      const ph = document.createElement("div"); ph.className = "ph";
      const img = new Image(); img.alt = ""; img.src = c.best.th.toDataURL("image/jpeg", 0.85); ph.appendChild(img);
      ph.insertAdjacentHTML("beforeend", `<span class="rk">#${c.rank}</span>`);
      b.appendChild(ph);
      b.insertAdjacentHTML("beforeend", `<div class="ln"><span class="mono">${mmss(c.best.t)}</span><span class="mono sc">${c.score.toFixed(2)}</span></div>
        <div class="sbar"><div class="${c.rank <= 3 ? "hi" : "mid"}" style="width:${Math.round(clamp(c.score, 0, 1) * 100)}%"></div></div>
        <span class="span">${mmss(c.t0)} – ${mmss(c.t1)} · ${c.n}프레임</span>`);
      b.onclick = () => goTo(c, true);
      box.appendChild(b);
    });
    drawOverlay();
  }

  function goTo(c, scroll) {
    const vid = $("#vr-video");
    vid.pause();
    vid.currentTime = c.best.t;
    if (scroll) $("#vr-screen").scrollIntoView({ behavior: "smooth", block: "center" });
  }

  function drawOverlay() {
    const R = V.R; if (!R) return;
    const vid = $("#vr-video"), t = vid.currentTime || 0, ov = $("#vr-overlay"), g = ov.getContext("2d");
    g.clearRect(0, 0, ov.width, ov.height);
    const ph = $("#vr-ph"); if (ph) ph.style.left = `${(t / R.duration) * 100}%`;
    let best = null, bd = 1e9;
    for (const f of R.frames) { const d = Math.abs(f.t - t); if (d < bd) { bd = d; best = f; } }
    if (!best || bd > 0.6 / V.fps + 0.05) return;
    const rankOf = new Map(shown().map((c) => [c.id, c]));
    const lw = Math.max(2, ov.width / 320);
    g.font = `600 ${Math.round(Math.max(12, ov.width / 64))}px "JetBrains Mono", monospace`;
    for (const d of best.dets) {
      const c = rankOf.get(d.tid), [x1, y1, x2, y2] = d.b;
      if (!c) { g.lineWidth = lw * 0.7; g.strokeStyle = "rgba(255,255,255,.35)"; g.strokeRect(x1, y1, x2 - x1, y2 - y1); continue; }
      const col = c.rank <= 3 ? "#F5B547" : "#9AA5BC";
      g.lineWidth = lw * (c.rank === 1 ? 1.6 : 1.1); g.strokeStyle = col; g.strokeRect(x1, y1, x2 - x1, y2 - y1);
      const label = `#${c.rank} · ${c.score.toFixed(2)}`, tw = g.measureText(label).width + 12, th = Math.max(18, ov.width / 48);
      g.fillStyle = col; g.fillRect(x1, Math.max(0, y1 - th), tw, th);
      g.fillStyle = "#1B2A4A"; g.fillText(label, x1 + 6, Math.max(0, y1 - th) + th * 0.75);
    }
  }

  function backToUpload() {
    $("#vr").classList.add("hidden"); $("#vu").classList.remove("hidden");
    $("#vr-video").pause(); $("#v-prog").classList.add("hidden");
    window.scrollTo({ top: 0 });
  }

  /* ---------- 초기화 ---------- */
  function bindDrop(drop, input, onFile) {
    input.addEventListener("change", () => { if (input.files[0]) onFile(input.files[0]); input.value = ""; });
    ["dragenter", "dragover"].forEach((e) => drop.addEventListener(e, (ev) => { ev.preventDefault(); drop.classList.add("over"); }));
    ["dragleave", "drop"].forEach((e) => drop.addEventListener(e, () => drop.classList.remove("over")));
    drop.addEventListener("drop", (ev) => { ev.preventDefault(); const f = ev.dataTransfer.files[0]; if (f) onFile(f); });
  }

  window.initVideo = async function initVideo() {
    if (V.inited) return;
    V.inited = true;
    bindDrop($("#vq-drop"), $("#vq-file"), (f) => {
      if (!f.type.startsWith("image/")) { $("#vq-stat").textContent = "사진 파일(JPG, PNG)을 넣어 주세요"; return; }
      setQuery(URL.createObjectURL(f), f.name); markSample(-1);
    });
    bindDrop($("#vv-drop"), $("#vv-file"), (f) => {
      if (!f.type.startsWith("video/")) { $("#vv-stat").textContent = "영상 파일(MP4, WebM)을 넣어 주세요"; return; }
      setVideo(URL.createObjectURL(f), f.name, false);
    });
    $("#v-fps").addEventListener("change", (e) => {
      V.fps = Number(e.target.value);
      $("#v-fps").querySelectorAll("label").forEach((l) => l.classList.toggle("on", l.querySelector("input").checked));
      refresh();
    });
    $("#v-go").onclick = run;
    $("#vr-back1").onclick = backToUpload;
    $("#vr-back2").onclick = () => { backToUpload(); $("#vv-file").click(); };
    $("#vr-thr").addEventListener("input", (e) => { V.thr = Number(e.target.value); $("#vr-thrv").textContent = V.thr.toFixed(2); renderCands(); });
    const vid = $("#vr-video");
    ["timeupdate", "seeked", "loadeddata"].forEach((e) => vid.addEventListener(e, drawOverlay));
    try {
      V.sample = await (await fetch("video/sample.json")).json();
      $("#vq-samples").innerHTML = V.sample.people.map((p, i) =>
        `<button type="button" data-i="${i}" aria-label="예제 인물 ${i + 1}" aria-pressed="false"><img src="video/${esc(p.file)}" alt=""></button>`).join("");
      $("#vq-samples").querySelectorAll("button").forEach((b) => {
        b.onclick = () => { const p = V.sample.people[+b.dataset.i]; setQuery(`video/${p.file}`, `예제 인물 ${+b.dataset.i + 1}`); markSample(+b.dataset.i); };
      });
      // 예제 영상은 통째로 받아 blob URL로 연다: Range 요청을 지원하지 않는 서버에서도 프레임 이동(seek)이 되게
      $("#vv-sample").onclick = async () => {
        $("#vv-stat").textContent = "예제 영상 받는 중 (16MB)…";
        try {
          const blob = await (await fetch(`video/${V.sample.video}`)).blob();
          setVideo(URL.createObjectURL(blob), V.sample.video, true);
        } catch (e) { $("#vv-stat").textContent = "예제 영상을 받지 못했습니다"; }
      };
    } catch (e) { $("#vv-stat").textContent = "예제를 불러오지 못했습니다"; $("#vv-sample").disabled = true; }
    refresh();
  };
  function markSample(i) {
    $("#vq-samples").querySelectorAll("button").forEach((b) => { const on = +b.dataset.i === i; b.classList.toggle("on", on); b.setAttribute("aria-pressed", String(on)); });
  }
})();
