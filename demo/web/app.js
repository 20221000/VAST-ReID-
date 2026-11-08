// VAST-ReID 웹 데모 — 화면은 디자인 시안 기준, 데이터는 demo/server.py API
const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
// 정적 모드(GitHub Pages 공개판, <html data-static="1">): 서버 없이 tools/build_static.py가 미리 계산한 JSON을 읽음
const STATIC = document.documentElement.dataset.static === "1";
const AI_OFF = "공개판에서는 AI 설명을 끕니다 (API 키 비용). 로컬 데모(python demo/server.py)에서 사용할 수 있습니다.";
let TEXT_EX = [];
const getJSON = async (f) => { const r = await fetch(f); if (!r.ok) throw new Error(`${f} ${r.status}`); return r.json(); };
async function staticApi(path, p) {
  switch (path) {
    case "/api/attr_values": return getJSON("api/attr_values.json");
    case "/api/bench": return getJSON("api/bench.json");
    case "/api/samples": return getJSON(`api/samples_${p.level}.json`);
    case "/api/search": {
      const m = `${p.csls ? "c" : ""}${p.kr ? "k" : ""}` || "p";
      return getJSON(`api/search/${p.ds}_${p.qidx}_${p.ds === "night" ? "x" : p.q_level}_${p.backbone}_${m}_${p.adabn ? 1 : 0}.json`);
    }
    case "/api/text": return getJSON(`api/text/${Math.max(0, TEXT_EX.indexOf(p.q))}_${p.g_level}.json`);
    default: return { text: AI_OFF };  // ai_explain, ai_caption
  }
}
const trackSrc = (lv, i) => (STATIC ? `track/${lv}/${i}.jpg` : `/api/track_frame?level=${lv}&i=${i}`);
const TRACK_STEP = STATIC ? 8 : 4;
const api = async (path, params = {}) => {
  if (STATIC) return staticApi(path, params);
  const q = new URLSearchParams(Object.entries(params).filter(([, v]) => v !== "" && v !== null && v !== undefined));
  const r = await fetch(`${path}?${q}`);
  if (!r.ok) throw new Error(`${path} ${r.status}`);
  return r.json();
};
const ICON_OK = '<svg width="11" height="11" viewBox="0 0 12 12" aria-hidden="true"><path d="M2 6.5l2.5 2.5L10 3" fill="none" stroke="currentColor" stroke-width="2.2"></path></svg>';
const ICON_NG = '<svg width="11" height="11" viewBox="0 0 12 12" aria-hidden="true"><path d="M3 3l6 6M9 3l-6 6" fill="none" stroke="currentColor" stroke-width="2.2"></path></svg>';
const LEVEL_KO = { normal: "Normal", low: "Low", extreme: "Extreme" };
let META = null;

/* ---------------- 라우팅 ---------------- */
function route() {
  const v = (location.hash || "#search").slice(1);
  const view = ["search", "text", "bench"].includes(v) ? v : "search";
  $$(".view").forEach((el) => el.classList.toggle("hidden", el.id !== `view-${view}`));
  $$("nav a").forEach((a) => { a.classList.toggle("on", a.dataset.view === view); a.setAttribute("aria-current", a.dataset.view === view ? "page" : "false"); });
  if (view === "bench" && !BENCH) loadBench();
}
window.addEventListener("hashchange", route);

/* ---------------- 1. 이미지 검색 ---------------- */
// q: 선택한 쿼리 {ds: "market"|"msmt"|"night", idx: 그 데이터셋 query 목록 인덱스}, seed: 무작위 샘플 묶음 (없으면 기본 24명)
const S = { level: "extreme", q: null, seed: null, src: null, adabn: false, last: null, selected: null };

function fillAttrSelect(id, key) {
  const el = $(id);
  el.innerHTML = `<option value="">상관없음</option>` + META.values[key].map((v) => `<option value="${esc(v)}">${esc(META.value_ko[v] || v)}</option>`).join("");
}

const DS_NAME = { market: "Market-1501", msmt: "MSMT17", night: "Night600 실제 야간" };
const DS_TAG = { msmt: "MSMT", night: "실제 야간" };

async function loadSamples() {
  const params = { level: S.level };
  if (S.seed != null) params.seed = S.seed;
  const list = await api("/api/samples", params);
  if (!S.q && list.length) S.q = { ds: list[0].ds, idx: list[0].idx };
  const isOn = (s) => !S.src && S.q && s.ds === S.q.ds && s.idx === S.q.idx;
  $("#picker").innerHTML = list.map((s) => `<button type="button" data-ds="${s.ds}" data-i="${s.idx}" class="${isOn(s) ? "on" : ""}"
    aria-label="${DS_NAME[s.ds] || "Market-1501"} 샘플 ${esc(s.name)}"><img src="${esc(s.url)}" alt="">${DS_TAG[s.ds] ? `<span class="dstag${s.ds === "night" ? " night" : ""}">${DS_TAG[s.ds]}</span>` : ""}</button>`).join("");
}

function attrsParams() {
  const p = {};
  ["upper_color", "bag", "lower_color", "sleeve", "lower_type", "hat"].forEach((k) => { p[k] = $(`#a-${k}`).value; });
  return p;
}

async function doSearch() {
  const m = $("#model").value;
  const params = { q_level: S.level, g_level: "normal", adabn: S.adabn, csls: m.includes("csls"), kr: m.includes("kr"),
                   backbone: $("#backbone").value, attr_strength: $("#attr-strength").value,
                   beta: 1.0, lam: 0.5, ...attrsParams() };
  if (S.src) Object.assign(params, { src_gallery: S.src.gidx, src_level: S.src.level, q_level: S.src.level });
  else Object.assign(params, { ds: S.q.ds, qidx: S.q.idx });
  $("#btn-search").disabled = true;
  try {
    const r = await api("/api/search", params);
    S.last = r; S.selected = null;
    renderSearch(r);
  } finally { $("#btn-search").disabled = false; }
}

function renderSearch(r) {
  const q = r.query;
  $("#q-img").src = q.url;
  $("#q-name").textContent = q.name;
  $("#q-badge").textContent = `${q.level}${q.luminance != null ? ` · 평균 밝기 ${q.luminance.toFixed(2)}` : ""}`;
  const a = attrsParams();
  const filt = Object.entries(a).filter(([, v]) => v).map(([k, v]) => `${META.attr_ko[k]} ${META.value_ko[v] || v}`);
  const m = $("#model").value;
  const model = [m.includes("csls") ? "Hub 보정" : null, r.kr ? "재순위화" : null].filter(Boolean).join(" + ") || "외형만";
  $("#kr-note").textContent = r.kr_note || "";
  $("#kr-note").classList.toggle("hidden", !r.kr_note);
  $("#r-sub").textContent = [`${r.dataset || "Market-1501"} 갤러리 ${r.gallery_size.toLocaleString()}장 (${r.g_level})`, `쿼리 ${q.level}`,
    r.backbone, model, S.adabn ? "AdaBN" : null, ...filt,
    filt.length ? `진술 가중치 ${r.attr_weight}/속성${r.attr_strength === "auto" ? " (자동)" : ""}` : null].filter(Boolean).join(" · ");
  $("#s-r1").textContent = r.rank1 ? "적중" : "실패";
  $("#s-r1").style.color = r.rank1 ? "#1E5FD8" : "#A1400B";
  $("#s-ap").textContent = r.ap == null ? "–" : r.ap.toFixed(2);
  $("#s-first").textContent = r.first_correct == null ? "없음" : `${r.first_correct.toLocaleString()}위`;
  $("#r-grid").innerHTML = r.results.map((x, i) => `
    <button type="button" class="card" data-i="${i}" aria-label="${x.rank}위 ${x.ok ? "정답" : "오답"} 유사도 ${x.sim}">
      <div class="photo"><span class="rank mono">${x.rank}</span><img src="${esc(x.url)}" alt=""></div>
      <div class="cbody">
        <div class="row"><span class="mono score" title="코사인 유사도 (순위는 보정 후 점수 기준)">${x.sim.toFixed(3)}</span>
          ${x.ok ? `<span class="badge ok">${ICON_OK}정답</span>` : `<span class="badge ng">${ICON_NG}오답</span>`}</div>
        <small>${esc(x.cam)} · ${esc(x.illum)}</small>
      </div>
    </button>`).join("");
  renderWhy();
  $("#compare-box").classList.add("hidden");
  refit();
}

function renderWhy() {
  const r = S.last;
  if (!r) return;
  const i = S.selected ?? r.results.findIndex((x) => !x.ok);
  if (i < 0) {
    $("#why-title").textContent = "Top-10이 모두 정답입니다";
    $("#why-chips").innerHTML = "";
    return;
  }
  const x = r.results[i];
  $("#why-title").textContent = x.ok ? `${x.rank}위 · 정답` : `오답 ${x.rank}번 실패 원인`;
  $("#why-chips").innerHTML = x.ok ? `<span class="chip">같은 인물 (ID ${x.pid})</span>` : x.causes.map((c) => `<span class="chip">${esc(c)}</span>`).join("");
}

function showCompare(i) {
  const r = S.last, x = r.results[i];
  S.selected = i;
  $$("#r-grid .card").forEach((c) => c.classList.toggle("sel", Number(c.dataset.i) === i));
  renderWhy();
  $("#compare").innerHTML = `
    <figure><div class="photo"><img src="${esc(r.query.url)}" alt="쿼리 (${esc(r.query.level)})"></div><figcaption>쿼리 · ${esc(r.query.level)}</figcaption></figure>
    ${r.query.original ? `<figure><div class="photo"><img src="${esc(r.query.original)}" alt="쿼리 원본"></div><figcaption>쿼리 원본 · Normal</figcaption></figure>` : ""}
    <figure><div class="photo"><img src="${esc(x.url)}" alt="${x.rank}위 결과"></div><figcaption>${x.rank}위 · ${x.ok ? "정답" : "오답"} · ID ${x.pid > 0 ? x.pid : "없음"}</figcaption></figure>
    <p class="foot" style="flex:1 1 220px;margin:0">${esc(r.query.caption)}${r.dataset ? "" : "<br>(쿼리 인물의 정답 속성, 참고용)"}</p>
    ${x.ok ? "" : `<div class="aibox wait" id="ai-explain"><b>AI 오답 설명 · 비전 LLM</b><span>쿼리와 오답 사진을 보고 설명을 만드는 중…</span></div>`}`;
  $("#compare-box").classList.remove("hidden");
  if (!x.ok) loadExplain(r, x);
  refit();
}

function initSearch() {
  ["upper_color", "bag", "lower_color", "sleeve", "lower_type", "hat"].forEach((k) => fillAttrSelect(`#a-${k}`, k));
  const syncSeg = () => $$("#seg-level label").forEach((l) => l.classList.toggle("on", l.querySelector("input").checked));
  syncSeg();
  $$("#seg-level input").forEach((inp) => inp.addEventListener("change", async () => {
    S.level = inp.value; S.src = null; syncSeg(); await loadSamples(); doSearch();
  }));
  $("#q-change").addEventListener("click", async () => {
    const p = $("#picker");
    if (p.classList.contains("hidden")) {  // 열 때마다 Market + MSMT17에서 새로 무작위로 뽑음
      S.seed = Math.floor(Math.random() * 2 ** 31);
      await loadSamples();
    }
    p.classList.toggle("hidden"); refit();
    $("#q-change").setAttribute("aria-expanded", String(!p.classList.contains("hidden")));
  });
  $("#picker").addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    S.q = { ds: b.dataset.ds, idx: Number(b.dataset.i) }; S.src = null;
    $$("#picker button").forEach((x) => x.classList.toggle("on", x === b));
    doSearch();
  });
  // 미세조정 백본은 이미 저조도를 학습해 AdaBN이 오히려 해로움(results/finetune.csv) → 백본에 맞춰 기본값 전환
  $("#attr-strength").addEventListener("change", () => {
    $("#strength-warn").classList.toggle("hidden", $("#attr-strength").value === "auto");
  });
  // 저조도 보정(AdaBN)은 사전학습 OSNet에만: 미세조정 백본은 이미 저조도를 학습해 효과가 없거나(최종 CLIP Extreme 78.0 vs 77.9)
  // 해로움(OSNet 미세조정 68.9 → 47.0, results/finetune*.csv) → 스위치를 끈 채 잠금
  const syncAdabn = () => {
    const base = $("#backbone").value === "base";
    S.adabn = base;
    $("#t-adabn").setAttribute("aria-pressed", String(base));
    $("#t-adabn").disabled = !base;
    $("#adabn-note").classList.toggle("hidden", base);
  };
  $("#backbone").addEventListener("change", syncAdabn);
  syncAdabn();
  $("#t-adabn").addEventListener("click", () => {
    if ($("#t-adabn").disabled) return;
    S.adabn = !S.adabn; $("#t-adabn").setAttribute("aria-pressed", String(S.adabn));
  });
  $("#btn-search").addEventListener("click", doSearch);
  $("#r-grid").addEventListener("click", (e) => { const c = e.target.closest(".card"); if (c) showCompare(Number(c.dataset.i)); });
}

let explainSeq = 0;
async function loadExplain(r, x) {
  const seq = ++explainSeq;
  const res = await api("/api/ai_explain", { q_level: r.query.lvl, q_split: r.query.split, q_name: r.query.name,
    g_level: x.glvl, g_name: x.name, causes: x.causes.join("|") }).catch((e) => ({ error: String(e) }));
  if (seq !== explainSeq || !$("#ai-explain")) return;  // 그 사이 다른 카드를 누름
  $("#ai-explain").classList.remove("wait");
  $("#ai-explain span").textContent = res.text || `설명을 만들지 못했습니다 (${res.error || "응답 없음"})`;
  if (res.model) $("#ai-explain b").textContent = `AI 오답 설명 · ${res.model}`;
}

/* ---------------- 2. 텍스트 검색 ---------------- */
const T = { extra: {}, results: [], sel: 0, level: "normal" };

async function doText(e) {
  if (e) e.preventDefault();
  T.level = $("#t-level").value;
  const r = await api("/api/text", { q: $("#t-q").value, g_level: T.level, llm: !!$("#t-llm")?.checked, ...T.extra });
  T.results = r.results; T.sel = 0;
  $("#t-chips").innerHTML = `<span class="k">추출된 속성</span>` +
    r.chips.map((c) => STATIC ? `<span class="pchip">${esc(c.label)}</span>` : `<button type="button" class="pchip" data-k="${esc(c.key)}" title="눌러서 제거">${esc(c.label)}</button>`).join("") +
    (STATIC ? "" : `<button type="button" class="addchip" id="t-addbtn">+ 속성 추가</button>`);
  $("#t-count").textContent = `${r.results.length}명`;
  $("#t-how").textContent = r.chips.length ? `속성 추출: ${r.how} · 카드를 눌러 상세 보기` : "인식된 속성이 없습니다. 예: 흰색 상의, 검은 바지, 백팩";
  $("#t-grid").innerHTML = r.results.map((x, i) => `
    <button type="button" class="tcard ${i === 0 ? "sel" : ""}" data-i="${i}" aria-pressed="${i === 0}" aria-label="속성 일치 ${x.score}, ID ${x.pid}">
      <div class="photo"><img src="${esc(x.url)}" alt=""></div>
      <div class="row"><span class="mono score">${x.score.toFixed(2)}</span><small style="font-size:12px;color:var(--sub)">${esc(x.illum)}</small></div>
    </button>`).join("");
  renderDetail();
  refit();
}

function renderDetail() {
  const x = T.results[T.sel];
  $("#t-detail").classList.toggle("hidden", !x);
  if (!x) return;
  $$("#t-grid .tcard").forEach((c) => { const on = Number(c.dataset.i) === T.sel; c.classList.toggle("sel", on); c.setAttribute("aria-pressed", String(on)); });
  $("#d-score").textContent = x.score.toFixed(2);
  $("#d-meta").textContent = `ID ${x.pid} · ${x.cam} · ${x.illum}`;
  $("#d-img").src = x.url;
  $("#d-cap").textContent = `속성 기반: ${x.caption}`;
  loadCaption(x);
  const a = x.attrs || {};
  $("#d-up").textContent = [a.upper_color, a.sleeve].filter(Boolean).join(" · ");
  $("#d-low").textContent = [a.lower_color, a.lower_type].filter(Boolean).join(" · ");
  $("#d-bag").textContent = a.bag || "";
  $("#d-hat").textContent = a.hat || "";
}

let capSeq = 0;
async function loadCaption(x) {
  const seq = ++capSeq;
  $("#d-cap-ai").textContent = "사진을 보고 캡션을 만드는 중…";
  $("#d-cap-head").textContent = "AI 시맨틱 캡션";
  const res = await api("/api/ai_caption", { level: T.level, split: "gallery", name: x.name }).catch((e) => ({ error: String(e) }));
  if (seq !== capSeq) return;
  $("#d-cap-ai").textContent = res.text || `캡션을 만들지 못했습니다 (${res.error || "응답 없음"})`;
  if (res.model) $("#d-cap-head").textContent = `AI 시맨틱 캡션 · ${res.model}`;
}

function initText() {
  $("#t-form").addEventListener("submit", doText);
  $("#t-grid").addEventListener("click", (e) => { const c = e.target.closest(".tcard"); if (c) { T.sel = Number(c.dataset.i); renderDetail(); } });
  const keySel = $("#t-add-key"), valSel = $("#t-add-val");
  keySel.innerHTML = Object.keys(META.values).map((k) => `<option value="${k}">${esc(META.attr_ko[k])}</option>`).join("");
  const fillVals = () => { valSel.innerHTML = META.values[keySel.value].map((v) => `<option value="${esc(v)}">${esc(META.value_ko[v] || v)}</option>`).join(""); };
  keySel.addEventListener("change", fillVals); fillVals();
  $("#t-chips").addEventListener("click", (e) => {
    if (e.target.id === "t-addbtn") { $("#t-add").classList.toggle("hidden"); return; }
    const chip = e.target.closest(".pchip");
    if (chip) { T.extra[chip.dataset.k] = ""; doText(); }
  });
  $("#t-add-ok").addEventListener("click", () => { T.extra[keySel.value] = valSel.value; $("#t-add").classList.add("hidden"); doText(); });
  $("#d-research").addEventListener("click", () => {
    const x = T.results[T.sel]; if (!x) return;
    S.src = { gidx: x.gidx, level: T.level };
    location.hash = "#search";
    doSearch();
  });
}

/* ---------------- 3. 벤치마크 ---------------- */
let BENCH = null;
const B = { metric: "rank1", fset: "OSNet", flev: "low", tlev: "extreme", frame: 1, timer: null };
const fmt = (v, d = 1) => (v == null ? "–" : `${v.toFixed(d)}%`);
const sign = (v) => (v == null ? "–" : `${v > 0 ? "+" : "−"}${Math.abs(v).toFixed(1)}%p`);

async function loadBench() {
  BENCH = await api("/api/bench");
  const c = BENCH.cards;
  $("#kpis").innerHTML = `
    <div class="kpi"><span>데이터셋 규모</span><span class="mono v">${c.dataset.persons.toLocaleString()}명 · ${c.dataset.images.toLocaleString()}장</span><small>× 조도 ${c.dataset.levels}단계 (Market-1501 합성)</small></div>
    <div class="kpi"><span>Normal → Extreme 하락폭</span><span class="mono v" style="color:var(--ng)">${sign(c.drop)}</span><small>Rank-1 · 극저조도 쿼리 → 주간 gallery</small></div>
    <div class="kpi"><span>저조도 개선 효과 (Zero-DCE)</span><span class="mono v" style="color:var(--ng)">${sign(c.zerodce_low)}</span><small>Rank-1 · Low 쿼리 → 주간, 오히려 하락</small></div>
    ${c.night ? `<div class="kpi"><span>실제 야간 (Night600)</span><span class="mono v">${fmt(c.night.final)}</span><small>Rank-1 · 합성만 학습 ${fmt(c.night.zero)} → 실제 야간 300명 추가 (학습한 카메라 기준, 새 장소는 12.1 → 라벨 없이 적응 17.8)</small></div>` : ""}
    <div class="kpi dark"><span>제안 방법 효과 (최고 조합)</span><span class="mono v">${sign(c.best_gain_ext ?? c.best_gain)}</span><small>${c.best_gain_ext != null ? `Extreme 쿼리 → 주간 Rank-1 ${fmt(c.best_value_ext)} · Low ${fmt(c.best_value)}` : `Low 쿼리 → 주간 Rank-1 ${fmt(c.best_value)}`}</small></div>`;
  renderBars(); renderFail(); renderTrack(); renderNight();
}

function renderNight() {
  const m = B.metric, rows = BENCH.night || [], ref = BENCH.night_ref || [];
  const box = $("#night-rows");
  if (!box) return;
  $("#n-name").textContent = m === "rank1" ? "Rank-1" : "mAP";
  const bar = (v, cls) => `<div class="bar"><div class="${cls}" style="width:${v == null ? 0 : v * 2.8}%"></div><span class="mono">${fmt(v)}</span></div>`;
  box.innerHTML = rows.map((r) => `
    <div class="mrow"><span class="n">${esc(r.name)}${r.train ? " †" : ""}${r.note ? `<small>${esc(r.note)}</small>` : ""}</span>
      <div class="b">${bar(r[m], r.train ? "c2" : "c1")}</div></div>`).join("") +
    `<div class="mrow"><span class="n">참고: 논문 (Night600 train 학습)<small>${ref.map((r) => esc(r.name)).join(" · ")}</small></span>
      <div class="b"><div class="bar"><div class="c0" style="width:${Math.max(...ref.map((r) => r[m])) * 2.8}%"></div><span class="mono">${ref.map((r) => r[m].toFixed(1)).join(" / ")}</span></div></div></div>`;
}

function renderBars() {
  const m = B.metric;
  $("#m-name").textContent = m === "rank1" ? "Rank-1" : "mAP";
  $("#m-rows").innerHTML = BENCH.rows.map((r) => `
    <div class="mrow"><span class="n">${esc(r.name)}${r.note ? `<small>${esc(r.note)}</small>` : ""}</span>
      <div class="b">${r[m].map((v, i) => `<div class="bar"><div class="c${i}" style="width:${v == null ? 0 : v * 0.85}%"></div><span class="mono"${i === 2 ? ' style="font-weight:700;color:#111827"' : ""}>${fmt(v)}</span></div>`).join("")}</div>
    </div>`).join("");
}

const CAUSE_KO = { distractor: "사람이 아닌/부분 이미지", hub: "hub 이미지 (노이즈를 무늬로 착각)", similar: "유사 의복", other: "외형 특징 손실 (기타)" };
const CAUSE_C = { distractor: "#C7D2E3", hub: "#F5B547", similar: "#4F6FA5", other: "#9CA3AF" };
function renderFail() {
  const rows = BENCH.failure[`${B.fset}|${B.flev}`] || [];
  $("#f-rows").innerHTML = rows.map((r) => `
    <div class="frow"><div class="h"><span>${esc(CAUSE_KO[r.cause])}</span><span class="mono" style="color:var(--sub)">${r.ratio.toFixed(1)}%</span></div>
      <div class="t"><div style="width:${r.ratio}%;background:${CAUSE_C[r.cause]}"></div></div></div>`).join("");
  $("#f-foot").textContent = rows.length ? `Rank-1 오답 ${rows[0].n_wrong.toLocaleString()}건 / 쿼리 3,368건 기준 (results/failure_causes.csv)` : "";
}

function renderTrack() {
  const rows = BENCH.tracking[B.tlev] || [];
  const main = rows.find((r) => r.tracker === "botsort-osnet-adabn" && r.detector === "yolo11m");
  $("#trk-metrics").innerHTML = main ? `<div><small>IDF1</small><span class="mono">${main.IDF1.toFixed(1)}</span></div><div><small>HOTA</small><span class="mono">${main.HOTA.toFixed(1)}</span></div><div><small>ID 전환</small><span class="mono">${main.IDSW}</span></div><div><small>트래커</small><span>BoT-SORT + OSNet(AdaBN)</span></div>` : "";
  $("#trk-table").innerHTML = `<tr><th>트래커 (검출기)</th><th>HOTA</th><th>IDF1</th><th>AssA</th></tr>` +
    rows.map((r) => `<tr><td>${esc(r.tracker)} <small style="color:var(--sub)">${esc(r.detector)}</small></td><td class="mono">${r.HOTA.toFixed(1)}</td><td class="mono">${r.IDF1.toFixed(1)}</td><td class="mono">${r.AssA.toFixed(1)}</td></tr>`).join("");
  $("#trk-img").src = trackSrc(B.tlev, B.frame);
}

function initBench() {
  $$(".mtoggle button").forEach((b) => b.addEventListener("click", () => {
    B.metric = b.dataset.metric; $$(".mtoggle button").forEach((x) => x.setAttribute("aria-pressed", String(x === b))); renderBars(); renderNight();
  }));
  const group = (attr, key, fn) => $$(`[data-${attr}]`).forEach((b) => b.addEventListener("click", () => {
    B[key] = b.dataset[attr]; $$(`[data-${attr}]`).forEach((x) => x.setAttribute("aria-pressed", String(x === b))); fn();
  }));
  group("fset", "fset", renderFail); group("flev", "flev", renderFail); group("tlev", "tlev", renderTrack);
  $("#trk-play").addEventListener("click", () => {
    const playing = !!B.timer;
    if (playing) { clearInterval(B.timer); B.timer = null; }
    else {
      B.timer = setInterval(() => { B.frame = B.frame + TRACK_STEP > 520 ? 1 : B.frame + TRACK_STEP; $("#trk-img").src = trackSrc(B.tlev, B.frame); }, 160);
    }
    $("#trk-play").setAttribute("aria-label", playing ? "트래킹 영상 재생" : "일시정지");
    $("#trk-icon").setAttribute("d", playing ? "M4 2.5v11l9-5.5z" : "M4 2.5h3v11H4zM9 2.5h3v11H9z");
  });
}

/* ---------------- 화면 배율 ----------------
   시안은 폭 1440px 기준. 넓은 화면에서는 레이아웃을 늘리지 않고 전체를 같은 비율로 확대해 시안 비율을 유지. */
function fitZoom() {
  const z = Math.min(1.6, Math.max(1, window.innerWidth / 1480));
  document.documentElement.style.zoom = z.toFixed(3);
}
const refit = fitZoom;
window.addEventListener("resize", fitZoom);
fitZoom();

/* ---------------- 정적 모드: 서버가 필요한 기능은 끄고 안내 ---------------- */
async function initStatic() {
  if (!STATIC) return;
  TEXT_EX = await getJSON("api/text_examples.json");
  const note = (html) => `<p class="foot static-note">${html}</p>`;
  // 목격자 진술 필터: 조합이 너무 많아 미리 계산하지 않음
  ["#a-upper_color", "#a-bag", "#a-lower_color", "#a-sleeve", "#a-lower_type", "#a-hat", "#attr-strength"].forEach((id) => { $(id).disabled = true; });
  $("#a-upper_color").closest("section").insertAdjacentHTML("beforeend",
    note("공개판(GitHub Pages)은 미리 계산한 결과만 보여줘 목격자 진술 필터·AI 설명은 로컬 데모에서만 동작합니다. 샘플은 Market-1501 24명 + Night600 실제 야간 8명입니다."));
  // 텍스트 검색: 자유 문장 대신 미리 계산한 예시 문장
  const inp = $("#t-q");
  inp.insertAdjacentHTML("afterend", `<select id="t-q-ex" aria-label="예시 문장">${TEXT_EX.map((t) => `<option>${esc(t)}</option>`).join("")}</select>`);
  inp.remove();
  $("#t-q-ex").id = "t-q";
  $("#t-q").addEventListener("change", () => doText());
  $("#t-llm").closest("label").replaceWith(Object.assign(document.createElement("span"), { className: "foot", textContent: "공개판은 예시 문장만 · 자유 입력·AI 속성 추출은 로컬 데모" }));
  $("#t-level").addEventListener("change", () => doText());
  $("#d-research").classList.add("hidden");
}

/* ---------------- 시작 ---------------- */
(async () => {
  META = await api("/api/attr_values");
  await initStatic();
  initSearch(); initText(); initBench(); route();
  await loadSamples();
  doSearch();
  doText();
})();
