// Moon Vision front-end: bilingual (English / Arabic), animated astrolabe dial, Lottie loaders.
// Talks to the FastAPI backend on the same origin (or set window.MOON_API before this script).
const API = window.MOON_API || "";
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const REDUCED = matchMedia("(prefers-reduced-motion: reduce)").matches;
const SAMPLES = ["samples/sample_crescent.jpg", "samples/sample_gibbous.jpg", "samples/sample_full.jpg", "samples/sample_quarter.jpg"];

const state = { lang: "en", blob: null, hemisphere: "auto", result: null, health: null, stream: null };
const t = () => I18N[state.lang];

/* ================================================================ language */
function initialLang() {
  try { const s = localStorage.getItem("moonvision.lang"); if (s) return s; } catch (e) {}
  return (navigator.language || "").startsWith("ar") ? "ar" : "en";
}
function setLang(lang) {
  state.lang = lang;
  try { localStorage.setItem("moonvision.lang", lang); } catch (e) {}
  document.documentElement.lang = lang;
  document.documentElement.dir = lang === "ar" ? "rtl" : "ltr";
  $$("[data-lang]").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.lang === lang)));
  $$("[data-i18n]").forEach(el => { const v = t()[el.dataset.i18n]; if (typeof v === "string") el.textContent = v; });
  document.title = t().brand;
  renderStatus();
  if (state.result) render(state.result, { animate: false });
}
$$("[data-lang]").forEach(b => b.addEventListener("click", () => setLang(b.dataset.lang)));

/* ================================================================ starfield */
(function sky() {
  const c = $("#sky"), ctx = c.getContext("2d");
  let W, H, stars = [], meteor = null, mx = 0, my = 0, next = performance.now() + 6000;
  function resize() {
    const dpr = Math.min(devicePixelRatio || 1, 2);
    W = c.width = innerWidth * dpr; H = c.height = innerHeight * dpr;
    const n = Math.round(innerWidth * innerHeight / 3800);
    stars = Array.from({ length: n }, () => ({
      x: Math.random() * W, y: Math.random() * H, r: (Math.random() ** 3 * 1.4 + .3) * dpr,
      z: Math.random(), ph: Math.random() * 6.28, sp: .4 + Math.random() * 1.6,
    }));
  }
  addEventListener("resize", resize); resize();
  addEventListener("pointermove", e => { mx = e.clientX / innerWidth - .5; my = e.clientY / innerHeight - .5; });
  function frame(now) {
    ctx.clearRect(0, 0, W, H);
    for (const s of stars) {
      const tw = REDUCED ? .8 : .55 + .45 * Math.sin(now / 1000 * s.sp + s.ph);
      ctx.globalAlpha = tw * (.35 + s.z * .65);
      ctx.fillStyle = s.z > .85 ? "#cfdcff" : "#f3ecd9";
      ctx.beginPath(); ctx.arc(s.x - mx * 14 * s.z, s.y - my * 14 * s.z, s.r, 0, 6.283); ctx.fill();
    }
    if (!REDUCED) {
      if (!meteor && now > next) {
        meteor = { x: Math.random() * W * .7 + W * .2, y: Math.random() * H * .3, vx: -W / 90, vy: H / 160, life: 0 };
        next = now + 7000 + Math.random() * 9000;
      }
      if (meteor) {
        meteor.life++; meteor.x += meteor.vx * .1; meteor.y += meteor.vy * .1;
        const a = Math.max(0, 1 - meteor.life / 70);
        const g = ctx.createLinearGradient(meteor.x, meteor.y, meteor.x - meteor.vx * 1.4, meteor.y - meteor.vy * 1.4);
        g.addColorStop(0, `rgba(243,236,217,${a})`); g.addColorStop(1, "rgba(243,236,217,0)");
        ctx.globalAlpha = 1; ctx.strokeStyle = g; ctx.lineWidth = 1.6;
        ctx.beginPath(); ctx.moveTo(meteor.x, meteor.y); ctx.lineTo(meteor.x - meteor.vx * 1.4, meteor.y - meteor.vy * 1.4); ctx.stroke();
        if (meteor.life > 70) meteor = null;
      }
      requestAnimationFrame(frame);
    }
  }
  requestAnimationFrame(frame);
})();

/* ================================================================ astrolabe dial */
const R = 112;
const dial = { E: 40, target: null, speed: 360 / 16, mode: "idle" };   // E = elongation 0..360 (0 new, 180 full)
(function buildDial() {
  const ticks = $("#ticks"), labels = $("#ring-labels"), ns = "http://www.w3.org/2000/svg";
  for (let d = 0; d < 360; d += 5) {
    const major = d % 30 === 0, a = (d - 90) * Math.PI / 180;
    const r1 = 190, r2 = major ? 172 : 182;
    const l = document.createElementNS(ns, "line");
    l.setAttribute("x1", r1 * Math.cos(a)); l.setAttribute("y1", r1 * Math.sin(a));
    l.setAttribute("x2", r2 * Math.cos(a)); l.setAttribute("y2", r2 * Math.sin(a));
    if (major) l.classList.add("major");
    ticks.appendChild(l);
    if (major) {
      const tx = document.createElementNS(ns, "text");
      tx.setAttribute("x", 161 * Math.cos(a)); tx.setAttribute("y", 161 * Math.sin(a));
      tx.textContent = d; labels.appendChild(tx);
    }
  }
})();

function moonPaths(k, waxing) {
  k = Math.max(0, Math.min(1, k));
  const rx = R * Math.abs(1 - 2 * k), sweep = k < .5 ? 0 : 1;
  if (k < .004) return { lit: "", term: "" };
  if (k > .996) return { lit: `M0,${-R} A${R},${R} 0 1 1 0,${R} A${R},${R} 0 1 1 0,${-R}Z`, term: "" };
  return {
    lit: `M0,${-R} A${R},${R} 0 0 1 0,${R} A${rx},${R} 0 0 ${sweep} 0,${-R}Z`,
    term: `M0,${R} A${rx},${R} 0 0 ${sweep} 0,${-R}`,
    flip: !waxing,
  };
}
function drawDial(E) {
  E = ((E % 360) + 360) % 360;
  const k = (1 - Math.cos(E * Math.PI / 180)) / 2, waxing = E < 180;
  const p = moonPaths(k, waxing), tf = waxing ? "" : "scale(-1,1)";
  const lit = $("#lit-path"), term = $("#term-shade");
  lit.setAttribute("d", p.lit); lit.setAttribute("transform", tf);
  term.setAttribute("d", p.term); term.setAttribute("transform", tf);
  const clip = $("#lit-clip"); clip.setAttribute("d", p.lit); clip.setAttribute("transform", tf);
  $("#pointer").style.transform = `rotate(${E}deg)`;
}
let lastT = performance.now();
function dialLoop(now) {
  const dt = Math.min(.05, (now - lastT) / 1000); lastT = now;
  if (dial.target != null) {
    const diff = ((dial.target - dial.E + 540) % 360) - 180;          // shortest way round
    dial.E += diff * Math.min(1, dt * 3.2);
    if (Math.abs(diff) < .05) { dial.E = dial.target; }
  } else if (!REDUCED) {
    dial.E += dial.speed * dt * (dial.mode === "scan" ? 5 : 1);
  }
  drawDial(dial.E);
  requestAnimationFrame(dialLoop);
}
$("#pointer").style.transition = "none";
requestAnimationFrame(dialLoop);
function setDialMode(mode) { dial.mode = mode; $("#astro").dataset.mode = mode; if (mode !== "done") dial.target = null; }

/* ================================================================ Lottie */
function lottieIn(id, path) {
  if (!window.lottie) return null;
  const a = lottie.loadAnimation({ container: $(id), renderer: "svg", loop: true, autoplay: !REDUCED, path });
  if (REDUCED) a.addEventListener("DOMLoaded", () => a.goToAndStop(40, true));
  return a;
}
lottieIn("#idle-lottie", "lottie/idle.json");
const scanAnim = lottieIn("#scan-lottie", "lottie/scan.json");

/* ================================================================ health */
function renderStatus() {
  const s = $("#status"), h = state.health;
  if (h === null) { s.dataset.state = "wait"; s.textContent = t().statusWait; }
  else if (h === false) { s.dataset.state = "bad"; s.textContent = t().statusOffline; }
  else if (h.cnn_loaded) { s.dataset.state = "ready"; s.textContent = t().statusReady; }
  else { s.dataset.state = "bad"; s.textContent = t().statusGeo; }
}
fetch(`${API}/api/health`).then(r => r.json()).then(h => { state.health = h; renderStatus(); })
  .catch(() => { state.health = false; renderStatus(); });

/* ================================================================ inputs */
$$(".tabs button").forEach(b => b.addEventListener("click", () => {
  $$(".tabs button").forEach(x => x.classList.toggle("on", x === b));
  $("#tab-upload").hidden = b.dataset.tab !== "upload";
  $("#tab-camera").hidden = b.dataset.tab !== "camera";
  if (b.dataset.tab !== "camera") stopCamera();
}));
function segmented(sel, cb) {
  $$(`${sel} button`).forEach(b => b.addEventListener("click", () => {
    $$(`${sel} button`).forEach(x => x.classList.toggle("on", x === b)); cb(b.dataset.v);
  }));
}
segmented("#hemi", v => state.hemisphere = v);
segmented("#viewsel", v => showViz(v));

function setImage(blob, fromCamera = false) {
  state.blob = blob; state.exif = {};
  if (fromCamera) $("#when").value = toLocalInput(new Date());
  else readExif(blob).then(x => {
    state.exif = x;
    if (x.date && !$("#when").value) $("#when").value = toLocalInput(x.date);
  });
  const img = $("#preview");
  img.src = URL.createObjectURL(blob); img.hidden = false;
  img.style.animation = "none"; img.offsetHeight; img.style.animation = "";
  $("#drop-hint").hidden = true; $("#go").disabled = false; $("#err").hidden = true;
}
$("#file").addEventListener("change", e => e.target.files[0] && setImage(e.target.files[0]));
$("#snap").addEventListener("change", e => e.target.files[0] && setImage(e.target.files[0]));
const drop = $("#drop");
["dragenter", "dragover"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", e => { const f = e.dataTransfer.files[0]; if (f && f.type.startsWith("image/")) setImage(f); });
$("#sample").addEventListener("click", async () => {
  const url = SAMPLES[Math.floor(Math.random() * SAMPLES.length)];
  setImage(await (await fetch(url)).blob());
});

$("#cam-start").addEventListener("click", async () => {
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment", width: { ideal: 1920 } }, audio: false });
    $("#video").srcObject = state.stream; $("#cam-shot").disabled = false;
  } catch (e) { showError(t().camError); }
});
$("#cam-shot").addEventListener("click", () => {
  const v = $("#video"), c = $("#grab");
  c.width = v.videoWidth; c.height = v.videoHeight; c.getContext("2d").drawImage(v, 0, 0);
  c.toBlob(b => { setImage(b, true); $('.tabs button[data-tab="upload"]').click(); }, "image/jpeg", .95);
});
function stopCamera() { if (state.stream) { state.stream.getTracks().forEach(x => x.stop()); state.stream = null; $("#cam-shot").disabled = true; } }

/* ================================================================ analyse */
$("#go").addEventListener("click", analyse);
async function analyse() {
  if (!state.blob) return;
  const go = $("#go"); go.disabled = true; go.textContent = t().analysing; $("#err").hidden = true;
  setDialMode("scan");
  const prog = $("#progress"), steps = $$("#progress li");
  prog.hidden = false; scanAnim && scanAnim.play();
  steps.forEach(s => s.className = "");
  let i = 0; steps[0].className = "active";
  const timer = setInterval(() => { if (i < steps.length - 1) { steps[i].className = "done"; steps[++i].className = "active"; } }, 420);

  const fd = new FormData();
  const upload = await prepareUpload(state.blob);
  fd.append("file", upload, state.blob.name || "moon.jpg");
  for (const k of ["exposure_time", "f_number", "iso"]) if (state.exif?.[k]) fd.append(k, state.exif[k]);
  fd.append("hemisphere", state.hemisphere);
  if ($("#when").value) fd.append("taken_at", new Date($("#when").value).toISOString());
  const minWait = new Promise(r => setTimeout(r, REDUCED ? 0 : 1300));    // let the dial "scan" for a moment
  try {
    const res = await fetch(`${API}/api/analyze`, { method: "POST", body: fd });
    const j = await res.json();
    if (!res.ok) throw new Error(j.detail || res.statusText);
    await minWait;
    clearInterval(timer); steps.forEach(s => s.className = "done");
    state.result = j; render(j, { animate: true });
    setTimeout(() => { prog.hidden = true; scanAnim && scanAnim.pause(); }, 600);
  } catch (e) {
    clearInterval(timer); prog.hidden = true; setDialMode("idle"); showError(e.message);
  } finally { go.disabled = false; go.textContent = t().analyse; }
}
function showError(m) { const e = $("#err"); e.textContent = m; e.hidden = false; }

/* ================================================================ upload prep
   Vercel functions accept at most 4.5 MB per request. Large photos are shrunk in
   the browser (max 2048 px, JPEG). Shrinking drops EXIF, so the date and exposure
   are read first and sent as separate form fields. */
const MAX_UPLOAD = 3.5 * 1024 * 1024, MAX_SIDE = 2048;
async function readExif(blob) {
  try {
    const e = await exifr.parse(blob);
    if (!e) return {};
    return { date: e.DateTimeOriginal instanceof Date ? e.DateTimeOriginal : null,
             exposure_time: e.ExposureTime, f_number: e.FNumber, iso: Array.isArray(e.ISO) ? e.ISO[0] : e.ISO };
  } catch (err) { return {}; }
}
async function prepareUpload(blob) {
  try {
    const bmp = await createImageBitmap(blob, { imageOrientation: "from-image" });
    const big = Math.max(bmp.width, bmp.height);
    if (blob.size <= MAX_UPLOAD && big <= 3000) { bmp.close(); return blob; }
    const s = Math.min(1, MAX_SIDE / big);
    const c = document.createElement("canvas");
    c.width = Math.round(bmp.width * s); c.height = Math.round(bmp.height * s);
    c.getContext("2d").drawImage(bmp, 0, 0, c.width, c.height); bmp.close();
    for (const q of [0.92, 0.85, 0.75]) {
      const out = await new Promise(r => c.toBlob(r, "image/jpeg", q));
      if (out && out.size <= MAX_UPLOAD) return out;
    }
  } catch (err) { /* e.g. HEIC the browser can't decode: send as is */ }
  return blob;
}

/* ================================================================ render */
function render(r, { animate }) {
  const L = t(), ph = r.phase, g = r.geometry, label = L.phases[ph.class];
  const waxing = ph.class.startsWith("waxing") || ph.class === "first_quarter";

  // --- the dial settles on the measured phase
  const k = g.moon_found ? g.illuminated_fraction : 0;
  const alpha = Math.acos(2 * k - 1) * 180 / Math.PI;
  const E = ph.class === "full_moon" ? 180 : ph.class === "new_moon" ? 0 : (waxing ? 180 - alpha : 180 + alpha);
  setDialMode("done"); dial.target = E; if (!animate) dial.E = E;
  $("#astro-title").textContent = label;
  $("#astro-sub").textContent = g.moon_found ? L.litLine(num(pct(k)), num(g.phase_angle_deg + "°")) : L.noMoonLine;
  if (animate) { const fig = $("#astro"); fig.dataset.mode = "scan"; fig.offsetHeight; fig.dataset.mode = "done"; }

  // --- chips
  const chips = [chip(L.chipConf, `${Math.round(ph.confidence * 100)}%`, ph.confidence > .6 ? "ok" : "warn")];
  if (r.cnn.available) chips.push(chip(L.chipCnn, L.phases[r.cnn.top]));
  chips.push(chip(L.chipWax, L.waxBy[ph.waxing_decided_by_code] || ph.waxing_decided_by));
  if (r.ephemeris) chips.push(chip(L.chipCal, L.phases[r.ephemeris.phase_class], r.ephemeris_agrees ? "ok" : "warn"));
  const B = r.brightness.pixel_measurement;
  (r.warning_codes || []).forEach(c => {
    const long = typeof L.warnLong[c] === "function" ? L.warnLong[c](B ? B.saturated_percent : "") : L.warnLong[c];
    chips.push(`<span class="chip warn" title="${esc(long)}"><b>${esc(L.warn[c])}</b></span>`);
  });
  $("#chips").innerHTML = chips.join("");

  // --- what the model saw
  showViz($("#viewsel .on").dataset.v);
  $("#geo").innerHTML = g.moon_found ? kv([
    [L.geo.center, g.disk_center_px.join(", ")],
    [L.geo.radius, g.disk_radius_px],
    [L.geo.lit, pct(k)],
    [L.geo.side, `${g.lit_side_angle_deg}${L.geo.sideUnit}`],
    [L.geo.residual, `${(g.fit_residual * 100).toFixed(2)}${L.geo.ofR}`],
  ]) : "";

  // --- probabilities (fused bar + thin marker for the neural network's own vote)
  const cnnP = r.cnn.probabilities || {};
  $("#probs").innerHTML = Object.entries(ph.probabilities).map(([c, p], i) => `
    <div class="prob ${i === 0 ? "lead" : ""}" title="${esc(L.fusedLbl)} ${(p * 100).toFixed(1)}% · ${esc(L.cnnLbl)} ${((cnnP[c] || 0) * 100).toFixed(1)}%">
      <span class="name">${esc(L.phases[c])}</span>
      <span class="track"><span class="fill" data-w="${(p * 100).toFixed(2)}"></span>${r.cnn.available ? `<span class="cnn" data-x="${((cnnP[c] || 0) * 100).toFixed(2)}"></span>` : ""}</span>
      <span class="pct">${(p * 100).toFixed(1)}%</span>
    </div>`).join("");
  const grow = () => {
    $$("#probs .fill").forEach(f => f.style.width = f.dataset.w + "%");
    $$("#probs .cnn").forEach(m => m.style.insetInlineStart = `calc(${m.dataset.x}% - 1px)`);
  };
  animate ? requestAnimationFrame(() => requestAnimationFrame(grow)) : grow();
  $("#probs-cap").textContent = r.cnn.available ? L.probCap : L.probCapGeo;

  // --- brightness A
  const A = r.brightness.class_reference;
  $("#a-num").textContent = `${A.magnitude_range[0]} … ${A.magnitude_range[1]}`;
  $("#a-unit").textContent = L.magRange;
  $("#a-kv").innerHTML = kv([
    [L.typical, `${L.mag} ${num(A.magnitude_typical)}`],
    [L.illum, `${fmtLux(A.lux_range[0])} – ${fmtLux(A.lux_range[1])}`],
    [L.vsFull, `${A.percent_of_full_range[0]}% – ${A.percent_of_full_range[1]}%`],
    [L.band, `${pct(A.illuminated_fraction_range[0])} – ${pct(A.illuminated_fraction_range[1])}`],
  ]);

  // --- brightness B
  if (B) {
    if (B.calibrated) { count($("#b-num"), B.surface_luminance_cd_m2, v => Math.round(v).toLocaleString("en"), animate); $("#b-unit").textContent = L.bLum; }
    else { count($("#b-num"), B.relative_index_0_100, v => v.toFixed(1), animate); $("#b-unit").textContent = L.bIndex; }
    $("#b-kv").innerHTML = kv([
      [L.bMean, B.mean_linear_luminance], [L.bMean255, B.mean_gray_0_255],
      [L.bContrast, `${B.contrast_vs_sky}×`], [L.bFlux, B.integrated_flux_pixels.toLocaleString("en")],
      [L.bClip, `${B.saturated_percent}%`],
    ]);
    const w = $("#b-warn"); w.hidden = !B.warnings.length; if (B.warnings.length) w.textContent = L.warnLong.overexposed(B.saturated_percent);
  } else { $("#b-num").textContent = "—"; $("#b-unit").textContent = L.bNone; $("#b-kv").innerHTML = ""; $("#b-warn").hidden = true; }

  // --- brightness C
  const C = r.brightness.photometric_model;
  count($("#c-num"), C.lux, fmtLux, animate);
  $("#c-unit").textContent = L.cUnit(num(C.magnitude));
  $("#c-kv").innerHTML = kv([
    [L.vsFull, `${C.percent_of_full}%`], [L.phaseAngle, `${C.phase_angle_deg}°`],
    [L.litMeasured, pct(C.illuminated_fraction)], [L.distCorr, C.distance_corrected ? L.yes : L.noDate],
  ]);
  renderScale(r);

  // --- calendar
  const Eph = r.ephemeris; $("#eph-panel").hidden = !Eph;
  if (Eph) {
    const date = new Date(Eph.utc).toLocaleString(state.lang === "ar" ? "ar-EG-u-nu-latn" : "en-GB", { dateStyle: "long", timeStyle: "short" });
    $("#eph-text").textContent = [
      L.ephText(date, L.ephSrc[r.ephemeris_source_code] || "", L.phases[Eph.phase_class], num(pct(Eph.illuminated_fraction))),
      r.ephemeris_agrees ? L.ephAgree : L.ephDisagree(label),
      r.fraction_difference != null ? L.ephDiff(num((r.fraction_difference * 100).toFixed(1))) : "",
    ].join(" ");
    $("#eph-kv").innerHTML = kv([
      [L.eph.age, `${Eph.age_days} ${L.eph.days}`], [L.eph.angle, `${Eph.phase_angle_deg}°`],
      [L.eph.dist, `${Eph.distance_km.toLocaleString("en")} ${L.eph.km}`], [L.eph.mag, Eph.magnitude],
    ]);
  }
  $("#raw").textContent = JSON.stringify({ ...r, images: r.images ? "[base64 images omitted]" : undefined }, null, 2);

  const main = $("#results"); main.hidden = false;
  if (animate) {
    main.classList.remove("reveal"); main.offsetHeight; main.classList.add("reveal");
    if (innerWidth < 920) setTimeout(() => $("#astro").scrollIntoView({ behavior: REDUCED ? "auto" : "smooth" }), 200);
  }
}

function showViz(which) {
  const r = state.result; if (!r) return;
  const L = t(), img = $("#viz"), im = r.images;
  if (!im) { img.removeAttribute("src"); $("#viz-cap").textContent = L.capNone; return; }
  const src = which === "heat" ? im.heatmap_png : im.overlay_png;
  if (img.getAttribute("src") !== src) {
    img.classList.add("swap");
    setTimeout(() => { img.src = src; img.classList.remove("swap"); }, img.getAttribute("src") ? 180 : 0);
  }
  $("#viz-cap").textContent = which === "heat" ? L.capHeat : L.capSeg;
}

function renderScale(r) {
  const L = t(), A = r.brightness.class_reference, C = r.brightness.photometric_model, E = r.ephemeris;
  const pos = p => Math.max(0, Math.min(100, (Math.log10(Math.max(p, .1)) + 1) / 3 * 100));   // 0.1% … 100%
  const rows = [
    `<div class="cbar"><span>${esc(L.scaleA)}</span><div class="track">
      <div class="range" style="left:${pos(A.percent_of_full_range[0])}%;width:${pos(A.percent_of_full_range[1]) - pos(A.percent_of_full_range[0])}%"></div>
      <div class="dot" style="left:${pos(A.percent_of_full_typical)}%"></div></div><span class="val">${A.percent_of_full_typical}%</span></div>`,
    `<div class="cbar"><span>${esc(L.scaleC)}</span><div class="track"><div class="dot" style="left:${pos(C.percent_of_full)}%;animation-delay:.15s"></div></div><span class="val">${C.percent_of_full}%</span></div>`,
  ];
  if (E) {
    const pe = 100 * Math.pow(10, -0.4 * (E.magnitude + 12.73));
    rows.push(`<div class="cbar"><span>${esc(L.scaleE)}</span><div class="track"><div class="dot az" style="left:${pos(pe)}%;animation-delay:.3s"></div></div><span class="val">${pe.toFixed(1)}%</span></div>`);
  }
  rows.push(`<div class="axis"><span></span><div><span>0.1%</span><span>1%</span><span>10%</span><span>100%</span></div><span></span></div>`);
  $("#compare-bars").innerHTML = rows.join("") + `<p class="cap">${esc(L.scaleCap)}</p>`;
}

/* ================================================================ utils */
function count(el, to, fmt, animate) {
  if (!animate || REDUCED) { el.textContent = fmt(to); return; }
  const t0 = performance.now(), d = 1100;
  (function step(now) {
    const p = Math.min(1, (now - t0) / d), e = 1 - Math.pow(1 - p, 3);
    el.textContent = fmt(to * e); if (p < 1) requestAnimationFrame(step);
  })(t0);
}
function chip(k, v, cls = "") { return `<span class="chip ${cls}">${esc(k)}: <b><bdi>${esc(String(v))}</bdi></b></span>`; }
const num = (x) => `\u2066${x}\u2069`;   // keep numbers (minus signs, %) left-to-right inside Arabic text
function kv(rows) { return rows.map(([k, v]) => `<dt>${esc(k)}</dt><dd><bdi>${esc(String(v))}</bdi></dd>`).join(""); }
function pct(x) { return (x * 100).toFixed(1) + "%"; }
function fmtLux(x) { const L = t(); return x >= .01 ? `${x.toFixed(3)} ${L.lux}` : `${(x * 1000).toFixed(2)} ${L.mlux}`; }
function esc(s) { return String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }
function toLocalInput(d) { return new Date(d.getTime() - d.getTimezoneOffset() * 60000).toISOString().slice(0, 16); }

setLang(initialLang());
