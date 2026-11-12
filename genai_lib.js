/* genai_lib.js — shared helpers for the Generative-AI playgrounds.
   Tiny MLP + Adam (no dependencies), RNG, canvas / tab / language / quiz helpers. */
(function (root) {
  'use strict';
  var G = {};

  /* ---------- random ---------- */
  G.rng = function (seed) {
    var s = (seed >>> 0) || 1;
    return function () {
      s = (s + 0x6D2B79F5) | 0;
      var t = Math.imul(s ^ (s >>> 15), 1 | s);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  };
  G.gauss = function (rand) {
    var spare = null;
    return function () {
      if (spare !== null) { var v = spare; spare = null; return v; }
      var u = 0; while (u === 0) u = rand();
      var w = rand(), r = Math.sqrt(-2 * Math.log(u)), th = 2 * Math.PI * w;
      spare = r * Math.sin(th);
      return r * Math.cos(th);
    };
  };

  /* ---------- MLP with Adam ---------- */
  var ACT = {
    tanh: [function (x) { return Math.tanh(x); }, function (a) { return 1 - a * a; }],
    relu: [function (x) { return x > 0 ? x : 0; }, function (a) { return a > 0 ? 1 : 0; }],
    lrelu: [function (x) { return x > 0 ? x : 0.1 * x; }, function (a) { return a > 0 ? 1 : 0.1; }],
    sigmoid: [function (x) { return 1 / (1 + Math.exp(-x)); }, function (a) { return a * (1 - a); }],
    linear: [function (x) { return x; }, function () { return 1; }]
  };
  function MLP(sizes, acts, rand) {
    var gs = G.gauss(rand || Math.random);
    this.sizes = sizes; this.L = sizes.length - 1; this.acts = acts; this.t = 0;
    this.W = []; this.b = []; this.gW = []; this.gb = []; this.mW = []; this.vW = []; this.mb = []; this.vb = []; this.a = [null];
    for (var l = 0; l < this.L; l++) {
      var i = sizes[l], o = sizes[l + 1];
      var sc = Math.sqrt((acts[l] === 'relu' || acts[l] === 'lrelu' ? 2 : 2) / (acts[l] === 'relu' || acts[l] === 'lrelu' ? i : i + o));
      var W = new Float64Array(i * o);
      for (var k = 0; k < W.length; k++) W[k] = gs() * sc;
      this.W.push(W); this.b.push(new Float64Array(o));
      this.gW.push(new Float64Array(i * o)); this.gb.push(new Float64Array(o));
      this.mW.push(new Float64Array(i * o)); this.vW.push(new Float64Array(i * o));
      this.mb.push(new Float64Array(o)); this.vb.push(new Float64Array(o));
      this.a.push(new Float64Array(o));
    }
  }
  MLP.prototype.forward = function (x) {
    this.a[0] = x;
    for (var l = 0; l < this.L; l++) {
      var i = this.sizes[l], o = this.sizes[l + 1], W = this.W[l], b = this.b[l], a = this.a[l], out = this.a[l + 1], f = ACT[this.acts[l]][0];
      for (var j = 0; j < o; j++) {
        var s = b[j], off = j * i;
        for (var k = 0; k < i; k++) s += W[off + k] * a[k];
        out[j] = f(s);
      }
    }
    return this.a[this.L];
  };
  /* dout = dLoss/d(output). accum=false → don't touch parameter grads (only return dLoss/dinput). */
  MLP.prototype.backward = function (dout, accum) {
    if (accum === undefined) accum = true;
    var delta = new Float64Array(dout.length), n = this.L;
    var df = ACT[this.acts[n - 1]][1], aout = this.a[n];
    for (var j = 0; j < delta.length; j++) delta[j] = dout[j] * df(aout[j]);
    for (var l = n - 1; l >= 0; l--) {
      var i = this.sizes[l], o = this.sizes[l + 1], W = this.W[l], a = this.a[l];
      var gW = this.gW[l], gb = this.gb[l], da = new Float64Array(i);
      for (var j2 = 0; j2 < o; j2++) {
        var d = delta[j2], off = j2 * i;
        if (accum) { gb[j2] += d; for (var k = 0; k < i; k++) gW[off + k] += d * a[k]; }
        for (var k2 = 0; k2 < i; k2++) da[k2] += W[off + k2] * d;
      }
      if (l > 0) { var dfl = ACT[this.acts[l - 1]][1]; for (var k3 = 0; k3 < i; k3++) da[k3] *= dfl(a[k3]); }
      delta = da;
    }
    return delta;
  };
  MLP.prototype.step = function (lr, scale, clip) {
    scale = scale || 1; this.t++;
    var b1 = 0.9, b2 = 0.999, c1 = 1 - Math.pow(b1, this.t), c2 = 1 - Math.pow(b2, this.t);
    if (b1 === 0) { c1 = 1; }
    for (var l = 0; l < this.L; l++) {
      var P = [this.W[l], this.b[l]], Gr = [this.gW[l], this.gb[l]], M = [this.mW[l], this.mb[l]], V = [this.vW[l], this.vb[l]];
      for (var p = 0; p < 2; p++) {
        var w = P[p], g = Gr[p], m = M[p], v = V[p];
        for (var k = 0; k < w.length; k++) {
          var gk = g[k] * scale;
          if (clip) gk = Math.max(-clip, Math.min(clip, gk));
          m[k] = b1 * m[k] + (1 - b1) * gk; v[k] = b2 * v[k] + (1 - b2) * gk * gk;
          w[k] -= lr * (m[k] / c1) / (Math.sqrt(v[k] / c2) + 1e-8);
          g[k] = 0;
        }
      }
    }
  };
  G.MLP = MLP;

  if (typeof document === 'undefined') { root.GenAI = G; if (typeof module !== 'undefined') module.exports = G; return; }

  /* ---------- DOM helpers (browser only) ---------- */
  var $ = function (id) { return document.getElementById(id); };
  G.$ = $;
  G.lang = function () { return document.documentElement.lang === 'id' ? 'id' : 'en'; };
  G.T = function (en, id) { return G.lang() === 'id' ? id : en; };
  var langCbs = [];
  G.onLang = function (fn) { langCbs.push(fn); };

  G.fit = function (c) {
    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    var w = Math.max(40, c.clientWidth), h = Math.max(40, c.clientHeight);
    if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); }
    var ctx = c.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx: ctx, w: w, h: h };
  };
  G.col = { ink: '#2D2D2D', soft: '#6E6259', moss: '#C17A3E', rust: '#A8392A', teal: '#2F7069', border: '#E4DACB', paper: '#FAF7F2', card: '#FFFDF9', deep: '#8F5620' };
  G.font = '11px "SF Mono",SFMono-Regular,Consolas,monospace';
  function lerp(a, b, t) { return a + (b - a) * t; }
  G.lerp = lerp;
  /* sequential colour map: cream → teal */
  G.seq = function (t) {
    t = Math.max(0, Math.min(1, t));
    return [Math.round(lerp(255, 47, t)), Math.round(lerp(253, 112, t)), Math.round(lerp(249, 105, t))];
  };
  /* diverging: rust (−) ↔ cream ↔ teal (+) */
  G.div = function (t) {
    t = Math.max(-1, Math.min(1, t));
    var a = Math.abs(t), c = t >= 0 ? [47, 112, 105] : [168, 57, 42];
    return [Math.round(lerp(255, c[0], a)), Math.round(lerp(253, c[1], a)), Math.round(lerp(249, c[2], a))];
  };
  G.rgb = function (c, a) { return a === undefined ? 'rgb(' + c[0] + ',' + c[1] + ',' + c[2] + ')' : 'rgba(' + c[0] + ',' + c[1] + ',' + c[2] + ',' + a + ')'; };

  /* simple multi-series line chart */
  G.lineChart = function (c, series, o) {
    o = o || {};
    var f = G.fit(c), ctx = f.ctx, W = f.w, H = f.h, pl = 38, pr = 8, pt = 8, pb = 18;
    ctx.clearRect(0, 0, W, H);
    var lo = Infinity, hi = -Infinity, n = 0;
    series.forEach(function (s) { n = Math.max(n, s.data.length); s.data.forEach(function (v) { if (isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); } }); });
    if (o.min !== undefined) lo = o.min; if (o.max !== undefined) hi = o.max;
    if (!isFinite(lo) || n < 2) { ctx.fillStyle = G.col.soft; ctx.font = G.font; ctx.fillText(o.empty || '…', pl + 6, H / 2); return; }
    if (hi - lo < 1e-9) { hi = lo + 1; }
    var X = function (i) { return pl + (W - pl - pr) * i / (n - 1); }, Y = function (v) { return pt + (H - pt - pb) * (1 - (v - lo) / (hi - lo)); };
    ctx.strokeStyle = G.col.border; ctx.lineWidth = 1; ctx.fillStyle = G.col.soft; ctx.font = G.font;
    for (var g = 0; g <= 3; g++) { var v = lo + (hi - lo) * g / 3, y = Y(v); ctx.beginPath(); ctx.moveTo(pl, y); ctx.lineTo(W - pr, y); ctx.stroke(); ctx.fillText(v.toFixed(Math.abs(hi) < 10 ? 2 : 0), 2, y + 3); }
    series.forEach(function (s) {
      ctx.strokeStyle = s.color; ctx.lineWidth = 1.8; ctx.beginPath();
      s.data.forEach(function (v, i) { var x = X(i * (n - 1) / Math.max(1, s.data.length - 1)), y = Y(v); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
      ctx.stroke();
    });
    if (o.labels) { var x0 = pl + 6; series.forEach(function (s, i) { ctx.fillStyle = s.color; ctx.fillRect(x0, H - 12, 8, 3); ctx.fillStyle = G.col.soft; ctx.fillText(s.label, x0 + 11, H - 7); x0 += 20 + s.label.length * 6.6; }); }
  };

  /* ---------- tabs ---------- */
  G.tabs = function (onChange) {
    var tabs = document.querySelectorAll('.tab');
    function show(name) {
      for (var i = 0; i < tabs.length; i++) tabs[i].classList.toggle('active', tabs[i].getAttribute('data-tab') === name);
      var secs = document.querySelectorAll('.section');
      for (var j = 0; j < secs.length; j++) secs[j].classList.toggle('active', secs[j].id === 'sec-' + name);
      G.active = name;
      if (onChange) onChange(name);
    }
    for (var i = 0; i < tabs.length; i++) tabs[i].addEventListener('click', function () { show(this.getAttribute('data-tab')); });
    G.showTab = show;
    var first = document.querySelector('.tab.active') || tabs[0];
    var h = (location.hash || '').slice(1); for (var k = 0; k < tabs.length; k++) if (tabs[k].getAttribute('data-tab') === h) first = tabs[k];
    show(first.getAttribute('data-tab'));
  };

  /* ---------- quiz ---------- */
  G.quiz = function (boxId, en, id) {
    function build() {
      var box = $(boxId); if (!box) return; box.innerHTML = '';
      (G.lang() === 'id' ? id : en).forEach(function (Q, qi) {
        var d = document.createElement('div'); d.className = 'q';
        d.innerHTML = '<div class="qt">' + (qi + 1) + '. ' + Q.q + '</div>';
        var opts = document.createElement('div'); opts.className = 'opts';
        var exp = document.createElement('div'); exp.className = 'exp'; exp.innerHTML = Q.e;
        Q.o.forEach(function (o, oi) {
          var b = document.createElement('button'); b.className = 'opt'; b.innerHTML = o;
          b.addEventListener('click', function () {
            if (oi === Q.a) { opts.querySelectorAll('.opt').forEach(function (x) { x.classList.remove('wrong'); }); b.classList.add('correct'); exp.style.display = 'block'; }
            else b.classList.add('wrong');
          });
          opts.appendChild(b);
        });
        d.appendChild(opts); d.appendChild(exp); box.appendChild(d);
      });
      G.math(box);
    }
    build(); G.onLang(build);
  };

  /* ---------- math (KaTeX auto-render, optional) ---------- */
  G.math = function (el) {
    try {
      if (window.renderMathInElement) renderMathInElement(el || document.body, {
        delimiters: [{ left: '$$', right: '$$', display: true }, { left: '$', right: '$', display: false }], throwOnError: false
      });
    } catch (e) { }
  };
  window.__mathReady = function () { G.math(document.body); };
  window.addEventListener('load', function () { if (window.renderMathInElement) G.math(document.body); });

  /* ---------- language toggle ---------- */
  (function () {
    var KEY = 'phxLang';
    function apply(l) {
      document.documentElement.lang = l;
      var bb = document.querySelectorAll('[data-lang-btn]');
      for (var i = 0; i < bb.length; i++) bb[i].classList.toggle('on', bb[i].getAttribute('data-lang-btn') === l);
      langCbs.forEach(function (f) { try { f(l); } catch (e) { console.error(e); } });
      if (window.__onLang) { try { window.__onLang(l); } catch (e) { } }
    }
    function set(l) { try { localStorage.setItem(KEY, l); } catch (e) { } apply(l); }
    G.initLang = function () {
      var s = null; try { s = localStorage.getItem(KEY); } catch (e) { }
      apply(s === 'id' ? 'id' : 'en');
      var bt = document.querySelectorAll('[data-lang-btn]');
      for (var i = 0; i < bt.length; i++) bt[i].addEventListener('click', function (e) { e.preventDefault(); set(this.getAttribute('data-lang-btn')); });
      window.setLang = set;
    };
  })();

  /* ---------- mouse helper: position in css px relative to canvas ---------- */
  G.pointer = function (c, handlers) {
    function pos(e) { var r = c.getBoundingClientRect(); var t = e.touches ? e.touches[0] : e; return [t.clientX - r.left, t.clientY - r.top]; }
    var down = false;
    c.addEventListener('pointerdown', function (e) { down = true; try { c.setPointerCapture(e.pointerId); } catch (_) { } if (handlers.down) handlers.down(pos(e), e); });
    c.addEventListener('pointermove', function (e) { var p = pos(e); if (down && handlers.drag) handlers.drag(p, e); else if (handlers.move) handlers.move(p, e); });
    c.addEventListener('pointerup', function (e) { down = false; if (handlers.up) handlers.up(pos(e), e); });
    c.addEventListener('pointerleave', function (e) { if (!down && handlers.leave) handlers.leave(e); });
    c.style.touchAction = 'none';
  };

  root.GenAI = G;
})(typeof window !== 'undefined' ? window : globalThis);
