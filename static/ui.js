/* VoiceForge Studio — shared shell.
   Draws the sidebar (navigation + engine switcher) on every page and exposes small helpers on window.VF.
   Pages listen for `vf:engine` events (detail = /api/engine status) to react to engine changes. */
(function () {
  const ICONS = {
    home: '<path d="M3 10.5 12 3l9 7.5V20a1 1 0 0 1-1 1h-5v-6h-6v6H4a1 1 0 0 1-1-1z"/>',
    speak: '<rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5 11a7 7 0 0 0 14 0M12 18v3"/>',
    scenes: '<rect x="3" y="5" width="18" height="14" rx="2"/><path d="M3 9h18M8 5v4M16 5v4"/>',
    cast: '<circle cx="9" cy="8" r="3.2"/><path d="M3.5 19a5.5 5.5 0 0 1 11 0"/><circle cx="17" cy="9" r="2.4"/><path d="M15.5 14.2A4.5 4.5 0 0 1 21 18.5"/>',
    voices: '<path d="M4 6h16M4 12h10M4 18h7"/><circle cx="18" cy="16" r="3"/>',
    clone: '<path d="M12 3a4 4 0 0 1 4 4v4a4 4 0 0 1-8 0V7a4 4 0 0 1 4-4z"/><path d="M19 4l2-1M19 9h2M5 4 3 3M5 9H3"/><path d="M8 20h8M12 15v5"/>',
    history: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
    engines: '<rect x="5" y="5" width="14" height="14" rx="2"/><rect x="9" y="9" width="6" height="6" rx="1"/><path d="M9 2v3M15 2v3M9 19v3M15 19v3M2 9h3M2 15h3M19 9h3M19 15h3"/>',
    docs: '<path d="M6 3h9l4 4v14H6z"/><path d="M14 3v5h5M9 13h7M9 17h7"/>',
    copy: '<rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V5a1 1 0 0 0-1-1H5a1 1 0 0 0-1 1v10a1 1 0 0 0 1 1h3"/>',
  };
  const icon = (n) => `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">${ICONS[n] || ''}</svg>`;

  const NAV = [
    { section: null, items: [{ href: '/', label: 'Home', icon: 'home' }] },
    { section: 'Create', items: [
      { href: '/studio', label: 'Speak', icon: 'speak' },
      { href: '/cast', label: 'Casting Desk', icon: 'cast' },
      { href: '/scenes', label: 'Scenes', icon: 'scenes' },
    ] },
    { section: 'Voices', items: [
      { href: '/voices', label: 'Voice Library', icon: 'voices' },
      { href: '/clone', label: 'Clone Studio', icon: 'clone' },
    ] },
    { section: 'Studio', items: [
      { href: '/history', label: 'Takes', icon: 'history' },
      { href: '/engines', label: 'Engines', icon: 'engines' },
    ] },
  ];

  // Short labels for the switcher; anything new in MODEL_REGISTRY falls back to its own name
  const SHORT = {
    qwen3_voicedesign: 'Qwen3 · Voice Design',
    qwen3_clone: 'Qwen3 · Clone',
    maya1: 'Maya-1 · Acting',
    chatterbox: 'Chatterbox · Clone',
  };

  const VF = {
    engine: null,
    models: [],
    shortName: (id) => SHORT[id] || (VF.models.find((m) => m.id === id) || {}).name || id || 'None',
    esc(str) {
      return String(str ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
    },
    async api(path, opts = {}) {
      const res = await fetch(path, { headers: { 'Content-Type': 'application/json' }, ...opts });
      let data = null;
      try { data = await res.json(); } catch (_) { /* non-JSON */ }
      if (!res.ok) {
        const d = data && data.detail;
        throw new Error(typeof d === 'string' ? d : (d ? JSON.stringify(d) : `HTTP ${res.status}`));
      }
      return data;
    },
    toast(msg, ms = 3800) {
      let t = document.getElementById('vf-toast');
      if (!t) { t = document.createElement('div'); t.id = 'vf-toast'; t.className = 'toast'; document.body.appendChild(t); }
      t.textContent = msg;
      t.classList.add('show');
      clearTimeout(VF._toastTimer);
      VF._toastTimer = setTimeout(() => t.classList.remove('show'), ms);
    },
    async refreshEngine() {
      try {
        const st = await VF.api('/api/engine');
        VF._setEngine(st);
      } catch (_) {
        VF._setEngine(null);
      }
      return VF.engine;
    },
    async loadModel(id) {
      if (!id) return;
      VF._setEngine({ ...(VF.engine || {}), loading_model_id: id, status_message: `Loading ${VF.shortName(id)}…` });
      try {
        const st = await VF.api('/api/model/load', { method: 'POST', body: JSON.stringify({ model_id: id }) });
        VF._setEngine(st);
        VF.toast(st.status_message || `Loaded ${VF.shortName(id)}`);
      } catch (e) {
        VF.toast(`Couldn't load ${VF.shortName(id)}: ${e.message}`, 6000);
        await VF.refreshEngine();
        throw e;
      }
      return VF.engine;
    },
    async unloadModel() {
      const st = await VF.api('/api/model/unload', { method: 'POST' });
      VF._setEngine(st);
      VF.toast('Model unloaded, memory freed');
      return st;
    },
    async copyLLMGuide() {
      try {
        const d = await VF.api('/api/llm-guide');
        await navigator.clipboard.writeText(d.system_prompt_markdown || JSON.stringify(d, null, 2));
        VF.toast('Copied the LLM system prompt to the clipboard');
      } catch (e) {
        VF.toast(`Copy failed: ${e.message}`);
      }
    },
    /** Draws a bar waveform of `url` into `canvas`. */
    async drawWave(canvas, url) {
      if (!canvas) return;
      const dpr = window.devicePixelRatio || 1;
      const W = canvas.clientWidth || 600, H = canvas.clientHeight || 64;
      canvas.width = W * dpr; canvas.height = H * dpr;
      const ctx = canvas.getContext('2d');
      ctx.scale(dpr, dpr);
      ctx.clearRect(0, 0, W, H);
      if (!url) return;
      try {
        const buf = await (await fetch(url)).arrayBuffer();
        const ac = new (window.AudioContext || window.webkitAudioContext)();
        const audio = await ac.decodeAudioData(buf);
        ac.close();
        const raw = audio.getChannelData(0);
        const bars = Math.max(40, Math.floor(W / 4));
        const step = Math.floor(raw.length / bars) || 1;
        let peak = 0.0001;
        const vals = [];
        for (let i = 0; i < bars; i++) {
          let s = 0;
          for (let j = 0; j < step; j++) s += Math.abs(raw[i * step + j] || 0);
          vals.push(s / step);
          peak = Math.max(peak, s / step);
        }
        ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--accent').trim() || '#8b7cf6';
        vals.forEach((v, i) => {
          const h = Math.max(2, (v / peak) * (H - 10));
          ctx.fillRect(i * (W / bars) + 0.5, (H - h) / 2, Math.max(1, W / bars - 1.5), h);
        });
      } catch (e) {
        console.warn('waveform skipped', e);
      }
    },
    _listeners: [],
    _setEngine(st) {
      VF.engine = st;
      if (st && st.models) VF.models = st.models;
      renderEngineBox();
      window.dispatchEvent(new CustomEvent('vf:engine', { detail: st }));
    },
  };
  window.VF = VF;

  function renderNav() {
    const path = location.pathname.replace(/\/+$/, '') || '/';
    const nav = document.createElement('aside');
    nav.className = 'vf-nav';
    nav.innerHTML = `
      <a class="vf-brand" href="/">
        <div class="vf-logo">VF</div>
        <div class="vf-brand-text"><div class="vf-brand-name">VoiceForge</div><div class="vf-brand-sub">Local voice studio</div></div>
      </a>
      ${NAV.map((g) => `
        ${g.section ? `<div class="vf-section">${g.section}</div>` : ''}
        ${g.items.map((it) => `<a class="vf-link${path === it.href ? ' active' : ''}" href="${it.href}">${icon(it.icon)}<span>${it.label}</span></a>`).join('')}
      `).join('')}
      <div class="vf-spacer"></div>
      <div class="vf-engine" id="vf-engine">
        <div class="vf-engine-head"><span>Engine</span><a href="/engines" class="dim" style="text-decoration:none; text-transform:none; letter-spacing:0;">Manage</a></div>
        <div class="vf-engine-state"><span class="dot" id="vf-engine-dot"></span><span id="vf-engine-text">Connecting…</span></div>
        <select id="vf-engine-select" title="Load a different TTS engine into memory"></select>
      </div>
      <div class="vf-foot">
        <a class="btn-sm ghost" href="/docs" target="_blank" title="Swagger API docs">${icon('docs').replace('<svg', '<svg width="14" height="14"')} API</a>
        <button class="btn-sm ghost" id="vf-copy-guide" title="Copy the system prompt that teaches an LLM to drive this studio">${icon('copy').replace('<svg', '<svg width="14" height="14"')} LLM prompt</button>
      </div>`;
    document.body.prepend(nav);
    document.body.classList.add('vf');
    nav.querySelector('#vf-copy-guide').onclick = VF.copyLLMGuide;
    nav.querySelector('#vf-engine-select').onchange = async (e) => {
      const id = e.target.value;
      if (id === '__unload') {
        await VF.unloadModel().catch((err) => VF.toast(err.message));
        return;
      }
      VF.loadModel(id).catch(() => {});
    };
  }

  function renderEngineBox() {
    const sel = document.getElementById('vf-engine-select');
    if (!sel) return;
    const st = VF.engine;
    const dot = document.getElementById('vf-engine-dot');
    const txt = document.getElementById('vf-engine-text');
    if (!st) {
      dot.className = 'dot bad';
      txt.textContent = 'Studio offline';
      return;
    }
    const loading = st.loading_model_id;
    dot.className = 'dot ' + (loading ? 'busy' : st.active_model_id ? 'ok' : '');
    txt.textContent = loading ? `Loading ${VF.shortName(loading)}…` : st.active_model_id ? `${VF.shortName(st.active_model_id)}` : 'Idle, nothing loaded';
    txt.title = st.status_message || '';
    const current = loading || st.active_model_id || '';
    const opts = VF.models.map((m) => {
      const off = m.available === false;
      return `<option value="${m.id}" ${off ? 'disabled' : ''} ${m.id === current ? 'selected' : ''}>${VF.esc(VF.shortName(m.id))}${off ? ' (not installed)' : ''}</option>`;
    });
    sel.innerHTML = `${current ? '' : '<option value="" selected>Load an engine…</option>'}${opts.join('')}${st.active_model_id ? '<option value="__unload">Unload (free memory)</option>' : ''}`;
    sel.disabled = !!loading;
  }

  function boot() {
    renderNav();
    VF.refreshEngine();
    setInterval(() => { if (!document.hidden) VF.refreshEngine(); }, 5000);
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
})();
