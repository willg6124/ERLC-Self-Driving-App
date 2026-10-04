const el = (id) => document.getElementById(id);

function fmt(n, d = 0) { return (typeof n === 'number') ? n.toFixed(d) : '—'; }

async function poll() {
  try {
    const res = await fetch('/api/telemetry', { cache: 'no-store' });
    const t = await res.json();
    render(t);
  } catch (e) {
    /* server not up yet / between runner restarts -- just keep retrying */
  } finally {
    setTimeout(poll, 150);
  }
}

function render(t) {
  if (!t || t.status === 'not_started') return;

  el('ovSpeed').textContent = fmt(t.speed_mph, 0);
  el('ovTarget').textContent = fmt(t.target_speed_mph, 0);
  el('ovStatus').textContent = (t.status || '—').replace(/_/g, ' ');
  el('overlayTier').textContent = t.model_tier || '—';

  const engaged = !!t.engaged;
  el('camBadge').textContent = engaged ? 'ENGAGED' : 'DISENGAGED';
  el('camBadge').className = 'cam-badge' + (engaged ? '' : ' off');
  el('engagedDot').style.background = engaged ? 'var(--good)' : 'var(--bad)';
  el('engagedDot').style.boxShadow = engaged ? '0 0 8px var(--good)' : '0 0 8px var(--bad)';
  el('btnOvToggle').textContent = engaged ? '⏸ Disengage' : '▶ Engage';

  const evs = t.emergency_vehicles || [];
  const active = evs.some(e => e.flashing);
  el('evBanner').classList.toggle('hidden', !active);

  el('ovSignal').textContent = t.turn_signal ? `signal: ${t.turn_signal.toUpperCase()}` :
    (t.speed_is_real === false ? 'speed: estimated (no OCR)' : '');
}

function postAction(action) {
  fetch('/api/control', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action }),
  });
}

let hazardsOn = false;
el('btnOvToggle').addEventListener('click', () => postAction('toggle'));
el('btnOvHonk').addEventListener('click', () => postAction('honk'));
el('btnOvHazard').addEventListener('click', () => {
  hazardsOn = !hazardsOn;
  postAction(hazardsOn ? 'hazards_on' : 'hazards_off');
});
el('btnOvSettings').addEventListener('click', () => { window.location.href = '/wizard'; });

poll();
