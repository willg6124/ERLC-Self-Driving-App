const el = (id) => document.getElementById(id);

const statusPill = el('statusPill');
const engagedBadge = el('engagedBadge');

async function poll() {
  try {
    const res = await fetch('/api/telemetry', { cache: 'no-store' });
    const t = await res.json();
    render(t);
    statusPill.textContent = 'LIVE';
    statusPill.className = 'status-pill ok';
  } catch (e) {
    statusPill.textContent = 'RECONNECTING…';
    statusPill.className = 'status-pill warn';
  } finally {
    setTimeout(poll, 150);
  }
}

function fmt(n, d = 1) { return (typeof n === 'number') ? n.toFixed(d) : '—'; }

function render(t) {
  if (!t || Object.keys(t).length === 0) return;

  el('hudSpeed').textContent = fmt(t.speed_mph, 0);
  el('hudTarget').textContent = fmt(t.target_speed_mph, 0);
  el('hudFps').textContent = fmt(t.fps, 0);

  engagedBadge.textContent = t.engaged ? 'ENGAGED' : 'DISENGAGED';
  engagedBadge.className = 'video-badge' + (t.engaged ? '' : ' off');

  const wheel = el('wheel');
  wheel.style.transform = `rotate(${(t.steer || 0) * 180}deg)`;
  el('steerValue').textContent = fmt(t.steer, 2);

  el('throttleBar').style.width = `${(t.throttle || 0) * 100}%`;
  el('throttleValue').textContent = `${Math.round((t.throttle || 0) * 100)}%`;
  el('brakeBar').style.width = `${(t.brake || 0) * 100}%`;
  el('brakeValue').textContent = `${Math.round((t.brake || 0) * 100)}%`;

  el('kvStatus').textContent = (t.status || '—').replace(/_/g, ' ');
  el('kvConfidence').textContent = fmt(t.lane_confidence, 2);
  el('kvOffset').textContent = fmt(t.lane_offset, 2);
  el('kvLight').textContent = t.nearest_light ? `${t.nearest_light.state} (${fmt(t.nearest_light.distance_m, 0)}m)` : 'none in range';
  el('kvVehicles').textContent = t.vehicles_ahead ?? '—';
  el('kvPeds').textContent = t.pedestrians_ahead ?? '—';

  const evs = t.emergency_vehicles || [];
  const activeEv = evs.find(e => e.flashing) || evs[0];
  el('kvEv').textContent = activeEv
    ? `${activeEv.flashing ? 'CONFIRMED' : 'possible'} ${fmt(activeEv.distance_m, 0)}m (conf ${fmt(activeEv.confidence, 2)})`
    : 'none detected';
  el('kvSignal').textContent = t.turn_signal ? t.turn_signal.toUpperCase() : 'off';

  const s = t.stats || {};
  el('statDistance').textContent = s.distance_m != null ? `${fmt(s.distance_m / 1000, 2)} km` : '—';
  el('statEngaged').textContent = s.time_engaged_s != null ? `${fmt(s.time_engaged_s, 0)} s` : '—';
  el('statTopSpeed').textContent = s.max_speed_mph != null ? `${fmt(s.max_speed_mph, 0)} mph` : '—';
  el('statLights').textContent = s.lights_stopped_for ?? '—';
  el('statEmergency').textContent = s.emergency_brakes ?? '—';
  el('statCollisions').textContent = s.collisions ?? '—';

  const list = el('eventList');
  list.innerHTML = '';
  (t.events || []).forEach(ev => {
    const li = document.createElement('li');
    li.className = ev.severity || 'info';
    li.innerHTML = `<span>${ev.message}</span><span class="t">${ev.wall_time}</span>`;
    list.appendChild(li);
  });

  if (!window.__cfgInit && t.config) {
    window.__cfgInit = true;
    initTuning(t.config);
  }
}

function initTuning(cfg) {
  const cruise = el('cruiseSpeed');
  const kp = el('steerKp');
  const headway = el('headway');

  cruise.value = (cfg.cruise_speed_mps * 2.23694).toFixed(0);
  kp.value = cfg.steer_kp;
  headway.value = cfg.follow_time_headway_s;
  el('cruiseVal').textContent = cruise.value;
  el('kpVal').textContent = Number(kp.value).toFixed(2);
  el('headwayVal').textContent = Number(headway.value).toFixed(1);

  let debounce;
  const pushConfig = (patch) => {
    clearTimeout(debounce);
    debounce = setTimeout(() => {
      fetch('/api/config', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(patch),
      });
    }, 150);
  };

  cruise.addEventListener('input', () => {
    el('cruiseVal').textContent = cruise.value;
    pushConfig({ cruise_speed_mps: Number(cruise.value) / 2.23694 });
  });
  kp.addEventListener('input', () => {
    el('kpVal').textContent = Number(kp.value).toFixed(2);
    pushConfig({ steer_kp: Number(kp.value) });
  });
  headway.addEventListener('input', () => {
    el('headwayVal').textContent = Number(headway.value).toFixed(1);
    pushConfig({ follow_time_headway_s: Number(headway.value) });
  });
}

function postAction(action) {
  fetch('/api/control', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action }),
  });
}

el('btnToggle').addEventListener('click', () => postAction('toggle'));
el('btnReset').addEventListener('click', () => postAction('reset'));
el('btnPedestrian').addEventListener('click', () => postAction('pedestrian'));
el('btnCutin').addEventListener('click', () => postAction('cutin'));
el('btnRedLight').addEventListener('click', () => postAction('red_light'));
el('btnEvParked').addEventListener('click', () => postAction('ev_police_parked'));
el('btnEvFire').addEventListener('click', () => postAction('ev_fire_overtaking'));
el('btnEvEms').addEventListener('click', () => postAction('ev_ems_overtaking'));

poll();
