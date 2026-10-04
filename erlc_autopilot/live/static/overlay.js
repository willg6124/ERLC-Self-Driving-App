/* Deliberately written in old-fashioned ES5 (no arrow functions, no
 * template literals, no fetch/async-await, no let/const-only features)
 * and using XMLHttpRequest instead of fetch(). pywebview's native window
 * on Windows can fall back to a legacy Internet-Explorer-based renderer
 * when the Edge WebView2 Runtime isn't registered for the Python
 * process -- that renderer has none of fetch/arrow-functions/async-await,
 * so modern JS here would silently fail to even attach the button click
 * handlers (which is exactly "every button does nothing"). This file
 * works the same in that legacy engine, a normal modern browser tab, and
 * a modern pywebview backend alike.
 */
(function () {
  "use strict";

  function el(id) { return document.getElementById(id); }

  function showError(msg) {
    var box = el('jsError');
    if (!box) return;
    box.style.display = 'block';
    box.textContent = 'UI error: ' + msg;
  }

  window.onerror = function (msg, url, line) {
    showError(msg + ' (line ' + line + ')');
    return false;
  };

  function post(url, body, cb) {
    try {
      var xhr = new XMLHttpRequest();
      xhr.open('POST', url, true);
      xhr.setRequestHeader('Content-Type', 'application/json');
      xhr.onreadystatechange = function () {
        if (xhr.readyState === 4) {
          var data = null;
          try { data = JSON.parse(xhr.responseText); } catch (e) { /* ignore */ }
          if (cb) cb(xhr.status, data);
        }
      };
      xhr.onerror = function () { showError('request to ' + url + ' failed (is the server running?)'); if (cb) cb(0, null); };
      xhr.send(JSON.stringify(body || {}));
    } catch (e) {
      showError('could not send request: ' + e.message);
    }
  }

  function get(url, cb) {
    try {
      var xhr = new XMLHttpRequest();
      xhr.open('GET', url, true);
      xhr.onreadystatechange = function () {
        if (xhr.readyState === 4) {
          var data = null;
          try { data = JSON.parse(xhr.responseText); } catch (e) { /* ignore */ }
          cb(xhr.status, data);
        }
      };
      xhr.onerror = function () { cb(0, null); };
      xhr.send();
    } catch (e) {
      showError('could not poll telemetry: ' + e.message);
    }
  }

  function fmt(n, d) {
    d = d || 0;
    return (typeof n === 'number') ? n.toFixed(d) : '--';
  }

  function poll() {
    get('/api/telemetry', function (status, t) {
      if (t) render(t);
      setTimeout(poll, 150);
    });
  }

  function render(t) {
    if (!t || t.status === 'not_started') {
      el('ovStatus').textContent = 'not started -- open Settings to run setup';
      return;
    }

    el('ovSpeed').textContent = fmt(t.speed_mph, 0);
    el('ovTarget').textContent = fmt(t.target_speed_mph, 0);
    el('ovStatus').textContent = (t.status || '--').replace(/_/g, ' ');
    el('overlayTier').textContent = t.model_tier || '--';

    var engaged = !!t.engaged;
    el('camBadge').textContent = engaged ? 'ENGAGED' : 'DISENGAGED';
    el('camBadge').className = 'cam-badge' + (engaged ? '' : ' off');
    el('engagedDot').style.background = engaged ? '#35d07f' : '#ff4d5e';
    el('engagedDot').style.boxShadow = engaged ? '0 0 8px #35d07f' : '0 0 8px #ff4d5e';
    el('btnOvToggle').innerHTML = engaged ? '&#9208; Disengage' : '&#9654; Engage';

    var evs = t.emergency_vehicles || [];
    var active = false;
    for (var i = 0; i < evs.length; i++) {
      if (evs[i].flashing) { active = true; break; }
    }
    el('evBanner').className = 'overlay-ev-banner' + (active ? '' : ' hidden');

    if (t.turn_signal) {
      el('ovSignal').textContent = 'signal: ' + String(t.turn_signal).toUpperCase();
    } else if (t.speed_is_real === false) {
      el('ovSignal').textContent = 'speed: estimated (no OCR)';
    } else {
      el('ovSignal').textContent = '';
    }
  }

  function postAction(action) {
    post('/api/control', { action: action }, function (status, data) {
      if (status === 400) {
        showError('autopilot is not running yet -- open Settings to finish setup');
      } else if (status === 0) {
        showError('could not reach the autopilot server');
      }
    });
  }

  var hazardsOn = false;

  function init() {
    el('btnOvToggle').addEventListener('click', function () { postAction('toggle'); });
    el('btnOvHonk').addEventListener('click', function () { postAction('honk'); });
    el('btnOvHazard').addEventListener('click', function () {
      hazardsOn = !hazardsOn;
      postAction(hazardsOn ? 'hazards_on' : 'hazards_off');
    });
    el('btnOvSettings').addEventListener('click', function () { window.location.href = '/wizard'; });
    poll();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
