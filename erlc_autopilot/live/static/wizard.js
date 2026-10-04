/* Same ES5 + XMLHttpRequest approach as overlay.js -- see the comment
 * there for why. This file must keep working in a legacy IE-based
 * renderer (pywebview's fallback on Windows when WebView2 isn't
 * registered for the process), not just modern browsers.
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
          cb(xhr.status, data);
        }
      };
      xhr.onerror = function () { showError('request to ' + url + ' failed (is the server running?)'); cb(0, null); };
      xhr.send(JSON.stringify(body || {}));
    } catch (e) {
      showError('could not send request: ' + e.message);
    }
  }

  function goStep(n) {
    var steps = document.querySelectorAll('.wizard-step');
    for (var i = 0; i < steps.length; i++) {
      var isMatch = Number(steps[i].getAttribute('data-step')) === n;
      if (isMatch) {
        steps[i].classList.remove('hidden');
      } else {
        steps[i].classList.add('hidden');
      }
    }
    var dots = document.querySelectorAll('.step-dot');
    for (var j = 0; j < dots.length; j++) {
      if (Number(dots[j].getAttribute('data-step')) <= n) {
        dots[j].classList.add('active');
      } else {
        dots[j].classList.remove('active');
      }
    }
    if (n === 3) refreshRoiBox();
  }
  window.goStep = goStep;

  function currentRegion() {
    return {
      left: Number(el('regLeft').value), top: Number(el('regTop').value),
      width: Number(el('regWidth').value), height: Number(el('regHeight').value)
    };
  }

  function doPreview(imgId, noteId) {
    post('/api/setup/preview', currentRegion(), function (status, r) {
      if (r && r.ok) {
        el(imgId).src = 'data:image/jpeg;base64,' + r.image_b64;
        if (noteId) el(noteId).textContent = r.note || '';
      } else if (noteId) {
        el(noteId).textContent = (r && r.error) || 'preview failed';
      }
    });
  }

  function refreshRoiBox() {
    var x1 = Number(el('roiX1').value), x2 = Number(el('roiX2').value);
    var y1 = Number(el('roiY1').value), y2 = Number(el('roiY2').value);
    var box = el('roiBox');
    box.style.left = Math.min(x1, x2) + '%';
    box.style.top = Math.min(y1, y2) + '%';
    box.style.width = Math.abs(x2 - x1) + '%';
    box.style.height = Math.abs(y2 - y1) + '%';
  }

  function selectedTier() {
    var radios = document.getElementsByName('tier');
    for (var i = 0; i < radios.length; i++) {
      if (radios[i].checked) return radios[i].value;
    }
    return 'standard';
  }

  function init() {
    el('btnDetect').addEventListener('click', function () {
      el('detectNote').textContent = 'Detecting...';
      post('/api/setup/detect_window', {}, function (status, r) {
        if (r && r.ok && r.region) {
          el('regLeft').value = r.region.left;
          el('regTop').value = r.region.top;
          el('regWidth').value = r.region.width;
          el('regHeight').value = r.region.height;
        }
        el('detectNote').textContent = (r && (r.note || r.error)) || 'detection failed';
      });
    });

    el('btnPreview1').addEventListener('click', function () { doPreview('previewImg1', 'detectNote'); });
    el('btnPreview2').addEventListener('click', function () { doPreview('previewImg2', 'ocrNote'); });

    var roiInputs = ['roiX1', 'roiY1', 'roiX2', 'roiY2'];
    for (var i = 0; i < roiInputs.length; i++) {
      el(roiInputs[i]).addEventListener('input', refreshRoiBox);
    }

    el('btnLaunch').addEventListener('click', function () {
      if (!el('acceptDisclaimer').checked) {
        el('launchNote').textContent = 'Please acknowledge the safety disclaimer first.';
        return;
      }
      el('launchNote').textContent = 'Launching autopilot...';
      var region = currentRegion();
      var x1 = Math.min(Number(el('roiX1').value), Number(el('roiX2').value)) / 100;
      var y1 = Math.min(Number(el('roiY1').value), Number(el('roiY2').value)) / 100;
      var x2 = Math.max(Number(el('roiX1').value), Number(el('roiX2').value)) / 100;
      var y2 = Math.max(Number(el('roiY1').value), Number(el('roiY2').value)) / 100;
      var body = {
        left: region.left, top: region.top, width: region.width, height: region.height,
        speed_roi: [x1, y1, x2, y2],
        model_tier: selectedTier(),
        input_backend: 'auto',
        accepted_disclaimer: true
      };
      post('/api/setup/complete', body, function (status, r) {
        if (r && r.ok) {
          window.location.href = r.redirect || '/overlay';
        } else {
          el('launchNote').textContent = (r && r.error) || 'failed to launch';
        }
      });
    });

    // initial previews so steps 2/3 aren't blank before the user clicks anything
    doPreview('previewImg1', 'detectNote');
    doPreview('previewImg2', 'ocrNote');
    refreshRoiBox();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
