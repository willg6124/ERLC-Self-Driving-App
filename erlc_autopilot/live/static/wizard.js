const el = (id) => document.getElementById(id);

function goStep(n) {
  document.querySelectorAll('.wizard-step').forEach(s => {
    s.classList.toggle('hidden', Number(s.dataset.step) !== n);
  });
  document.querySelectorAll('.step-dot').forEach(d => {
    d.classList.toggle('active', Number(d.dataset.step) <= n);
  });
  if (n === 3) refreshRoiBox();
}
window.goStep = goStep;

async function post(url, body) {
  const res = await fetch(url, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {}),
  });
  return res.json();
}

el('btnDetect').addEventListener('click', async () => {
  el('detectNote').textContent = 'Detecting…';
  const r = await post('/api/setup/detect_window');
  if (r.ok && r.region) {
    el('regLeft').value = r.region.left;
    el('regTop').value = r.region.top;
    el('regWidth').value = r.region.width;
    el('regHeight').value = r.region.height;
  }
  el('detectNote').textContent = r.note || r.error || '';
});

function currentRegion() {
  return {
    left: Number(el('regLeft').value), top: Number(el('regTop').value),
    width: Number(el('regWidth').value), height: Number(el('regHeight').value),
  };
}

async function doPreview(imgId, noteId) {
  const r = await post('/api/setup/preview', currentRegion());
  if (r.ok) {
    el(imgId).src = 'data:image/jpeg;base64,' + r.image_b64;
    if (noteId) el(noteId).textContent = r.note || '';
  } else if (noteId) {
    el(noteId).textContent = r.error || 'preview failed';
  }
}

el('btnPreview1').addEventListener('click', () => doPreview('previewImg1', 'detectNote'));
el('btnPreview2').addEventListener('click', () => doPreview('previewImg2', 'ocrNote'));

function refreshRoiBox() {
  const x1 = Number(el('roiX1').value), x2 = Number(el('roiX2').value);
  const y1 = Number(el('roiY1').value), y2 = Number(el('roiY2').value);
  const box = el('roiBox');
  box.style.left = Math.min(x1, x2) + '%';
  box.style.top = Math.min(y1, y2) + '%';
  box.style.width = Math.abs(x2 - x1) + '%';
  box.style.height = Math.abs(y2 - y1) + '%';
}
['roiX1', 'roiY1', 'roiX2', 'roiY2'].forEach(id => el(id).addEventListener('input', refreshRoiBox));

function selectedTier() {
  return document.querySelector('input[name="tier"]:checked').value;
}

el('btnLaunch').addEventListener('click', async () => {
  if (!el('acceptDisclaimer').checked) {
    el('launchNote').textContent = 'Please acknowledge the safety disclaimer first.';
    return;
  }
  el('launchNote').textContent = 'Launching autopilot…';
  const body = {
    ...currentRegion(),
    speed_roi: [
      Math.min(Number(el('roiX1').value), Number(el('roiX2').value)) / 100,
      Math.min(Number(el('roiY1').value), Number(el('roiY2').value)) / 100,
      Math.max(Number(el('roiX1').value), Number(el('roiX2').value)) / 100,
      Math.max(Number(el('roiY1').value), Number(el('roiY2').value)) / 100,
    ],
    model_tier: selectedTier(),
    input_backend: 'auto',
    accepted_disclaimer: true,
  };
  const r = await post('/api/setup/complete', body);
  if (r.ok) {
    window.location.href = r.redirect || '/overlay';
  } else {
    el('launchNote').textContent = r.error || 'failed to launch';
  }
});

// kick off an initial preview so step 2/3 aren't blank before the user clicks anything
doPreview('previewImg1', 'detectNote');
doPreview('previewImg2', 'ocrNote');
refreshRoiBox();
