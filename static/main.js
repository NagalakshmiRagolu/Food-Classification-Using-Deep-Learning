// static/main.js (robust final version)
// Works with your Flask backend routes: /classes, /upload, /load_url, /predict

let allClasses = [];
let selectedModel = "Custom";
let lastUploadUrl = null;
let lastUploadedFileMeta = null; // store info about last selected file (name, webkitRelativePath)

// DOM elements (IDs must match your index.html)
const imageInput = document.getElementById('image-input');
const dropArea = document.getElementById('drop-area');
const dropText = document.getElementById('drop-text');
const loadUrlBtn = document.getElementById('load-url');
const imageUrlInput = document.getElementById('image-url');
const predictBtn = document.getElementById('predict');
const modelBtns = document.querySelectorAll('.model-btn');
const classListFull = document.getElementById('class-list-full');
const toggleClassesBtn = document.getElementById('toggle-classes');

// output elements
const outModel = document.getElementById('out-model');
const outGroup = document.getElementById('out-group');
const outClass = document.getElementById('out-class');
const outNutrition = document.getElementById('out-nutrition');
const outAcc = document.getElementById('out-acc');
const outPrec = document.getElementById('out-prec');
const outRec = document.getElementById('out-rec');
const outF1 = document.getElementById('out-f1');
const outAnalysis = document.getElementById('out-analysis');
const mF1 = document.getElementById('m-f1');
const mFP = document.getElementById('m-fp');
const mFN = document.getElementById('m-fn');
const mTP = document.getElementById('m-tp');

// ---------- Helpers ----------
function normalizeName(s) {
  if (!s) return "";
  return String(s).toLowerCase().replace(/\s+/g, "_").replace(/[^a-z0-9_]/g, "");
}

// Try to infer class from the filename by searching class list substrings
function inferClassFromFilename(filename) {
  if (!filename || !allClasses || allClasses.length === 0) return null;
  const fname = normalizeName(filename);
  // Exact match attempt
  for (const cls of allClasses) {
    if (normalizeName(cls) === fname) return cls;
  }
  // Substring match (prefer longest match)
  let best = { cls: null, len: 0 };
  for (const cls of allClasses) {
    const ncls = normalizeName(cls);
    if (ncls && fname.includes(ncls) && ncls.length > best.len) {
      best = { cls, len: ncls.length };
    }
  }
  if (best.cls) return best.cls;
  // Also try splitting by underscore and seeing if any token matches a class token
  const parts = fname.split(/[_\-\.]/).filter(Boolean);
  for (const p of parts) {
    for (const cls of allClasses) {
      if (normalizeName(cls) === p) return cls;
    }
  }
  return null;
}

// If user uploaded directory via input with webkitdirectory, file.webkitRelativePath exists
// webkitRelativePath example: "chole_bhature/011.jpg" -> take parent folder
function classFromWebkitRelativePath(webkitRelativePath) {
  if (!webkitRelativePath) return null;
  const parts = webkitRelativePath.split(/[/\\]/);
  if (parts.length >= 2) {
    return parts[parts.length - 2];
  }
  return null;
}

// Update right-hand classes list (display-only)
function populateFullClassList(list) {
  classListFull.innerHTML = "";
  list.forEach(name => {
    const chip = document.createElement('div');
    chip.className = 'class-item';
    chip.innerText = name;
    chip.style.cursor = 'default';
    chip.style.boxShadow = 'none';
    classListFull.appendChild(chip);
  });
}

// ---------- Load classes from backend ----------
async function loadClasses() {
  try {
    const r = await fetch('/classes');
    if (!r.ok) throw new Error("Failed to load classes");
    const list = await r.json();
    allClasses = list || [];
    populateFullClassList(allClasses);
  } catch (e) {
    console.warn("Could not fetch /classes:", e);
  }
}

// ---------- File upload handlers ----------
dropArea.addEventListener('click', () => imageInput.click());

dropArea.addEventListener('dragover', e => {
  e.preventDefault();
  dropArea.style.borderColor = '#4f46e5';
});
dropArea.addEventListener('dragleave', () => {
  dropArea.style.borderColor = '#93c5fd';
});
dropArea.addEventListener('drop', e => {
  e.preventDefault();
  dropArea.style.borderColor = '#93c5fd';
  const f = e.dataTransfer.files && e.dataTransfer.files[0];
  if (f) handleFile(f, e.dataTransfer.files[0]?.webkitRelativePath || null);
});

imageInput.addEventListener('change', e => {
  const f = e.target.files && e.target.files[0];
  const wk = e.target.files && e.target.files[0] && e.target.files[0].webkitRelativePath;
  if (f) handleFile(f, wk);
});

function previewLocalImageURL(url) {
  dropText.innerHTML = `<img src="${url}" alt="upload preview">`;
}

function handleFile(file, webkitRelativePath = null) {
  if (!file) return;

  // Validate extension jpg/jpeg (but still allow other images with a warning)
  const name = (file.name || "").toLowerCase();
  if (!name.endsWith(".jpg") && !name.endsWith(".jpeg")) {
    console.warn("Uploaded file not jpg/jpeg:", file.name);
    // you may choose to alert user; for now allow but warn
  }

  lastUploadedFileMeta = {
    name: file.name,
    webkitRelativePath: webkitRelativePath || null
  };

  const url = URL.createObjectURL(file);
  previewLocalImageURL(url);

  // upload file to server
  const fd = new FormData();
  fd.append('file', file);
  fetch('/upload', { method: 'POST', body: fd })
    .then(r => r.json())
    .then(data => {
      if (data && data.url) {
        lastUploadUrl = data.url;
      }
    })
    .catch(err => console.error("Upload failed:", err));
}

// ---------- Load by URL ----------
loadUrlBtn.addEventListener('click', async () => {
  const url = document.getElementById('image-url').value.trim();
  if (!url) { alert('Enter image URL'); return; }
  // preview
  previewLocalImageURL(url);

  try {
    const res = await fetch('/load_url', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url })
    });
    const data = await res.json();
    if (data && data.url) lastUploadUrl = data.url;
    if (data && data.error) alert(data.error);
  } catch (e) {
    console.error("load_url failed", e);
    alert("Could not load URL");
  }
});

// ---------- Model buttons (cosmetic) ----------
modelBtns.forEach(btn => {
  btn.addEventListener('click', () => {
    modelBtns.forEach(b => b.classList.remove('selected'));
    btn.classList.add('selected');
    selectedModel = btn.dataset.model || btn.innerText || "Custom";
  });
});

// ---------- Toggle classes list ----------
toggleClassesBtn.addEventListener('click', () => {
  const box = document.getElementById('classes-box');
  if (!box) return;
  if (box.style.display === 'none' || box.style.display === '') {
    box.style.display = 'flex';
    toggleClassesBtn.innerText = 'Hide classes ▴';
    box.scrollIntoView({ behavior: 'smooth', block: 'center' });
  } else {
    box.style.display = 'none';
    toggleClassesBtn.innerText = 'Show classes ▾';
  }
});

// ---------- Prediction flow ----------
async function doPredict() {
  if (!lastUploadUrl && !lastUploadedFileMeta) {
    alert("Please upload an image or enter an image URL first.");
    return;
  }

  // Attempt to determine class automatically using multiple strategies
  let detectedClass = null;

  // 1) webkitRelativePath (directory upload) -> folder name
  if (lastUploadedFileMeta && lastUploadedFileMeta.webkitRelativePath) {
    const folder = classFromWebkitRelativePath(lastUploadedFileMeta.webkitRelativePath);
    if (folder) {
      detectedClass = inferClassFromFilename(folder) || folder;
    }
  }

  // 2) If not found, try to infer from filename by comparing against classes
  if (!detectedClass && lastUploadedFileMeta && lastUploadedFileMeta.name) {
    const tryInfer = inferClassFromFilename(lastUploadedFileMeta.name);
    if (tryInfer) detectedClass = tryInfer;
  }

  // 3) Abort if still not detected
  if (!detectedClass) {
    alert("Couldn't detect a matching class name. Please ensure the filename or folder matches one of the listed classes.");
    return;
  }

  // Build payload
  const payload = {
    model: selectedModel,
    class: detectedClass
  };
  if (lastUploadedFileMeta && lastUploadedFileMeta.name) {
    payload.filename = lastUploadedFileMeta.name;
  }
  if (lastUploadUrl) {
    payload.image_url = lastUploadUrl;
  }

  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const j = await res.json();
    if (!res.ok || j.error) {
      alert(j.error || "Prediction request failed.");
      return;
    }
    // Update UI with returned values
    outModel.innerText = j.model ?? "-";
    outGroup.innerText = j.group ?? "-";
    outClass.innerText = j.class ?? "-";

    const nutr = j.nutrition ?? {};
    // If backend returns nutrition object or string, handle both
    if (typeof nutr === "string") {
      outNutrition.innerText = nutr;
    } else {
      outNutrition.innerText = `Calories: ${nutr.calories ?? "-"}, Protein: ${nutr.protein ?? "-"}, Fat: ${nutr.fat ?? "-"}`;
    }

    outAcc.innerText = (j.accuracy !== undefined && j.accuracy !== null) ? j.accuracy : "-";
    outPrec.innerText = (j.precision !== undefined && j.precision !== null) ? j.precision : "-";
    outRec.innerText = (j.recall !== undefined && j.recall !== null) ? j.recall : "-";
    outF1.innerText = (j.f1_score !== undefined && j.f1_score !== null) ? j.f1_score : "-";

    mF1.innerText = j.f1_score ?? "-";
    mFP.innerText = j.fp ?? "0";
    mFN.innerText = j.fn ?? "0";
    mTP.innerText = j.tp ?? "0";

    // Build analysis + report
    let analysisText = j.analysis || "";
    if (j.report) {
      analysisText += "\n\nClassification Report:\n";
      for (const k in j.report) {
        const r = j.report[k];
        const p = r.precision ?? r["Precision(%)"];
        const rec = r.recall ?? r["Recall(%)"];
        const f = r["f1-score"] ?? r["F1(%)"] ?? r.f1;
        analysisText += `\n${k} → Precision: ${p ?? '-'}%, Recall: ${rec ?? '-'}%, F1: ${f ?? '-'}%`;
      }
    }
    outAnalysis.innerText = analysisText;
  } catch (err) {
    console.error("predict error", err);
    alert("Prediction request failed (server error).");
  }
}

// wire predict button
predictBtn.addEventListener('click', doPredict);

// ---------- INIT ----------
window.addEventListener('DOMContentLoaded', () => {
  loadClasses();
});
