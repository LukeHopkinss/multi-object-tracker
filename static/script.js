// ===== Canvas / ROI selection =====
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
let image = new Image();
let rois = [];
let startX, startY, drawing = false;
let originalWidth = 640;
let originalHeight = 480;

// ===== Toasts =====
function toast(msg, type = "info", ms = 2600) {
  let host = document.getElementById("toaster");
  if (!host) {
    host = document.createElement("div");
    host.id = "toaster";
    document.body.appendChild(host);
  }
  const el = document.createElement("div");
  el.className = `toast ${type}`;
  el.textContent = msg;
  host.appendChild(el);
  requestAnimationFrame(() => el.classList.add("show"));
  setTimeout(() => {
    el.classList.remove("show");
    setTimeout(() => el.remove(), 250);
  }, ms);
}

// ---- One-time ROI tip (first upload or webcam start only) ----
function showRoiTipOnce() {
  const KEY = "roiTipShown";
  if (sessionStorage.getItem(KEY)) return;
  toast("Tip: draw a bounding box on the preview canvas to select your object(s).", "info", 4200);
  sessionStorage.setItem(KEY, "1");
}

// Keep console overrides (quiet)
(function () {
  const log = console.log, warn = console.warn, err = console.error;
  console.log = (...a) => { log.apply(console, a); };
  console.warn = (...a) => { warn.apply(console, a); };
  console.error = (...a) => { err.apply(console, a); toast(a.join(" "), "error", 4000); };
})();

/* Put the preview image directly under the canvas in the DOM */
function placePreviewUnderCanvas() {
  const stream = document.getElementById("stream");
  if (!canvas || !stream || !canvas.parentNode) return;
  if (stream.previousElementSibling !== canvas) {
    canvas.parentNode.insertBefore(stream, canvas.nextSibling);
  }
}

/* ============ TITLE CLICK -> show app (no app fade-in) ============ */
function enterApp() {
  const title = document.getElementById("title-screen");
  title.classList.add("fade-out");
  setTimeout(() => {
    title.style.display = "none";
    document.body.classList.add("app-ready");  // this reveals #app-interface
    placePreviewUnderCanvas();
    attachPreviewObservers();
    syncCanvasToPreview();
  }, 600);
}

/* Utility sleep */
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

/* === Match canvas size to the preview (#stream) exactly === */
function syncCanvasToPreview() {
  const stream = document.getElementById("stream");
  let targetW, targetH;

  if (stream && stream.clientWidth > 0 && stream.clientHeight > 0) {
    targetW = Math.round(stream.clientWidth);
    targetH = Math.round(stream.clientHeight);
  } else {
    const container = canvas.parentElement;
    const maxW = (container && container.clientWidth) ? container.clientWidth : originalWidth;
    const ratio = originalWidth / originalHeight;
    targetW = Math.min(Math.round(maxW), originalWidth);
    targetH = Math.round(targetW / ratio);
  }

  // Bitmap size
  canvas.width = targetW;
  canvas.height = targetH;

  // CSS size so mouse coords == canvas coords
  canvas.style.width = `${targetW}px`;
  canvas.style.height = `${targetH}px`;

  // Redraw
  if (image && image.complete && image.naturalWidth) {
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
  }
  drawAllROIs();
}

/* Observe preview size changes */
let streamResizeObserver = null;
function attachPreviewObservers() {
  const stream = document.getElementById("stream");
  if (!stream) return;

  if (window.ResizeObserver && !streamResizeObserver) {
    streamResizeObserver = new ResizeObserver(() => {
      if (document.body.classList.contains("app-ready")) {
        placePreviewUnderCanvas();
        syncCanvasToPreview();
      }
    });
    streamResizeObserver.observe(stream);
  }

  stream.addEventListener("load", () => {
    if (document.body.classList.contains("app-ready")) {
      requestAnimationFrame(() => {
        placePreviewUnderCanvas();
        syncCanvasToPreview();
      });
    }
  });

  window.addEventListener("resize", () => {
    if (document.body.classList.contains("app-ready")) {
      placePreviewUnderCanvas();
      syncCanvasToPreview();
    }
  });
}

/* Fetch and show first frame so users can draw ROIs before tracking */
async function loadFirstFrame(maxRetries = 40, delayMs = 150) {
  for (let i = 0; i < maxRetries; i++) {
    try {
      const res = await fetch("/first_frame");
      if (!res.ok) { await sleep(delayMs); continue; }
      const data = await res.json();

      originalWidth = data.width;
      originalHeight = data.height;

      const byteArray = new Uint8Array(data.image.match(/.{1,2}/g).map(b => parseInt(b, 16)));
      const blob = new Blob([byteArray], { type: "image/jpeg" });
      const url = URL.createObjectURL(blob);

      image.onload = () => {
        placePreviewUnderCanvas();
        syncCanvasToPreview();
        ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
        drawAllROIs();
      };
      image.src = url;
      return;
    } catch (_) {
      await sleep(delayMs);
    }
  }
  toast("No webcam frame received yet. Try again.", "warn", 3500);
}

// ===== ROI drawing =====
canvas.addEventListener("mousedown", e => {
  startX = e.offsetX;
  startY = e.offsetY;
  drawing = true;
});
canvas.addEventListener("mousemove", e => {
  if (!drawing) return;
  const width = e.offsetX - startX;
  const height = e.offsetY - startY;
  ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
  drawAllROIs();
  ctx.strokeStyle = "blue";
  ctx.strokeRect(startX, startY, width, height);
});
canvas.addEventListener("mouseup", e => {
  drawing = false;
  rois.push({
    x: startX,
    y: startY,
    w: e.offsetX - startX,
    h: e.offsetY - startY
  });
  ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
  drawAllROIs();
});

function drawAllROIs() {
  for (const box of rois) {
    ctx.strokeStyle = "red";
    ctx.lineWidth = 2;
    ctx.strokeRect(box.x, box.y, box.w, box.h);
  }
}

// ===== Upload & Webcam =====
function uploadVideo() {
  const fileInput = document.getElementById("videoInput");
  if (!fileInput.files.length) {
    toast("Please choose a video file.", "warn");
    return;
  }
  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  fetch("/upload", { method: "POST", body: formData }).then(() => {
    rois = [];
    loadFirstFrame();
    document.getElementById("stream").src = "";
    toast("Video uploaded.", "info");
    showRoiTipOnce();
  });
}

let webcamStream, pushTimer;

async function startWebcam() {
  await fetch("/start_webcam", { method: "POST" });

  const media = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
  webcamStream = media;

  const videoEl = document.createElement("video");
  videoEl.srcObject = media;
  await videoEl.play();

  const buf = document.createElement("canvas");
  const W = 640, H = 480;
  buf.width = W; buf.height = H;
  const bctx = buf.getContext("2d");

  clearInterval(pushTimer);
  pushTimer = setInterval(async () => {
    bctx.drawImage(videoEl, 0, 0, W, H);
    const blob = await new Promise(r => buf.toBlob(r, "image/jpeg", 0.7));
    const fd = new FormData();
    fd.append("frame", blob, "frame.jpg");
    try { await fetch("/ingest_frame", { method: "POST", body: fd }); } catch {}
  }, 50);

  rois = [];
  setTimeout(() => loadFirstFrame(), 300);
  document.getElementById("stream").src = "";
  toast("Webcam started.", "info");
  showRoiTipOnce();
}

// ===== Tracking =====
function startTracking() {
  if (rois.length === 0) {
    toast("Please draw at least one bounding box.", "warn");
    return;
  }

  const scaleX = originalWidth / canvas.width;
  const scaleY = originalHeight / canvas.height;

  const scaledROIs = rois.map(r => ({
    x: Math.round(r.x * scaleX),
    y: Math.round(r.y * scaleY),
    w: Math.round(r.w * scaleX),
    h: Math.round(r.h * scaleY)
  }));

  fetch("/select_rois", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ rois: scaledROIs })
  }).then(() => {
    const stream = document.getElementById("stream");
    attachPreviewObservers();
    stream.src = "/video_feed";

    // When the preview renders, put it directly under the canvas & sync sizes
    const ensureSync = () => {
      placePreviewUnderCanvas();
      requestAnimationFrame(syncCanvasToPreview);
    };
    stream.addEventListener("load", ensureSync, { once: true });
    setTimeout(ensureSync, 150); // safety

    toast("Tracking started.", "info");
  });
}

function clearROIs() {
  rois = [];
  ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
  toast("Boxes cleared.", "info");
}

function stopTracking() {
  const stream = document.getElementById("stream");
  stream.src = "";
  placePreviewUnderCanvas();
  syncCanvasToPreview();
  ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
  drawAllROIs();
  toast("Tracking stopped.", "info");
}

// ===== Recording & Download =====
function toggleRecording() {
  fetch("/toggle_recording", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ record: true })
  }).then(r => r.json()).then(() => {
    toast("Recording started.", "info");
    document.getElementById("startRecBtn").disabled = true;
    document.getElementById("stopRecBtn").disabled = false;
    document.getElementById("downloadBtn").disabled = true;
  }).catch(() => toast("Could not start recording.", "error"));
}

function stopRecording() {
  fetch("/stop_recording", { method: "POST" }).then(() => {
    toast("Recording stopped.", "info");
    document.getElementById("startRecBtn").disabled = false;
    document.getElementById("stopRecBtn").disabled = true;
    document.getElementById("downloadBtn").disabled = false;
  }).catch(() => toast("Could not stop recording.", "error"));
}

function downloadVideo() {
  const filename = document.getElementById("filenameInput").value.trim();
  if (!filename) { toast("Please enter a filename.", "warn"); return; }

  const xhr = new XMLHttpRequest();
  xhr.open("GET", `/download/${encodeURIComponent(filename)}`, true);
  xhr.responseType = "blob";

  const progressBar = document.getElementById("downloadProgress");
  if (progressBar) { progressBar.style.display = "block"; progressBar.value = 0; }

  xhr.onprogress = (e) => {
    if (progressBar && e.lengthComputable) {
      progressBar.value = (e.loaded / e.total) * 100;
    }
  };

  xhr.onload = () => {
    if (xhr.status === 200) {
      const blob = new Blob([xhr.response], { type: "video/x-msvideo" });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${filename}.avi`;
      a.click();
      window.URL.revokeObjectURL(url);
      toast("Download ready.", "info");
    } else {
      toast("Download failed.", "error");
    }
    if (progressBar) progressBar.style.display = "none";
  };

  xhr.onerror = () => {
    toast("An error occurred during the download.", "error");
    if (progressBar) progressBar.style.display = "none";
  };

  xhr.send();
}
