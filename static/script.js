const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
let image = new Image();
let rois = [];
let startX, startY, drawing = false;
let originalWidth = 640;
let originalHeight = 480;

function enterApp() {
  document.getElementById("title-screen").classList.add("fade-out");
  setTimeout(() => {
    document.getElementById("title-screen").style.display = "none";
    document.getElementById("app-interface").style.display = "block";
  }, 1000);
}

function loadFirstFrame() {
  fetch("/first_frame")
    .then(res => res.json())
    .then(data => {
      originalWidth = data.width;
      originalHeight = data.height;

      const byteArray = new Uint8Array(data.image.match(/.{1,2}/g).map(b => parseInt(b, 16)));
      const blob = new Blob([byteArray], { type: 'image/jpeg' });
      const url = URL.createObjectURL(blob);

      image.onload = () => {
        canvas.width = originalWidth;
        canvas.height = originalHeight;
        ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
        drawAllROIs();
      };
      image.src = url;
    });
}

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

function uploadVideo() {
  const fileInput = document.getElementById("videoInput");
  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  fetch("/upload", {
    method: "POST",
    body: formData
  }).then(() => {
    rois = [];
    loadFirstFrame();
    document.getElementById("stream").src = "";
  });
}

function startWebcam() {
  fetch("/start_webcam", { method: "POST" })
    .then(() => {
      rois = [];
      loadFirstFrame();
      document.getElementById("stream").src = "";
    });
}

function startTracking() {
  if (rois.length === 0) {
    alert("Please draw at least one bounding box.");
    return;
  }

  const scaleX = originalWidth / canvas.clientWidth;
  const scaleY = originalHeight / canvas.clientHeight;

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
    document.getElementById("stream").src = "/video_feed";
  });
}

function clearROIs() {
  rois = [];
  ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
}

function stopTracking() {
  document.getElementById("stream").src = "";
  ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
  drawAllROIs();
}

function toggleRecording() {
    fetch("/toggle_recording", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ record: true })
    }).then(() => {
      alert("recording started.");
      document.getElementById("startRecBtn").disabled = true;
      document.getElementById("stopRecBtn").disabled = false;
      document.getElementById("downloadBtn").disabled = true;
    });
}


function stopRecording() {
    fetch("/stop_recording", {
      method: "POST"
    }).then(() => {
      alert("recording stopped.");
      document.getElementById("startRecBtn").disabled = false;
      document.getElementById("stopRecBtn").disabled = true;
      document.getElementById("downloadBtn").disabled = false;
    });
}
  
function downloadVideo() {
    const filename = document.getElementById("filenameInput").value.trim();
    if (!filename) {
      alert("please enter a filename.");
      return;
    }
  
    const xhr = new XMLHttpRequest();
    xhr.open("GET", `/download/${encodeURIComponent(filename)}`, true);
    xhr.responseType = "blob";
  
    const progressBar = document.getElementById("downloadProgress");
    progressBar.style.display = "block";
    progressBar.value = 0;
  
    xhr.onprogress = (event) => {
      if (event.lengthComputable) {
        const percent = (event.loaded / event.total) * 100;
        progressBar.value = percent;
      }
    };
  
    xhr.onload = () => {
      if (xhr.status === 200) {
        const blob = new Blob([xhr.response], { type: "video/mp4" });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `${filename}.avi`;
        a.click();
        window.URL.revokeObjectURL(url);
      } else {
        alert("download failed.");
      }
      progressBar.style.display = "none";
    };
  
    xhr.onerror = () => {
      alert("an error occurred during the download.");
      progressBar.style.display = "none";
    };
  
    xhr.send();
}
