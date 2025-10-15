from flask import Flask, render_template, Response, request, jsonify, send_file, abort
import cv2
import os
from werkzeug.utils import secure_filename
from collections import deque
import numpy as np
import time
import logging
from uuid import uuid4
from pathlib import Path
import mimetypes
import re

# ===== Live preview buffer (latest frame only) =====
frame_buf = deque(maxlen=1)

app = Flask(__name__)

# Uploads: original source videos (sanitized and randomized)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Outputs: randomized recordings (not publicly served)
OUTPUT_FOLDER = "outputs"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Upload limits
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024  # 512 MB cap
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}

# Download safety
ALLOWED_DOWNLOAD_EXTS = [".avi", ".mp4", ".mkv", ".mov"]
SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]{1,120}$")

# ===== Globals / state =====
video_path = None
use_webcam = False
selected_rois = []

# Recording state
recording = False
writer = None
writer_path = None  # set to outputs/<uuid>.avi when recording starts
last_file_id = None  # remember last finished recording

# Preview tuning
PREVIEW_W, PREVIEW_H = 640, 360
JPEG_QUALITY = 70          # 55–80 is a good live tradeoff
TRACK_EVERY_N = 2          # run tracker every N frames for preview


def _safe_clamp(x, y, w, h, W, H):
    x = max(0, min(int(x), W - 1))
    y = max(0, min(int(y), H - 1))
    w = max(1, min(int(w), W - x))
    h = max(1, min(int(h), H - y))
    return x, y, w, h


def _new_tracker():
    """
    Robust tracker factory:
    Try CSRT (legacy), then KCF, then MIL. Keeps the app running even
    if the OpenCV build does not include legacy trackers.
    """
    try:
        return cv2.legacy.TrackerCSRT_create()
    except Exception:
        pass
    try:
        return cv2.TrackerKCF_create()
    except Exception:
        pass
    try:
        return cv2.TrackerMIL_create()
    except Exception:
        raise RuntimeError(
            "No supported OpenCV tracker found. "
            "Install opencv-contrib-python (not headless) to get legacy trackers."
        )


@app.get("/healthz")
def healthz():
    return {"ok": True, "use_webcam": use_webcam, "has_video": bool(video_path)}


@app.route('/')
def index():
    return render_template('index.html')


@app.post('/upload')
def upload():
    """Upload a source video: sanitize, restrict type, randomize stored name."""
    global video_path, use_webcam, selected_rois
    f = request.files.get('file')
    if not f or not f.filename:
        return "no file", 400

    safe_name = secure_filename(f.filename)
    ext = os.path.splitext(safe_name)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return "unsupported file type", 415

    rand_name = f"{uuid4().hex}{ext}"
    path = os.path.join(UPLOAD_FOLDER, rand_name)
    f.save(path)

    video_path = path
    use_webcam = False
    selected_rois = []
    app.logger.info("Uploaded video saved to %s", video_path)
    return '', 204


@app.post('/start_webcam')
def start_webcam():
    """Enable webcam mode; browser will push frames to /ingest_frame."""
    global use_webcam, selected_rois
    use_webcam = True
    selected_rois = []
    app.logger.info("Webcam mode enabled. Expecting frames via /ingest_frame")
    return '', 204


@app.post('/ingest_frame')
def ingest_frame():
    """Receive a JPEG frame from the browser and stash the latest one."""
    f = request.files.get('frame')
    if f is None:
        return '', 204

    data = f.read()
    if not data:
        return '', 204

    arr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return '', 204

    frame_buf.append(frame)
    return '', 204


@app.get('/first_frame')
def first_frame():
    """Return first frame for ROI UI."""
    if use_webcam:
        frame = None
        for _ in range(50):  # ~2.5s max
            if frame_buf:
                frame = frame_buf[-1]
                break
            time.sleep(0.05)
        if frame is None:
            return "no webcam frame yet", 503
    else:
        if not video_path:
            return "no video uploaded", 400
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return "cannot open video", 500
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            return "failed to capture frame", 500

    h, w = frame.shape[:2]
    ok, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        return "encode error", 500
    return {"image": buffer.tobytes().hex(), "width": w, "height": h}


@app.post('/select_rois')
def select_rois():
    """Receive ROI list from the UI and store for tracking."""
    global selected_rois
    if not request.is_json:
        return "expected JSON", 400
    rois = request.json.get('rois', [])
    try:
        selected_rois = [(int(r['x']), int(r['y']), int(r['w']), int(r['h'])) for r in rois]
    except Exception:
        return "invalid ROI payload", 400
    app.logger.info("ROIs set: %s", selected_rois)
    return '', 204


@app.get('/video_feed')
def video_feed():
    gen = _generate_stream()
    return Response(gen, mimetype='multipart/x-mixed-replace; boundary=frame')


@app.post('/toggle_recording')
def toggle_recording():
    """
    Start recording to a randomized file in OUTPUT_FOLDER.
    Returns file_id the client can later use to download.
    """
    global recording, writer, writer_path, last_file_id
    if recording:
        return jsonify({
            "ok": True,
            "message": "already recording",
            "file_id": os.path.basename(writer_path) if writer_path else None
        }), 200

    recording = True
    writer = None  # created lazily on first frame when we know W,H,fps
    file_id = f"{uuid4().hex}.avi"
    writer_path = os.path.join(OUTPUT_FOLDER, file_id)
    last_file_id = None  # reset; set on stop
    app.logger.info("Recording starting. File: %s", writer_path)
    return jsonify({"ok": True, "file_id": file_id}), 200


@app.post('/stop_recording')
def stop_recording():
    global recording, writer, writer_path, last_file_id
    recording = False
    if writer is not None:
        try:
            writer.release()
        except Exception:
            pass
        writer = None

    # Report the finished file (if any) and guard against zero-byte downloads
    file_id = os.path.basename(writer_path) if writer_path else None
    path = Path(writer_path) if writer_path else None
    size = path.stat().st_size if path and path.exists() else 0
    if size > 0:
        last_file_id = file_id
        app.logger.info("Recording stopped. Saved %s (%d bytes)", file_id, size)
        return jsonify({"ok": True, "file_id": file_id, "size": size}), 200
    else:
        # If nothing was written, clean up stub if present
        if path and path.exists():
            try:
                path.unlink()
            except Exception:
                pass
        app.logger.warning("Recording stopped but no data was written.")
        return jsonify({"ok": False, "error": "no data written"}), 409


@app.get('/last_recording')
def last_recording():
    """Return most recent completed recording id (or 404)."""
    if not last_file_id:
        return "no completed recording", 404
    return jsonify({"file_id": last_file_id})


def _resolve_output_file(name: str) -> Path | None:
    """
    Resolve a requested download name to an actual file in OUTPUT_FOLDER.
    Supports:
    - exact randomized file_id, e.g., 'abc123...def.avi'
    - stem-only (tries common extensions)
    - 'latest'/'last' alias for most recent finished file
    """
    global last_file_id
    base = Path(OUTPUT_FOLDER).resolve()
    if not name or len(name) > 120:
        return None

    # latest alias
    if name.lower() in ("latest", "last") and last_file_id:
        p = (base / last_file_id).resolve()
        if p.exists() and p.is_file():
            return p

    # sanitize input
    if not SAFE_NAME.match(name):
        return None

    # if has ext, try that exact file
    p = Path(name)
    candidates = []
    if p.suffix.lower() in ALLOWED_DOWNLOAD_EXTS:
        candidates.append(p.name)
    else:
        # try common extensions
        for ext in ALLOWED_DOWNLOAD_EXTS:
            candidates.append(p.name + ext)

    for cand in candidates:
        path = (base / cand).resolve()
        try:
            if base in path.parents or path.parent == base:
                if path.exists() and path.is_file() and path.stat().st_size > 0:
                    return path
        except Exception:
            pass
    return None


@app.get('/download')
def download_query():
    """Support /download?filename=foo"""
    name = (request.args.get("filename") or "").strip()
    if not name:
        abort(400, description="Missing filename")
    return _download_impl(name)


@app.get('/download/<path:name>')
def download_path(name):
    """Support /download/foo (stem, exact, or 'latest')"""
    return _download_impl(name)


def _download_impl(name: str):
    path = _resolve_output_file(name)
    if not path:
        abort(404, description="File not found")

    ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    # Use the on-disk randomized name; browsers will save that by default
    return send_file(path, mimetype=ctype, as_attachment=True, download_name=path.name)


def _generate_stream():
    """
    Streams MJPEG preview and writes full-res .avi if recording is enabled.
    Webcam mode pulls freshest frames from frame_buf.
    File mode reads sequentially from uploaded video.
    """
    global use_webcam, video_path, selected_rois, writer, recording

    # Get first frame
    if use_webcam:
        frame = None
        for _ in range(50):  # wait a bit for first pushed frame
            if frame_buf:
                frame = frame_buf[-1].copy()
                break
            time.sleep(0.05)
        if frame is None:
            app.logger.warning("No webcam frames available for streaming")
            return
        fps = 20.0  # browser pushes ~20fps from JS
        cap = None
    else:
        if not video_path:
            app.logger.warning("No video_path set")
            return
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            app.logger.error("Failed to open video: %s", video_path)
            return
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            app.logger.error("Failed to read first frame from video")
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    H, W = frame.shape[:2]

    # Init trackers on first frame
    trackers = []
    tracker_active = []
    if selected_rois:
        for (x, y, w, h) in selected_rois:
            x, y, w, h = _safe_clamp(x, y, w, h, W, H)
            try:
                t = _new_tracker()
                t.init(frame, (x, y, w, h))
                trackers.append(t)
                tracker_active.append(True)
            except Exception as e:
                app.logger.exception("Tracker init failed: %s", e)
                trackers.append(None)
                tracker_active.append(False)

    frame_idx = 0
    last_boxes = [None] * len(trackers)

    # Write full-res recording while preview is downscaled
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    try:
        while True:
            if use_webcam:
                time.sleep(0.01)  # avoid busy-wait
                if not frame_buf:
                    continue
                src = frame_buf[-1]
                frame = src.copy()
            else:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break

            # Lazily init writer after first frame once recording is on
            if recording and writer is None:
                try:
                    fps_val = float(fps) if 'fps' in locals() else 30.0
                    H, W = frame.shape[:2]
                    writer = cv2.VideoWriter(writer_path, fourcc, fps_val, (W, H))
                    app.logger.info("VideoWriter opened: %s", writer_path)
                except Exception as e:
                    app.logger.exception("VideoWriter init failed: %s", e)
                    writer = None

            # Update trackers every N frames for preview
            boxes = last_boxes
            if trackers and (frame_idx % TRACK_EVERY_N == 0):
                boxes = [None] * len(trackers)
                for i, t in enumerate(trackers):
                    if not tracker_active[i] or t is None:
                        continue
                    ok_t, bbox = t.update(frame)
                    if ok_t:
                        x, y, w, h = _safe_clamp(*bbox, W, H)
                        boxes[i] = (x, y, w, h)
                    else:
                        tracker_active[i] = False
                        boxes[i] = None
                last_boxes = boxes

            # Draw minimally to keep preview cheap
            annotated = frame
            for b in boxes:
                if b is None:
                    continue
                x, y, w, h = b
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 220, 0), 2)

            # Write full-res if recording
            if recording and writer is not None:
                try:
                    writer.write(annotated)
                except Exception as e:
                    app.logger.exception("Writer error: %s", e)

            # Downscale for preview and JPEG encode with tuned quality
            preview = cv2.resize(annotated, (PREVIEW_W, PREVIEW_H), interpolation=cv2.INTER_AREA)
            ok_jpg, jpeg = cv2.imencode('.jpg', preview, [int(cv2.IMWRITE_JPEG_QUALITY), int(JPEG_QUALITY)])
            if not ok_jpg:
                frame_idx += 1
                continue

            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            frame_idx += 1
    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            try:
                writer.release()
            except Exception:
                pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    port = int(os.environ.get("PORT", 5000))
    app.logger.info("Starting server on http://127.0.0.1:%d", port)
    app.run(host="127.0.0.1", port=port, debug=True, threaded=True)

