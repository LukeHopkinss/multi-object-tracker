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
from typing import Optional, List

# ======== App & Config ========
app = Flask(__name__)

# Allow configuring output dir in prod (e.g., /tmp/outputs on some hosts)
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "outputs")).resolve()
UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", "uploads")).resolve()
for p in [OUTPUT_DIR, UPLOAD_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Upload caps & types
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024  # 512 MB
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}

# Download safety
ALLOWED_DOWNLOAD_EXTS = [".avi", ".mp4", ".mkv", ".mov"]
SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]{1,120}$")

# Preview tuning
PREVIEW_W, PREVIEW_H = 640, 360
JPEG_QUALITY = 70
TRACK_EVERY_N = 2

# Webcam recording FPS target (wall-clock paced)
WEBCAM_RECORD_FPS = float(os.environ.get("WEBCAM_RECORD_FPS", "20.0"))

# Live webcam frame buffer
frame_buf = deque(maxlen=1)

# Global state (per process/worker)
video_path: Optional[Path] = None
use_webcam: bool = False
selected_rois: List[tuple] = []

# Recording state
recording: bool = False
writer: Optional[cv2.VideoWriter] = None
writer_path: Optional[Path] = None  # OUTPUT_DIR / <uuid>.<ext>
last_file_id: Optional[str] = None  # convenience; not relied on for 'latest' anymore

# Pacing state (webcam only)
_rec_next_due: Optional[float] = None    # perf_counter time when next frame should be written
_rec_fps_target: Optional[float] = None  # fps used to pace & to open writer


# ======== Utilities ========
def _safe_clamp(x, y, w, h, W, H):
    x = max(0, min(int(x), W - 1))
    y = max(0, min(int(y), H - 1))
    w = max(1, min(int(w), W - x))
    h = max(1, min(int(h), H - y))
    return x, y, w, h


def _new_tracker():
    # Try CSRT (legacy), then KCF, then MIL
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
            "Install opencv-contrib-python to get legacy trackers."
        )


def _open_writer_safe(path: Path, fps: float, size_wh: tuple):
    """
    Try several codecs in order of broad Linux compatibility.
    Returns (writer, final_path) or (None, None).
    """
    W, H = size_wh
    trials = [
        ("MJPG", ".avi", cv2.VideoWriter_fourcc(*"MJPG")),
        ("mp4v", ".mp4", cv2.VideoWriter_fourcc(*"mp4v")),
        ("XVID", ".avi", cv2.VideoWriter_fourcc(*"XVID")),
    ]
    for name, ext, fourcc in trials:
        try_path = path.with_suffix(ext)
        wr = cv2.VideoWriter(str(try_path), fourcc, float(fps or 30.0), (W, H))
        if wr is not None and wr.isOpened():
            app.logger.info("VideoWriter opened with %s at %s (fps=%s)", name, try_path, fps)
            return wr, try_path
        if wr is not None:
            wr.release()
    app.logger.error("All VideoWriter codec attempts failed (MJPG/mp4v/XVID).")
    return None, None


def _scan_latest_nonzero() -> Optional[Path]:
    cand = []
    for ext in ALLOWED_DOWNLOAD_EXTS:
        cand.extend(OUTPUT_DIR.glob(f"*{ext}"))
    cand = [p for p in cand if p.is_file() and p.stat().st_size > 0]
    if not cand:
        return None
    cand.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cand[0]


def _resolve_output_file(name: str) -> Optional[Path]:
    """
    Resolve a download name:
      - 'latest'/'last' -> newest non-zero file on disk
      - exact filename (with allowed ext)
      - stem only -> try .avi, .mp4, .mkv, .mov
    """
    if not name or len(name) > 120:
        return None

    if name.lower() in ("latest", "last"):
        p = _scan_latest_nonzero()
        return p

    if not SAFE_NAME.match(name):
        return None

    p = Path(name)
    candidates = []
    if p.suffix.lower() in ALLOWED_DOWNLOAD_EXTS:
        candidates.append(p.name)
    else:
        for ext in ALLOWED_DOWNLOAD_EXTS:
            candidates.append(p.name + ext)

    for cand in candidates:
        path = (OUTPUT_DIR / cand).resolve()
        try:
            if OUTPUT_DIR in path.parents or path.parent == OUTPUT_DIR:
                if path.exists() and path.is_file() and path.stat().st_size > 0:
                    return path
        except Exception:
            pass
    return None


# ======== Routes ========
@app.get("/healthz")
def healthz():
    return {"ok": True, "use_webcam": use_webcam, "has_video": bool(video_path)}


@app.route("/")
def index():
    return render_template("index.html")


@app.post("/upload")
def upload():
    global video_path, use_webcam, selected_rois
    f = request.files.get("file")
    if not f or not f.filename:
        return "no file", 400

    safe_name = secure_filename(f.filename)
    ext = os.path.splitext(safe_name)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return "unsupported file type", 415

    rand_name = f"{uuid4().hex}{ext}"
    path = (UPLOAD_DIR / rand_name).resolve()
    f.save(str(path))
    video_path = path
    use_webcam = False
    selected_rois = []
    app.logger.info("Uploaded video saved to %s", video_path)
    return "", 204


@app.post("/start_webcam")
def start_webcam():
    global use_webcam, selected_rois
    use_webcam = True
    selected_rois = []
    app.logger.info("Webcam mode enabled; push frames to /ingest_frame")
    return "", 204


@app.post("/ingest_frame")
def ingest_frame():
    f = request.files.get("frame")
    if f is None:
        return "", 204
    data = f.read()
    if not data:
        return "", 204
    arr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return "", 204
    frame_buf.append(frame)
    return "", 204


@app.get("/first_frame")
def first_frame():
    if use_webcam:
        frame = None
        for _ in range(50):
            if frame_buf:
                frame = frame_buf[-1]
                break
            time.sleep(0.05)
        if frame is None:
            return "no webcam frame yet", 503
    else:
        if not video_path:
            return "no video uploaded", 400
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return "cannot open video", 500
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            return "failed to capture frame", 500

    h, w = frame.shape[:2]
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        return "encode error", 500
    return {"image": buf.tobytes().hex(), "width": w, "height": h}


@app.post("/select_rois")
def select_rois():
    global selected_rois
    if not request.is_json:
        return "expected JSON", 400
    rois = request.json.get("rois", [])
    try:
        selected_rois = [(int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"])) for r in rois]
    except Exception:
        return "invalid ROI payload", 400
    app.logger.info("ROIs set: %s", selected_rois)
    return "", 204


@app.get("/video_feed")
def video_feed():
    return Response(_generate_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.post("/toggle_recording")
def toggle_recording():
    global recording, writer, writer_path, last_file_id, _rec_next_due, _rec_fps_target
    if recording:
        return jsonify({"ok": True, "message": "already recording",
                       "file_id": writer_path.name if writer_path else None}), 200
    recording = True
    writer = None  # create lazily when we know W/H/fps
    writer_path = (OUTPUT_DIR / f"{uuid4().hex}").resolve()
    last_file_id = None

    # For webcam, pace at WEBCAM_RECORD_FPS; for file we'll use source fps later
    _rec_next_due = None
    _rec_fps_target = None  # will be set when writer is opened
    app.logger.info("Recording requested; base path: %s", writer_path)
    return jsonify({"ok": True, "file_id": writer_path.name}), 200


@app.post("/stop_recording")
def stop_recording():
    global recording, writer, writer_path, last_file_id
    recording = False
    if writer is not None:
        try:
            writer.release()
        except Exception:
            pass
        writer = None

    # Determine final path (could be .avi or .mp4 based on writer open)
    final = None
    if writer_path:
        for ext in ALLOWED_DOWNLOAD_EXTS:
            candidate = writer_path.with_suffix(ext)
            if candidate.exists() and candidate.is_file():
                final = candidate
                break

    size = final.stat().st_size if final else 0
    if final and size > 0:
        last_file_id = final.name
        app.logger.info("Recording done: %s (%d bytes)", final, size)
        return jsonify({"ok": True, "file_id": final.name, "size": size}), 200

    # clean up zero-byte stub(s)
    if writer_path:
        for ext in ALLOWED_DOWNLOAD_EXTS:
            candidate = writer_path.with_suffix(ext)
            if candidate.exists() and candidate.stat().st_size == 0:
                try: candidate.unlink()
                except Exception: pass

    app.logger.warning("Recording stopped but no data was written.")
    return jsonify({"ok": False, "error": "no data written"}), 409


@app.get("/last_recording")
def last_recording():
    p = _scan_latest_nonzero()
    if not p:
        return "no completed recording", 404
    return jsonify({"file_id": p.name})


@app.get("/download")
def download_query():
    name = (request.args.get("filename") or "").strip()
    if not name:
        abort(400, description="Missing filename")
    return _download_impl(name)


@app.get("/download/<path:name>")
def download_path(name):
    return _download_impl(name)


def _download_impl(name: str):
    path = _resolve_output_file(name)
    if not path:
        abort(404, description="File not found")
    ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return send_file(path, mimetype=ctype, as_attachment=True, download_name=path.name)


# ======== Stream loop ========
def _generate_stream():
    global use_webcam, video_path, selected_rois, writer, writer_path, recording
    global _rec_next_due, _rec_fps_target

    # Get first frame source
    if use_webcam:
        frame = None
        for _ in range(50):
            if frame_buf:
                frame = frame_buf[-1].copy()
                break
            time.sleep(0.05)
        if frame is None:
            app.logger.warning("No webcam frames available for streaming")
            return
        fps_src = WEBCAM_RECORD_FPS  # UI pushes ~20fps; we *record* at this exact rate
        cap = None
    else:
        if not video_path:
            app.logger.warning("No video_path set")
            return
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            app.logger.error("Failed to open video: %s", video_path)
            return
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            app.logger.error("Failed to read first frame from video")
            return
        fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0  # use source fps for file mode

    H, W = frame.shape[:2]

    # Init trackers
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

    try:
        while True:
            if use_webcam:
                time.sleep(0.01)
                if not frame_buf:
                    continue
                src = frame_buf[-1]
                frame = src.copy()
            else:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break

            # Lazily open writer after first available frame when recording starts
            if recording and writer is None and writer_path is not None:
                # Choose fps for writer:
                #  - webcam: WEBCAM_RECORD_FPS (paced)
                #  - file:   fps_src (no pacing)
                _rec_fps_target = float(WEBCAM_RECORD_FPS if use_webcam else fps_src)
                (wrt, final_path) = _open_writer_safe(writer_path, _rec_fps_target, (W, H))
                if wrt and wrt.isOpened():
                    writer = wrt
                    writer_path = final_path  # update with real extension
                    # initialize pacing clock for webcam
                    if use_webcam:
                        _rec_next_due = time.perf_counter()  # write immediately
                    else:
                        _rec_next_due = None
                else:
                    # Couldn't open a writer; log & disable recording this session
                    recording = False
                    app.logger.error("Recording disabled: no video codec opened.")

            # Update trackers for preview
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

            annotated = frame
            for b in boxes:
                if b is None:
                    continue
                x, y, w, h = b
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 220, 0), 2)

            # ======= Write full-res if recording =======
            if recording and writer is not None and writer.isOpened():
                if use_webcam:
                    # Pace by wall-clock so playback == real-time
                    now = time.perf_counter()
                    # If we're early for the next tick, skip writing this frame
                    if _rec_next_due is not None and now + 1e-6 < _rec_next_due:
                        pass  # skip
                    else:
                        try:
                            writer.write(annotated)
                        except Exception as e:
                            app.logger.exception("Writer error: %s", e)
                        # schedule next due time
                        tick = 1.0 / max(_rec_fps_target or WEBCAM_RECORD_FPS, 1.0)
                        if _rec_next_due is None:
                            _rec_next_due = now + tick
                        else:
                            _rec_next_due += tick
                            # if we fell behind a lot, resync to now to avoid drift
                            if now - _rec_next_due > 0.5:
                                _rec_next_due = now + tick
                else:
                    # File mode: write every decoded frame (header fps = source fps)
                    try:
                        writer.write(annotated)
                    except Exception as e:
                        app.logger.exception("Writer error: %s", e)

            # Downscale + encode for preview
            preview = cv2.resize(annotated, (PREVIEW_W, PREVIEW_H), interpolation=cv2.INTER_AREA)
            ok_jpg, jpeg = cv2.imencode(".jpg", preview, [int(cv2.IMWRITE_JPEG_QUALITY), int(JPEG_QUALITY)])
            if not ok_jpg:
                frame_idx += 1
                continue

            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
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
    app.logger.info("Starting server on http://0.0.0.0:%d (WEBCAM_RECORD_FPS=%s)", port, WEBCAM_RECORD_FPS)
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
