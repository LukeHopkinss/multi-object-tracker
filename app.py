from flask import Flask, render_template, Response, request, send_from_directory
import cv2
import os
from werkzeug.utils import secure_filename
from collections import deque
import numpy as np, time
frame_buf = deque(maxlen=1) 



app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

video_path = None
use_webcam = False
selected_rois = []
multi_tracker = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    global video_path, use_webcam, selected_rois
    file = request.files['file']
    if file:
        filename = secure_filename(file.filename)
        video_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(video_path)
        use_webcam = False
        selected_rois = []
    return '', 204

@app.route('/start_webcam', methods=['POST'])
def start_webcam():
    global use_webcam, selected_rois
    use_webcam = True
    selected_rois = []
    return '', 204

@app.post('/ingest_frame')
def ingest_frame():
    f = request.files.get('frame')
    if f is None:
        return '', 204  # no frame; ignore quietly

    data = f.read()
    if not data:
        return '', 204  # empty blob; ignore

    arr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return '', 204  # not decodable yet; ignore

    frame_buf.append(frame)
    return '', 204
    

@app.route('/first_frame')
def first_frame():
    if use_webcam:
        for _ in range(50):  # 50 * 50ms = ~2.5s
            if frame_buf:
                frame = frame_buf[-1]
                break
            time.sleep(0.05)
        else:
            return "no webcam frame yet", 503
    else:
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return "Failed to capture frame", 500

    h, w = frame.shape[:2]
    ok, buffer = cv2.imencode('.jpg', frame)
    if not ok:
        return "encode error", 500
    return {"image": buffer.tobytes().hex(), "width": w, "height": h}

@app.route('/select_rois', methods=['POST'])
def select_rois():
    global selected_rois
    rois = request.json['rois']
    selected_rois = [(r['x'], r['y'], r['w'], r['h']) for r in rois]
    return '', 204

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/download/<filename>')
def download_video(filename):
    path = 'static/output_raw.avi'
    if not os.path.exists(path):
        return "output video not found", 404
    safe_filename = secure_filename(filename) + '.avi'
    return send_from_directory('static', 'output_raw.avi', as_attachment=True, download_name=safe_filename)


def generate_frames():
    """
    Streams MJPEG frames and writes 'static/output_raw.avi'.
    - Webcam mode: reads latest browser-fed frame from frame_buf (no VideoCapture(0)).
    - File mode:     reads frames from `video_path` via cv2.VideoCapture.
    Uses CSRT trackers + simple template-matching re-detection.
    """
    import time
    import numpy as np

    global use_webcam, video_path, selected_rois

    #helpers
    def clamp_box(x, y, w, h, W, H):
        x = max(0, min(int(x), W - 1))
        y = max(0, min(int(y), H - 1))
        w = max(1, min(int(w), W - x))
        h = max(1, min(int(h), H - y))
        return x, y, w, h

    def new_tracker():
        # CSRT lives under legacy in OpenCV 4.x headless wheels
        return cv2.legacy.TrackerCSRT_create()

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    recheck_interval = 30           # try re-detect every N frames
    confidence_threshold = 0.60
    frame_count = 0

    #get first frame
    cap = None
    fps = 10 if use_webcam else 20

    if use_webcam:
        # wait briefly for the browser to push the first frame
        for _ in range(50):  # ~2.5s max
            if frame_buf:
                frame = frame_buf[-1].copy()
                break
            time.sleep(0.05)
        else:
            return  # no webcam frames arrived
    else:
        if not video_path:
            return
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return
        got, frame = cap.read()
        if not got:
            cap.release()
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or 20

    H, W = frame.shape[:2]

    #init writer 
    out_writer = cv2.VideoWriter('static/output_raw.avi', fourcc, float(fps), (W, H))

    #init trackers & templates
    # track each ROI with its own CSRT tracker (easier to handle per-track failures)
    trackers = []
    tracker_active = []
    templates = []

    for (x, y, w, h) in selected_rois:
        x, y, w, h = clamp_box(x, y, w, h, W, H)
        t = new_tracker()
        t.init(frame, (x, y, w, h))
        trackers.append(t)
        tracker_active.append(True)
        templates.append(frame[y:y+h, x:x+w].copy())

    #streaming loop
    while True:
        # fetch next frame
        if use_webcam:
            # throttle to ~10 fps
            time.sleep(0.1)
            if not frame_buf:
                continue
            frame = frame_buf[-1].copy()
        else:
            ok, frame = cap.read()
            if not ok:
                break  # end of file

        frame_count += 1

        # update each tracker
        boxes = [None] * len(trackers)
        for i, t in enumerate(trackers):
            if not tracker_active[i]:
                continue
            ok, bbox = t.update(frame)
            if not ok:
                tracker_active[i] = False
                boxes[i] = None
            else:
                x, y, w, h = clamp_box(*bbox, W, H)
                boxes[i] = (x, y, w, h)

        # periodic re-detection for inactive tracks (template matching)
        if frame_count % recheck_interval == 0:
            for i, active in enumerate(tracker_active):
                if active:
                    continue
                template = templates[i]
                th, tw = template.shape[:2]
                if th == 0 or tw == 0:
                    continue
                result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                if max_val >= confidence_threshold:
                    x, y = max_loc
                    w, h = tw, th
                    x, y, w, h = clamp_box(x, y, w, h, W, H)
                    trackers[i] = new_tracker()
                    trackers[i].init(frame, (x, y, w, h))
                    tracker_active[i] = True
                    boxes[i] = (x, y, w, h)

        # draw boxes & IDs
        for i, bbox in enumerate(boxes):
            if bbox is None:
                continue
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {i+1}", (x, max(18, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # write frame to output and stream as MJPEG
        out_writer.write(frame)
        ok, jpeg = cv2.imencode('.jpg', frame)
        if not ok:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

    #cleanup
    if cap is not None:
        cap.release()
    out_writer.release()


if __name__ == '__main__':
    app.run(debug=True)
