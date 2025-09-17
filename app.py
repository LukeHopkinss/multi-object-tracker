from flask import Flask, render_template, Response, request, send_from_directory
import cv2
import os
from werkzeug.utils import secure_filename

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

@app.route('/first_frame')
def first_frame():
    cap = cv2.VideoCapture(0 if use_webcam else video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return "Failed to capture frame", 500
    height, width = frame.shape[:2]
    _, buffer = cv2.imencode('.jpg', frame)
    return {
        "image": buffer.tobytes().hex(),
        "width": width,
        "height": height
    }

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
    global multi_tracker, selected_rois

    cap = cv2.VideoCapture(0 if use_webcam else video_path)
    if not cap.isOpened():
        return

    success, frame = cap.read()
    if not success:
        return

    raw_output_path = 'static/output_raw.avi'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS) 
    if use_webcam:
        fps = 10;  
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out_writer = cv2.VideoWriter(raw_output_path, fourcc, fps, frame_size)

    templates = []
    tracker_ids = []
    multi_tracker = cv2.legacy.MultiTracker_create()

    for roi in selected_rois:
        x, y, w, h = roi
        templates.append(frame[y:y + h, x:x + w])
        tracker = cv2.legacy.TrackerCSRT_create()
        multi_tracker.add(tracker, frame, roi)
        tracker_ids.append(True)

    frame_count = 0
    recheck_interval = 10
    confidence_threshold = 0.8

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        success, boxes = multi_tracker.update(frame)

        for i, new_box in enumerate(boxes):
            x, y, w, h = map(int, new_box)
            if w <= 0 or h <= 0:
                tracker_ids[i] = False
                continue
            tracker_ids[i] = True
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {i+1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if frame_count % recheck_interval == 0:
            for i, active in enumerate(tracker_ids):
                if active:
                    continue
                template = templates[i]
                h, w = template.shape[:2]
                result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                if max_val > confidence_threshold:
                    x, y = max_loc
                    roi = (x, y, w, h)
                    tracker = cv2.legacy.TrackerCSRT_create()
                    multi_tracker.add(tracker, frame, roi)
                    tracker_ids[i] = True

        out_writer.write(frame)
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()
    out_writer.release()

if __name__ == '__main__':
    app.run(debug=True)
