from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import tempfile
import uuid
import shutil
import subprocess
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort 
import cv2
import numpy as np
import traceback
import datetime
import imageio
import cv2
import math
line1 = []
line2 = []






app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize models and parameters for detection
model_yolov8n = YOLO("yolo11n.pt")  # Updated to YOLO11n model
model_custom = YOLO("yolo_trained_model.pt")  # Updated path to your custom trained model
tracker = DeepSort(max_age=30)

gate_zone = None  # Will be set by user input via API
prohibited_area = None  # Will be set by user input via API
footpath_polygons = []  # List of footpath polygons set by user input via API
line1 = None  # For vehicle count DOWN
line2 = None  # For vehicle count UP
original_width = 0
original_height = 0

vehicle_positions = {}
wrong_way_ids = set()
yolov8n_classes = [0, 2, 3, 5, 7]  # person, car, motorcycle, bus, truck
custom_classes = [0, 1, 4, 6]  # classes for yolo_trained_model.pt

def is_inside_gate(xc, yc, gate):
    x1, y1, x2, y2 = gate
    return x1 < xc < x2 and y1 < yc < y2

def is_inside_prohibited_area(xc, yc, area):
    x1, y1, x2, y2 = area
    return x1 < xc < x2 and y1 < yc < y2

emergency_vehicle_classes = [0, 1, 6]  # classes from custom model allowed to go wrong way
yolo11n_person_class = 0  # person class in yolo11n


def create_kalman_filter(initial_point):
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
    kf.errorCovPost = np.eye(4, dtype=np.float32)
    kf.statePost = np.array([[initial_point[0]],
                             [initial_point[1]],
                             [0],
                             [0]], dtype=np.float32)
    return kf

def interpolate_points(p1, p2, gap):
    return [(p1[0] + (p2[0] - p1[0]) * i / gap,
             p1[1] + (p2[1] - p1[1]) * i / gap) for i in range(1, gap)]

def get_color(tid):
    if tid not in colors:
        colors[tid] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    return colors[tid]


def detect_wrong_way(tracked_objects):
    for obj in tracked_objects:
        x1, y1, x2, y2, track_id, cls = obj
        xc = int((x1 + x2) / 2)
        yc = int((y1 + y2) / 2)

        print(f"Checking ID={track_id}, Class={cls}, Center=({xc},{yc})")

        # Skip person and emergency vehicle classes
        if cls == yolo11n_person_class or cls in emergency_vehicle_classes:
            print(f"Skipping ID={track_id} (class={cls}) as allowed class.")
            vehicle_positions[track_id] = yc
            continue

        if track_id in vehicle_positions:
            prev_y = vehicle_positions[track_id]
            print(f"Prev Y={prev_y}, Curr Y={yc}, Inside gate={is_inside_gate(xc, yc, gate_zone)}")

            if yc > prev_y and is_inside_gate(xc, yc, gate_zone):
                print(f"Wrong-way detected for ID={track_id}")
                wrong_way_ids.add(track_id)

        vehicle_positions[track_id] = yc

ALLOWED_PROHIBITED_CLASSES = [0, 2, 3, 5, 7]  # person, car, motorcycle, bus, truck

def detect_prohibited_area_intrusion(tracked_objects):
    intrusions = set()

    if prohibited_area is None:
        print("Prohibited area not set.")
        return intrusions

    x1_area, y1_area, x2_area, y2_area = prohibited_area
    print(f"Prohibited Area: ({x1_area}, {y1_area}) to ({x2_area}, {y2_area})")

    for obj in tracked_objects:
        x1, y1, x2, y2, track_id, cls = obj

        if cls not in ALLOWED_PROHIBITED_CLASSES:
            continue

        center_x = (x1 + x2) // 2
        feet_y = y2  # bottom of box

        print(f"Object ID={track_id}, Class={cls}, Center=({center_x}, {feet_y})")

        if x1_area <= center_x <= x2_area and y1_area <= feet_y <= y2_area:
            print(f"Intrusion Detected for ID={track_id}")
            intrusions.add(track_id)

    return intrusions



def detect_footpath_intrusion(tracked_objects, frame):
    intrusions = set()
    if not footpath_polygons:
        return intrusions
    for obj in tracked_objects:
        x1, y1, x2, y2, track_id, cls = obj
        center_x, person_feet_y = (x1 + x2) // 2, y2  # Bottom-center of person
        for polygon in footpath_polygons:
            # polygon is list of (x, y) tuples
            pts = np.array(polygon, np.int32)
            if cv2.pointPolygonTest(pts, (center_x, person_feet_y), False) >= 0:
                # Person is inside footpath polygon
                intrusions.add(track_id)
                break
    return intrusions

def draw_gate_zone(frame, gate):
    x1, y1, x2, y2 = map(int, gate)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
    cv2.putText(frame, "EXIT ONLY", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

def draw_prohibited_area(frame, area):
    print("Drawing prohibited area on frame")
    x1, y1, x2, y2 = map(int, area)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(frame, "PROHIBITED AREA", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

def draw_footpath(frame, polygons):
    print("Drawing footpath polygons on frame")  # Debug log
    for polygon in polygons:
        pts = np.array(polygon, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        # Optionally add label
        x, y = pts[0][0]
        cv2.putText(frame, "FOOTPATH", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

FFMPEG_PATH = 'ffmpeg'  # Default to 'ffmpeg', can be overridden with full path

def convert_to_mp4(input_path, output_path):
    # Convert any video format to mp4 using ffmpeg
    command = [
        FFMPEG_PATH,
        '-i', input_path,
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '22',
        '-c:a', 'aac',
        '-strict', 'experimental',
        '-y',  # overwrite output file if exists
        output_path
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion failed: {result.stderr.decode()}")

@app.route('/')
def hello_world():
    return 'Hello from the Python server!'


@app.route('/set_prohibited_area', methods=['POST'])
def set_prohibited_area():
    global prohibited_area
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    coords = data.get('prohibited_area')
    if not coords or len(coords) != 4:
        return jsonify({"error": "Invalid prohibited_area coordinates"}), 400
    try:
        prohibited_area = tuple(map(int, coords))
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"message": "Prohibited area updated", "prohibited_area": prohibited_area})

@app.route('/set_footpath', methods=['POST'])
def set_footpath():
    global footpath_polygons
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    polygons = data.get('footpath_polygons')
    if not polygons or not isinstance(polygons, list):
        return jsonify({"error": "Invalid footpath polygons"}), 400
    try:
        # Expect polygons as list of list of [x, y] points
        footpath_polygons = []
        for poly in polygons:
            if not isinstance(poly, list) or not all(isinstance(pt, list) and len(pt) == 2 for pt in poly):
                return jsonify({"error": "Invalid polygon format"}), 400
            footpath_polygons.append([(int(pt[0]), int(pt[1])) for pt in poly])
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"message": "Footpath polygons updated", "footpath_polygons": footpath_polygons})




# --- 4. VEHICLE COUNT FUNCTION (put here) ---

#line_up = [(700, 0), (700, 720)]
#line_down = [(1150, 0), (1150, 720)]

def is_crossing_vertical(prev, curr, line_x):
    return (prev[0] < line_x and curr[0] >= line_x) or (prev[0] > line_x and curr[0] <= line_x)

def run_vehicle_counter(input_video_path, output_path):
    global line1, line2
    print("DEBUG: line1 =", line1)
    print("DEBUG: line2 =", line2)
    cap = cv2.VideoCapture(input_video_path)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    line1_px = line1 if line1 else None
    line2_px = line2 if line2 else None

    print("line1_px:", line1_px)
    print("line2_px:", line2_px)

    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    count_up = 0
    count_down = 0
    tracked_ids = {}
    counted_ids = set()
    vehicle_paths = {}
    frame_num = 0


    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if line1:
            cv2.line(frame, tuple(line1[0]), tuple(line1[1]), (0, 255, 0), 2)
        if line2:
            cv2.line(frame, tuple(line2[0]), tuple(line2[1]), (255, 0, 0), 2)

        frame_num += 1  # ← Increment frame counter each loop


        combined_results = []
        for model in [model_yolov8n, model_custom]:
            results = model(frame, verbose=False)[0]
            for det in results.boxes:
                cls_id = int(det.cls)
                conf = float(det.conf[0])
                x1, y1, x2, y2 = map(int, det.xyxy[0])
                w, h = x2 - x1, y2 - y1
                if conf > 0.5 and cls_id in [2, 3, 5, 7] and w > 30 and h > 30:
                    combined_results.append(([x1, y1, w, h], conf, "vehicle"))

        tracks = tracker.update_tracks(combined_results, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            center = (cx, cy)

            if track_id not in tracked_ids:
                tracked_ids[track_id] = []
            tracked_ids[track_id].append(center)
            # Speed calculation
            if track_id not in vehicle_paths:
                vehicle_paths[track_id] = []
            vehicle_paths[track_id].append((frame_num, cx, cy))

            if len(vehicle_paths[track_id]) >= 2:
                (f1, x1_, y1_), (f2, x2_, y2_) = vehicle_paths[track_id][0], vehicle_paths[track_id][-1]
                pixel_dist = math.sqrt((x2_ - x1_)**2 + (y2_ - y1_)**2)
                meters = pixel_dist * 0.05  # your scale
                time = (f2 - f1) / fps
                if time > 0:
                    speed = (meters / time) * 3.6  # km/h
                    label = f"{int(speed)} km/h"
                    color = (0, 0, 255) if speed > 60 else (255, 0, 0)  # red for overspeed

                    y_pos = max(30, y1 - 10)
                    cv2.putText(frame, label, (x1, y_pos),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                    if speed > 60:
                        alert_y = max(20, y_pos - 25)
                        cv2.putText(frame, "ALERT: Overspeed!", (x1, alert_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


            if len(tracked_ids[track_id]) >= 5 and track_id not in counted_ids:
                prev = tracked_ids[track_id][-2]
                curr = tracked_ids[track_id][-1]

                if line2_px and is_crossing_vertical(prev, curr, line2_px[0][0]):
                    count_down += 1
                    counted_ids.add(track_id)
                elif line1_px and is_crossing_vertical(prev, curr, line1_px[0][0]):
                    count_up += 1
                    counted_ids.add(track_id)


        # Draw lines
        # Draw user-defined lines instead of hardcoded
        #if line1_px:
        #    cv2.line(frame, line1_px[0], line1_px[1], (255, 0, 0), 2)  # Blue for UP
        #if line2_px:
        #    cv2.line(frame, line2_px[0], line2_px[1], (0, 255, 0), 2)  # Green for DOWN
        # Draw count overlays on top left of the frame
        cv2.putText(frame, f"UP: {count_up}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        cv2.putText(frame, f"DOWN: {count_down}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)


       
        out.write(frame)

    cap.release()
    out.release()

    return {
        "up": count_up,
        "down": count_down,
        "total": count_up + count_down,
        "output_path": output_path
    }






@app.route('/detect', methods=['POST'])
def detect():
    global gate_zone, prohibited_area
    print("/detect request received")
    data = request.form

    detection_type_raw = data.get('detection_type', 'wrong_way')
    detection_type = detection_type_raw.split(',') if isinstance(detection_type_raw, str) else ['wrong_way']

    if 'wrong_way' in detection_type and gate_zone is None:
        return jsonify({"error": "Gate zone not set."}), 400
    if 'prohibited_area' in detection_type and prohibited_area is None:
        return jsonify({"error": "Prohibited area not set."}), 400
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    # Save uploaded video to temp file
    original_ext = os.path.splitext(video_file.filename)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=original_ext) as temp_video:
        video_file.save(temp_video.name)
        temp_video_path = temp_video.name

    # Convert to mp4 if needed
    converted_temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    try:
        convert_to_mp4(temp_video_path, converted_temp_path)
        print("Video converted:", converted_temp_path)
    except Exception as e:
        print("⚠️ ffmpeg failed:", e)
        traceback.print_exc()
        converted_temp_path = temp_video_path  # fallback to original

    # Prepare output path
    output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"processed_{uuid.uuid4().hex}.mp4"
    temp_output_video_path = os.path.join(output_dir, output_filename)

    try:
        # Run detection and counting
        anomalies, vehicle_count_result = run_on_video(
            converted_temp_path,
            temp_output_video_path,
            detection_type,
            gate_zone=gate_zone,
            prohibited_area=prohibited_area,
            footpath_polygons=footpath_polygons,
            line1=line1,
            line2=line2
        )        
        print("Detection done")

        vehicle_count_result = None
        #if 'vehicle_count' in detection_type:
        #    vehicle_count_result = run_vehicle_counter(converted_temp_path, temp_output_video_path)
        print("Vehicle counting done:", vehicle_count_result)

    except Exception as e:
        print("Error in processing:", str(e))
        traceback.print_exc()
        os.remove(temp_video_path)
        if os.path.exists(converted_temp_path): os.remove(converted_temp_path)
        if os.path.exists(temp_output_video_path): os.remove(temp_output_video_path)
        return jsonify({"error": str(e)}), 500

    # Cleanup input files
    os.remove(temp_video_path)
    if converted_temp_path != temp_video_path:
        os.remove(converted_temp_path)

    # Return results
    response_data = {
        "anomalies": anomalies,
        **({
            "vehicle_count": {
                "up": vehicle_count_result.get("up", 0),
                "down": vehicle_count_result.get("down", 0),
                "total": vehicle_count_result.get("total", 0)
            }
        } if vehicle_count_result else {}),
        "video_url": f"/processed_video/{output_filename}",
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "output_file": output_filename
    }

    print("Response ready")
    return jsonify(response_data)

# Class name mappings for yolo11n and custom model
yolo11n_class_names = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorbike",
    4: "aeroplane",
    5: "bus",
    6: "train",
    7: "truck",
    # Add other classes as needed
}

custom_class_names = {
    0: "police car",
    1: "firetruck",
    2: "helmet",
    3: "traffic cones",
    4: "rickshaw",
    5: "accident",
    6: "ambulance"
}

def get_class_name(cls):
    if cls in yolo11n_class_names:
        return yolo11n_class_names[cls]
    elif cls in custom_class_names:
        return custom_class_names[cls]
    else:
        return f"class_{cls}"
    


def run_on_video(video_path, output_path, detection_type, gate_zone=None, prohibited_area=None, footpath_polygons=None, line1=None, line2=None):
    global vehicle_positions, wrong_way_ids
    vehicle_positions = {}
    wrong_way_ids = set()

    print("Reloading YOLO models to reset tracker state...")
    from ultralytics import YOLO
    model_yolov8n_local = YOLO("yolo11n.pt")
    model_custom_local = YOLO("yolo_trained_model.pt")
    vehicle_count_result = None 

    import imageio.v3 as iio
    import datetime
    import os
    import traceback

    print("Using imageio to read frames...")
    try:
        reader = iio.imiter(video_path, plugin="pyav")
        meta = iio.immeta(video_path, plugin="pyav")
        def compute_iou(boxA, boxB):
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])
            interArea = max(0, xB - xA) * max(0, yB - yA)
            boxAArea = max(1.0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
            boxBArea = max(1.0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
            return interArea / (boxAArea + boxBArea - interArea + 1e-5)

    except Exception as e:
        print(f"Failed to open video with imageio: {e}")
        traceback.print_exc()
        return []

    fps = meta.get('fps', 25)
    first_frame = next(reader, None)
    if first_frame is None:
        print("Failed to read any frame from video.")
        return []
    height, width = first_frame.shape[0], first_frame.shape[1]
    if width % 2: width -= 1
    if height % 2: height -= 1
    frame_size = (width, height)

    reader.close()
    reader = iio.imiter(video_path, plugin="pyav")

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    anomalies = []
    frame_count = 0
    vehicle_paths = {}
    scale = 0.05  # meters per pixel — adjust based on your camera setup
    speed_threshold = 60  # km/h — speed limit threshold

    print("Starting frame-by-frame processing...")

    for frame_rgb in reader:
        frame_count += 1
        # After resizing the frame
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        frame_bgr = cv2.resize(frame_bgr, frame_size, interpolation=cv2.INTER_AREA)
        # --- Draw scaled lines based on original video resolution ---
        global original_width, original_height  # values set via /set_lines
        h, w = frame_bgr.shape[:2]

        if original_width > 0 and original_height > 0:
            scale_x = w / original_width
            scale_y = h / original_height

            def scale_line(line):
                return [
                    (int(x * scale_x), int(y * scale_y))
                    for (x, y) in line
                ] if line else None

            scaled_line1 = scale_line(line1)
            scaled_line2 = scale_line(line2)

            if scaled_line1:
                cv2.line(frame_bgr, scaled_line1[0], scaled_line1[1], (0, 255, 0), 2)
            if scaled_line2:
                cv2.line(frame_bgr, scaled_line2[0], scaled_line2[1], (255, 0, 0), 2)


        results_list = []
        # Always use both models
        overlap = set(yolov8n_classes).intersection(custom_classes)
        yolov8n_filtered = [c for c in yolov8n_classes if c not in overlap or c == 0]
        results_list.append(model_yolov8n_local.track(frame_bgr, persist=True, classes=yolov8n_filtered, verbose=False))
        results_list.append(model_custom_local.track(frame_bgr, persist=True, classes=custom_classes, verbose=False))

        combined = []
        for res in results_list:
            if res and res[0].boxes is not None and res[0].boxes.id is not None:
                boxes = res[0].boxes.xyxy.cpu().numpy()
                ids = res[0].boxes.id.cpu().numpy()
                cls_list = res[0].boxes.cls.cpu().numpy()
                for box, tid, cls in zip(boxes, ids, cls_list):
                    x1, y1, x2, y2 = map(int, box)
                    print(f"Detected box - ID: {tid}, Class: {cls}")
                    combined.append((x1, y1, x2, y2, int(tid), int(cls)))

        # Speed detection for vehicles
        for track in combined:
            x1, y1, x2, y2, track_id, cls = track
            if cls not in [2, 3, 5, 7]:  # car, motorbike, bus, truck
                continue

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            if track_id not in vehicle_paths:
                vehicle_paths[track_id] = []
            vehicle_paths[track_id].append((frame_count, center_x, center_y))

            path = vehicle_paths[track_id]
            if len(path) >= 2:
                (f1, x1_, y1_), (f2, x2_, y2_) = path[0], path[-1]
                pixel_dist = math.sqrt((x2_ - x1_)**2 + (y2_ - y1_)**2)
                meters = pixel_dist * scale
                time = (f2 - f1) / fps
                if time > 0:
                    speed = (meters / time) * 3.6  # km/h
                    color = (0, 0, 255) if speed > speed_threshold else (255, 0, 0)
                    label = f"{int(speed)} km/h"
                    y_pos = max(30, y1 - 10)

                    # Always show speed (red if overspeeding, blue otherwise)
                    cv2.putText(frame_bgr, label, (x1, y_pos),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                    if speed > speed_threshold:
                        # Show overspeed alert (no box)
                        cv2.putText(frame_bgr, "ALERT: Overspeed!", (x1, y1 - 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        anomalies.append({
                            "id": int(track_id),
                            "class": int(cls),
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "description": f"Overspeed: {int(speed)} km/h"
                        })


        # --- Helmet Detection (always active) ---
        persons = [box for box in combined if get_class_name(box[5]).lower() == "person"]
        motorcycles = [box for box in combined if get_class_name(box[5]).lower() == "motorbike"]

        helmet_boxes = []
        if model_custom_local:
            helmet_result = model_custom_local(frame_bgr)[0]
            helmet_boxes = [box for box, conf in zip(helmet_result.boxes.xyxy.cpu().numpy(),
                                                    helmet_result.boxes.conf.cpu().numpy()) if conf > 0.5]

        for mbox in motorcycles:
            mx1, my1, mx2, my2, _, _ = mbox
            riders = []
            for pbox in persons:
                px1, py1, px2, py2, _, _ = pbox
                if compute_iou((px1, py1, px2, py2), (mx1, my1, mx2, my2)) > 0.3 or (
                    abs(py1 - my1) < 50 and abs(px1 - mx1) < 50):
                    riders.append((px1, py1, px2, py2))

            if riders:
                all_boxes = [(mx1, my1, mx2, my2)] + riders
                x1 = int(min(b[0] for b in all_boxes))
                y1 = int(min(b[1] for b in all_boxes))
                x2 = int(max(b[2] for b in all_boxes))
                y2 = int(max(b[3] for b in all_boxes))

                has_helmet = any(
                    compute_iou(r, h) > 0.3 for r in riders for h in helmet_boxes
                )

                color = (0, 255, 0) if has_helmet else (0, 0, 255)
                label = "Rider with Helmet" if has_helmet else "ALERT: No Helmet"
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame_bgr, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                anomalies.append({
                    "bbox": [x1, y1, x2, y2],
                    "description": label
                })

        # --- Detection Logic ---
        if 'wrong_way' in detection_type:
            detect_wrong_way(combined)
            for x1, y1, x2, y2, tid, cls in combined:
                if tid in wrong_way_ids:
                    anomalies.append({
                        "id": tid,
                        "class": cls,
                        "bbox": [x1, y1, x2, y2],
                        "description": "Wrong Way"
                    })
                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame_bgr, "WRONG WAY", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if 'prohibited_area' in detection_type and prohibited_area is not None:
            if all(0 < val <= 1 for val in prohibited_area):
                x_center, y_center, box_w, box_h = [val * dim for val, dim in zip(prohibited_area, [width, height, width, height])]
                x1_pa = int(x_center - box_w / 2)
                y1_pa = int(y_center - box_h / 2)
                x2_pa = int(x_center + box_w / 2)
                y2_pa = int(y_center + box_h / 2)
            else:
                x1_pa, y1_pa, x2_pa, y2_pa = prohibited_area

            for x1, y1, x2, y2, track_id, cls in combined:
                if cls in [0, 2, 3, 5, 7]:
                    center_x = (x1 + x2) // 2
                    feet_y = y2
                    if x1_pa <= center_x <= x2_pa and y1_pa <= feet_y <= y2_pa:
                        label = "Person" if cls == 0 else "Vehicle"
                        description = f"{label} in Prohibited Area"
                        anomalies.append({
                            "id": track_id,
                            "class": int(cls),
                            "bbox": [x1, y1, x2, y2],
                            "description": description
                        })
                        cv2.putText(frame_bgr, f"ALERT: {description}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    color = (0, 0, 255) if x1_pa <= center_x <= x2_pa and y1_pa <= feet_y <= y2_pa else (0, 255, 0)
                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
                    cv2.circle(frame_bgr, (center_x, feet_y), 5, color, -1)

        # --- Footpath Alert Detection ---
        if 'footpath' in detection_type and footpath_polygons:
            for x1, y1, x2, y2, tid, cls in combined:
                label = get_class_name(cls).lower()
                if "person" not in label:
                    continue

                # Skip if overlaps with any motorbike (rider case)
                is_rider = False
                for vx1, vy1, vx2, vy2, _, vcls in combined:
                    if get_class_name(vcls).lower() == "motorbike":
                        iou = compute_iou((x1, y1, x2, y2), (vx1, vy1, vx2, vy2))
                        if iou > 0.25:
                            is_rider = True
                            break
                if is_rider:
                    continue  # don't trigger road alert for riders

                # Proceed with normal footpath check
                cx, feet_y = (x1 + x2) // 2, y2
                on_footpath = any(
                    cv2.pointPolygonTest(np.array(poly, np.int32), (cx, feet_y), False) >= 0
                    for poly in footpath_polygons
                )

                desc = "Person on Footpath" if on_footpath else "Person on Road"
                color = (0, 255, 0) if on_footpath else (0, 0, 255)

                if not on_footpath:
                    cv2.putText(frame_bgr, "ALERT: Person on Road!", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                anomalies.append({
                    "id": tid,
                    "class": int(cls),
                    "bbox": [x1, y1, x2, y2],
                    "description": desc
                })
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame_bgr, (cx, feet_y), 5, color, -1)
                cv2.putText(frame_bgr, desc, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Draw configured zones
        if 'wrong_way' in detection_type and gate_zone is not None:
            draw_gate_zone(frame_bgr, gate_zone)
        if 'prohibited_area' in detection_type and prohibited_area is not None:
            draw_prohibited_area(frame_bgr, prohibited_area)
        if 'footpath' in detection_type and footpath_polygons:
            draw_footpath(frame_bgr, footpath_polygons)

        print("line1 (used for drawing):", line1)
        print("line2 (used for drawing):", line2)
        print("Frame size:", frame_bgr.shape)
        if line1:
            cv2.line(frame_bgr, tuple(line1[0]), tuple(line1[1]), (0, 255, 0), 2)
        if line2:
            cv2.line(frame_bgr, tuple(line2[0]), tuple(line2[1]), (255, 0, 0), 2)

        '''if 'vehicle_count' in detection_type:
            cv2.putText(frame_bgr, f"UP: {count_up}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame_bgr, f"DOWN: {count_down}", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame_bgr, f"TOTAL: {count_up + count_down}", (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        '''
        # --- Ambulance Detection (always active) ---
        ambulance_classes = [0, 1, 6]  # police car, firetruck, ambulance from custom model
        ambulance_detections = model_custom_local(frame_bgr)[0]
        for det in ambulance_detections.boxes:
            cls_id = int(det.cls)
            conf = float(det.conf[0])
            if cls_id in ambulance_classes and conf > 0.5:
                x1, y1, x2, y2 = map(int, det.xyxy[0])
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(frame_bgr, "Ambulance Detected", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                anomalies.append({
                    "bbox": [x1, y1, x2, y2],
                    "class": cls_id,
                    "description": "Ambulance Detected"
                })

        out.write(frame_bgr)

    out.release()
    reader.close()
    print(f"Total frames processed: {frame_count}")
    return anomalies, vehicle_count_result



from flask import send_from_directory, request, Response, abort

@app.route('/processed_video/<filename>')
def processed_video(filename):
    output_dir = os.path.join(os.getcwd(), "output")
    file_path = os.path.join(output_dir, filename)
    if not os.path.exists(file_path):
        abort(404)

    range_header = request.headers.get('Range', None)
    if not range_header:
        return send_from_directory(output_dir, filename)

    size = os.path.getsize(file_path)
    byte1, byte2 = 0, None

    m = None
    import re
    m = re.search(r'bytes=(\d+)-(\d*)', range_header)
    if m:
        g = m.groups()
        byte1 = int(g[0])
        if g[1]:
            byte2 = int(g[1])

    length = size - byte1
    if byte2 is not None:
        length = byte2 - byte1 + 1

    with open(file_path, 'rb') as f:
        f.seek(byte1)
        data = f.read(length)

    # Determine mimetype based on file extension for processed video streaming
    ext = os.path.splitext(filename)[1].lower()
    if ext == '.mp4' or ext == '.m4v':
        mimetype = 'video/mp4'
    elif ext == '.avi':
        mimetype = 'video/x-msvideo'
    elif ext == '.mov':
        mimetype = 'video/quicktime'
    elif ext == '.mkv':
        mimetype = 'video/x-matroska'
    elif ext == '.flv':
        mimetype = 'video/x-flv'
    elif ext == '.wmv':
        mimetype = 'video/x-ms-wmv'
    else:
        mimetype = 'application/octet-stream'

    rv = Response(data, 206, mimetype=mimetype, content_type=mimetype, direct_passthrough=True)
    rv.headers.add('Content-Range', f'bytes {byte1}-{byte1 + length - 1}/{size}')
    rv.headers.add('Accept-Ranges', 'bytes')
    rv.headers.add('Content-Length', str(length))
    return rv

@app.route('/list_output_files', methods=['GET'])
def list_output_files():
    output_dir = os.path.join(os.getcwd(), "output")
    try:
        files = [f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))]
        return jsonify({"files": files})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/set_gate_zone', methods=['POST'])
def set_gate_zone():
    global gate_zone
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    coords = data.get('gate_zone')
    if not coords or len(coords) != 4:
        return jsonify({"error": "Invalid gate_zone coordinates"}), 400
    try:
        gate_zone = tuple(map(int, coords))
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"message": "Gate zone updated", "gate_zone": gate_zone})

'''@app.route('/set_lines', methods=['POST'])
def set_lines():
    global line1, line2
    try:
        data = request.get_json()

        raw_line1 = data.get("line1", [])
        raw_line2 = data.get("line2", [])

        line1 = [(int(pt[0]), int(pt[1])) for pt in raw_line1]
        line2 = [(int(pt[0]), int(pt[1])) for pt in raw_line2]

        print("Line1:", line1)
        print("Line2:", line2)

        return jsonify({"message": "Lines set successfully", "line1": line1, "line2": line2})

    except Exception as e:
        return jsonify({"error": str(e)}), 400'''

@app.route('/set_lines', methods=['POST'])
def set_lines():
    global line1, line2, original_width, original_height
    try:
        data = request.get_json()

        raw_line1 = data.get("line1", [])
        raw_line2 = data.get("line2", [])
        original_width = int(data.get("video_width", 1))
        original_height = int(data.get("video_height", 1))

        line1 = [(int(pt[0]), int(pt[1])) for pt in raw_line1]
        line2 = [(int(pt[0]), int(pt[1])) for pt in raw_line2]

        print("Line1:", line1)
        print("Line2:", line2)
        print("Original resolution received:", original_width, original_height)

        return jsonify({"message": "Lines and resolution set successfully"})

    except Exception as e:
        return jsonify({"error": str(e)}), 400





if __name__ == '__main__':
    app.run(port=8000, debug=True)

