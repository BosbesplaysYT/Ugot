# app.py
from flask import Flask, request, jsonify, render_template, Response
import math
import threading
import time
import numpy as np
import cv2
from ugot import ugot

app = Flask(__name__)

# ---------------------- Configuration ----------------------
DEVICE_IP = "10.60.134.226"   # change as needed
SCALE_CM_PER_PX = 0.5         # centimeters per canvas pixel (adjust -> frontend must match)
FORWARD_SPEED_CM_S = 80       # [5-80] cm/s for transform_move_speed_times
TURN_SPEED_DEG_S = 120        # [5-280] deg/s for transform_turn_speed_times
MAX_MOVE_CHUNK_CM = 30        # chunk moves into <= this many cm per transform_move_speed_times call
SENSOR_ID = 21
STOP_DISTANCE_CM = 28         # if sensor <= this, stop immediately
CORRECTION = 0.96  # tweak experimentally
# -----------------------------------------------------------

# UGOT robot connection and initialization
robot = ugot.UGOT()
try:
    robot.initialize(DEVICE_IP)
    robot.transform_adaption_control(False)
    robot.transform_set_chassis_height(2)
    robot.open_camera() # Open the camera stream
    # optional: set chassis height safe default
    try:
        robot.transform_set_chassis_height(5)
    except Exception:
        pass
except Exception as e:
    # If initialization fails, keep going; connect on-demand in execute
    print("Robot init error (ignored for now):", e)

# Robot runtime state (shared)
robot_state_lock = threading.Lock()
robot_state = {
    "x_cm": 0.0,           # virtual position in cm
    "y_cm": 0.0,
    "heading_deg": 0.0,    # 0 degrees points to +X (canvas right); degrees increase CCW
    "executing": False,
    "status_text": "idle",
    "last_stop": None      # details of last obstacle stop
}

# Helper: normalize degrees to [-180,180)
def normalize_deg(a):
    a = (a + 180) % 360 - 180
    return a

# 2. Generator function to capture frames
def gen_frames():
    while True:
        try:
            frame_data = robot.read_camera_data()
            if frame_data is not None:
                # Convert bytes to numpy array and decode to image
                nparr = np.frombuffer(frame_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # Optional: Resize image to reduce bandwidth/latency
                # img = cv2.resize(img, (480, 320))

                # Encode back to JPEG to send over the network
                ret, buffer = cv2.imencode('.jpg', img)
                frame = buffer.tobytes()

                # Yield the frame in the MJPEG format
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                time.sleep(0.05) # Wait slightly if no frame
        except Exception as e:
            print(f"Camera stream error: {e}")
            break

# 3. Add the Video Feed Route
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Safe wrappers for UGOT commands
def safe_turn(direction, speed, degrees):
    try:
        robot.transform_turn_speed_times(direction, speed, int(degrees), 2)
    except Exception as e:
        # best-effort fallback: use non-times and sleep
        print("turn command error:", e)
        try:
            robot.transform_turn_speed(direction, speed)
            time.sleep(max(0.01, abs(degrees) / max(1, speed)))
            robot.transform_stop()
        except Exception as e2:
            print("turn fallback failed:", e2)

def safe_move(direction, speed, cm):
    """
    Use transform_move_speed_times with unit=1 (centimeters).
    cm must be integer (we cast). This blocks until the motion completes.
    """
    try:
        robot.transform_move_speed_times(direction, speed, int(cm), 1)
    except Exception as e:
        # fallback to timed motion if that fails
        print("move command error:", e)
        try:
            # send continuous move and sleep (approximate)
            robot.transform_move_speed(direction, speed)
            time.sleep(max(0.01, float(cm) / max(1, speed)))
            robot.transform_stop()
        except Exception as e2:
            print("move fallback failed:", e2)

def read_distance():
    try:
        d = robot.read_distance_data(SENSOR_ID)
        if d is None:
            return -1
        return int(d)
    except Exception as e:
        print("distance read error:", e)
        return -1

def play_tts(text, voice_type=0, wait=False):
    try:
        robot.play_audio_tts(text, voice_type=voice_type, wait=wait)
    except Exception as e:
        print("tts error:", e)

# Execution thread: takes waypoints in pixel coordinates {x,y} as list of dicts
def execute_route_thread(waypoints_px, scale_cm_per_px):
    with robot_state_lock:
        if robot_state["executing"]:
            # already executing
            return
        robot_state["executing"] = True
        robot_state["status_text"] = "starting route"

    try:
        # If robot connection is lost or not initialized, attempt re-init with DEVICE_IP
        try:
            robot.initialize(DEVICE_IP)
            robot.transform_adaption_control(False)
        except Exception:
            pass

        play_tts("Ik ga de route uitvoeren!", wait=False)
        robot.transform_set_chassis_height(4)
        robot.screen_print_text_newline("Ik ga de route uitvoeren!", 8)

        # If no waypoints or less than 2, nothing to do
        if not waypoints_px or len(waypoints_px) < 2:
            with robot_state_lock:
                robot_state["status_text"] = "no route"
                robot_state["executing"] = False
            return

        # Convert waypoints to cm coordinates matching virtual frame
        points_cm = []
        for p in waypoints_px:
            # p['x'], p['y'] are pixels on canvas
            # map to cm: x_cm = x_px * scale, y_cm = y_px * scale
            # Note: canvas Y increases downward; we'll keep consistent mapping so frontend and backend match.
            points_cm.append({"x": float(p['x']) * float(scale_cm_per_px),
                              "y": float(p['y']) * float(scale_cm_per_px)})

        # Initialize virtual pose: set start at first waypoint
        with robot_state_lock:
            robot_state["x_cm"] = points_cm[0]['x']
            robot_state["y_cm"] = points_cm[0]['y']
            # Keep current heading; if desired to reset heading to 0: uncomment next line
            # robot_state["heading_deg"] = 0.0
            current_heading = robot_state["heading_deg"]
            robot_state["status_text"] = "executing"

        # For each segment, turn toward target, then move forward in integer-cm chunks
        for idx in range(1, len(points_cm)):
            # compute vector in cm
            with robot_state_lock:
                rx = robot_state["x_cm"]
                ry = robot_state["y_cm"]
                current_heading = robot_state["heading_deg"]

            tx = points_cm[idx]['x']
            ty = points_cm[idx]['y']
            dx = tx - rx
            dy = ty - ry
            segment_dist = math.hypot(dx, dy)
            if segment_dist < 0.5:
                # negligible
                continue

            # target heading (degrees, atan2 uses y then x; note canvas y downwards consistent)
            target_heading = math.degrees(math.atan2(dy, dx))
            # compute minimal delta
            delta = normalize_deg(target_heading - current_heading)

            # turn: choose direction 2=Left (positive delta), 3=Right (negative delta)
            if abs(delta) >= 1.0:
                if delta > 0:
                    turn_dir = 2
                    degs = int(round(delta))
                else:
                    turn_dir = 3
                    degs = int(round(-delta))
                # clamp degs to [1,360]
                # corrected motor command for real robot
                degs_motor = max(1, min(360, int(round(abs(delta) * CORRECTION))))
                safe_turn(turn_dir, TURN_SPEED_DEG_S, degs_motor)
                # update heading
                with robot_state_lock:
                    robot_state["heading_deg"] = normalize_deg(current_heading + delta)
                    current_heading = robot_state["heading_deg"]
                    robot_state["status_text"] = f"turned {delta:.1f} degrees"

            # now move forward along heading by segment_dist, but chunk into integer cm pieces
            remaining = float(segment_dist)
            while remaining > 0.49:
                chunk = min(MAX_MOVE_CHUNK_CM, remaining)
                chunk_int = int(round(chunk))
                if chunk_int <= 0:
                    # safety
                    break
                # Before moving, check distance sensor
                dist_read = read_distance()
                if dist_read >= 0 and dist_read <= STOP_DISTANCE_CM:
                    # obstacle already very close; do not move
                    with robot_state_lock:
                        robot_state["status_text"] = f"stopped (obstacle {dist_read} cm)"
                        robot_state["last_stop"] = {
                            "reason": "obstacle_before_move",
                            "distance_cm": dist_read,
                            "x_cm": robot_state["x_cm"],
                            "y_cm": robot_state["y_cm"]
                        }
                        robot_state["executing"] = False
                    play_tts(f"Ik heb een obstakel {dist_read} centimeter voor mij gedetecteerd! Ik doe niet meer mee.", wait=False)
                    robot.transform_set_chassis_height(2)
                    robot.screen_print_text_newline("Obstakel!", 3)
                    return

                # move chunk_int centimeters forward (direction 0)
                safe_move(0, FORWARD_SPEED_CM_S, chunk_int)

                # update virtual pose based on actual executed chunk
                # heading in degrees: 0 => +X, angle increases CCW (atan2 consistent)
                rad = math.radians(current_heading)
                dx_ex = chunk_int * math.cos(rad)
                dy_ex = chunk_int * math.sin(rad)
                with robot_state_lock:
                    robot_state["x_cm"] += dx_ex
                    robot_state["y_cm"] += dy_ex
                    robot_state["status_text"] = f"moved {chunk_int} cm"
                # after motion, check sensor
                dist_read = read_distance()
                if dist_read >= 0 and dist_read <= STOP_DISTANCE_CM:
                    # obstacle detected, stop route
                    # call transform_stop as a precaution
                    try:
                        robot.transform_stop()
                    except Exception:
                        pass
                    with robot_state_lock:
                        robot_state["status_text"] = f"stopped (obstacle {dist_read} cm)"
                        robot_state["last_stop"] = {
                            "reason": "obstacle_during_move",
                            "distance_cm": dist_read,
                            "x_cm": robot_state["x_cm"],
                            "y_cm": robot_state["y_cm"]
                        }
                        robot_state["executing"] = False
                    play_tts(f"Ik heb een obstakel {dist_read} centimeter voor mij gedetecteerd! Ik doe niet meer mee.", wait=False)
                    robot.transform_set_chassis_height(2)
                    robot.screen_print_text_newline("Obstakel!", 3)
                    return

                remaining -= chunk_int
                # small pause to let sensors settle
                time.sleep(0.06)

            # segment complete, continue to next segment

        # finished entire route
        with robot_state_lock:
            robot_state["status_text"] = "route complete"
            robot_state["executing"] = False
        play_tts("Ik heb de route uitgevoerd", wait=False)
        robot.transform_set_chassis_height(2)
        robot.screen_print_text_newline("Ik heb de route uitgevoerd", 6)

    except Exception as e:
        print("Execution thread error:", e)
        with robot_state_lock:
            robot_state["status_text"] = f"error: {e}"
            robot_state["executing"] = False

# -------------------- Flask endpoints --------------------

@app.route('/')
def index():
    return render_template('index.html')  # will serve frontend (below)

@app.route('/execute_strict', methods=['POST'])
def execute_strict():
    req = request.json or {}
    waypoints = req.get('waypoints')   # expected list of dicts {x,y} in pixels (canvas coords)
    scale = float(req.get('scale_cm_per_px', SCALE_CM_PER_PX))
    if not waypoints or len(waypoints) < 2:
        return jsonify({"ok": False, "message": "Need at least 2 waypoints"}), 400

    # update global scale used for conversion (frontend must use same)
    thread = threading.Thread(target=execute_route_thread, args=(waypoints, scale), daemon=True)
    thread.start()
    return jsonify({"ok": True, "message": "Execution started"})

@app.route('/vibe', methods=["GET"])
def vibe():
    robot.transform_set_chassis_height(2)
    time.sleep(0.2)
    robot.transform_set_chassis_height(7)
    time.sleep(0.2)
    robot.transform_set_chassis_height(2)
    time.sleep(0.2)
    robot.transform_set_chassis_height(7)
    time.sleep(0.2)
    robot.transform_set_chassis_height(2)
    time.sleep(0.2)
    robot.transform_arm_control(1, 90, 150)
    robot.transform_arm_control(3, 90, 150)
    time.sleep(0.2)
    robot.transform_restory()
    robot.transform_arm_control(2, 90, 150)
    robot.transform_arm_control(4, 90, 150)
    time.sleep(0.2)
    robot.transform_restory()
    return jsonify({"ok": True, "message": "Vibing!"})

@app.route('/execute_tail', methods=['POST'])
def execute_tail():
    """
    Execute only a tail of the route, using the robot current pose as the start point.
    Request JSON: { "waypoints": [ {x,y}, ... ], "scale_cm_per_px": <float> }
    - waypoints: tail points in the same pixel coord space you already send (frontend flips Y before sending)
    - scale: same scale used normally (optional, defaults to SCALE_CM_PER_PX)
    """
    req = request.json or {}
    tail_waypoints = req.get('waypoints')   # expected list of dicts {x,y} in pixels (backend coordinate space)
    scale = float(req.get('scale_cm_per_px', SCALE_CM_PER_PX))

    if not tail_waypoints or len(tail_waypoints) < 1:
        return jsonify({"ok": False, "message": "Need at least 1 waypoint for tail execution"}), 400

    # check if robot is busy
    with robot_state_lock:
        if robot_state.get("executing"):
            return jsonify({"ok": False, "message": "Robot is already executing"}), 409
        # build start waypoint from current robot pose (robot_state stores cm)
        start_px = {
            "x": robot_state["x_cm"] / scale,
            "y": robot_state["y_cm"] / scale
        }

    # Compose waypoints: start at the robot's current pose (in backend pixel-space), then tail
    waypoints_to_execute = [start_px] + tail_waypoints

    # Start execution thread (same thread entrypoint used elsewhere)
    thread = threading.Thread(target=execute_route_thread, args=(waypoints_to_execute, scale), daemon=True)
    thread.start()
    return jsonify({"ok": True, "message": "Tail execution started"})


@app.route('/status', methods=['GET'])
def status():
    with robot_state_lock:
        return jsonify({
            "x_cm": robot_state["x_cm"],
            "y_cm": robot_state["y_cm"],
            "heading_deg": robot_state["heading_deg"],
            "executing": robot_state["executing"],
            "status_text": robot_state["status_text"],
            "last_stop": robot_state["last_stop"]
        })

if __name__ == '__main__':
    # Serve on all interfaces by default for demo laptop; change if needed.
    app.run(host='0.0.0.0', port=5000, debug=True)
