# (Full updated app.py — replace your file with this)
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
    "last_stop": None,     # details of last obstacle stop
    "obstacles": []        # persistent list of detected obstacles (kept until /reset)
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
        #robot.play_audio_tts(text, voice_type=voice_type, wait=wait)
        pass
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

        # Helper inside thread: attempt avoidance and map width
        def attempt_avoidance_and_map(current_x, current_y, current_heading, scale, from_idx, to_idx):
            """
            Try to find a side (left/right) with clearance and insert detour waypoints into points_cm.
            Returns True if avoidance inserted and we should continue, False to abort route and keep last_stop.
            """
            try:
                # Small scan parameters
                scan_angles = [15, 30, 45]
                max_scan = 60
                clearance_values = {"left": 0, "right": 0}

                # Try scanning left and right (rotate a bit, measure distance, rotate back)
                for side in ("left", "right"):
                    best = -1
                    for a in scan_angles:
                        ang = a if side == "left" else -a
                        try:
                            # rotate there
                            safe_turn(2 if ang > 0 else 3, TURN_SPEED_DEG_S, int(abs(ang) * CORRECTION))
                            time.sleep(0.08)
                            d = read_distance()
                            # rotate back
                            safe_turn(3 if ang > 0 else 2, TURN_SPEED_DEG_S, int(abs(ang) * CORRECTION))
                        except Exception:
                            d = read_distance()
                        if d and d > best:
                            best = d
                    clearance_values[side] = best if best > 0 else -1

                # pick side with larger clearance
                left_clear = clearance_values['left']
                right_clear = clearance_values['right']
                chosen_side = None
                if left_clear > right_clear and left_clear > STOP_DISTANCE_CM + 10:
                    chosen_side = 'left'
                elif right_clear > left_clear and right_clear > STOP_DISTANCE_CM + 10:
                    chosen_side = 'right'

                if not chosen_side:
                    # no clear side large enough
                    return False, None

                # compute lateral offset (how far to pass the obstacle)
                lateral_offset = max(40, STOP_DISTANCE_CM + 25)  # cm (conservative)

                # compute unit perpendicular vector
                perp_angle_deg = current_heading + (90 if chosen_side == 'left' else -90)
                perp_rad = math.radians(perp_angle_deg)
                perp_dx = math.cos(perp_rad)
                perp_dy = math.sin(perp_rad)

                # create two detour points in cm (relative to current pose)
                # 1) step out laterally + short forward
                forward_short = 30
                forward_far = 80

                wp1 = {
                    'x': current_x + perp_dx * lateral_offset + math.cos(math.radians(current_heading)) * forward_short,
                    'y': current_y + perp_dy * lateral_offset + math.sin(math.radians(current_heading)) * forward_short
                }

                # 2) a further point ahead that should rejoin past the obstacle
                wp2 = {
                    'x': current_x + perp_dx * lateral_offset + math.cos(math.radians(current_heading)) * forward_far,
                    'y': current_y + perp_dy * lateral_offset + math.sin(math.radians(current_heading)) * forward_far
                }

                # insert these in cm-space before the current to_idx target
                insert_idx = to_idx
                points_cm.insert(insert_idx, wp2)
                points_cm.insert(insert_idx, wp1)

                # estimate obstacle width conservatively: 2 * lateral_offset
                width_cm = 2 * lateral_offset

                obs = {
                    'timestamp': time.time(),
                    'reason': 'mapped_via_avoidance',
                    'center_x_cm': current_x,
                    'center_y_cm': current_y,
                    'x_cm': current_x,
                    'y_cm': current_y,
                    'x_px': current_x / scale,
                    'y_px': current_y / scale,
                    'width_cm': width_cm,
                    'side_chosen': chosen_side,
                    'stop_segment_from_idx': from_idx,
                    'stop_segment_to_idx': to_idx
                }

                with robot_state_lock:
                    robot_state['obstacles'].append(obs)

                return True, obs

            except Exception as e:
                print('avoidance error', e)
                return False, None

        # iterate segments using an index so we can modify points_cm on the fly
        idx = 1
        while idx < len(points_cm):
            with robot_state_lock:
                rx = robot_state['x_cm']
                ry = robot_state['y_cm']
                current_heading = robot_state['heading_deg']

            tx = points_cm[idx]['x']
            ty = points_cm[idx]['y']
            dx = tx - rx
            dy = ty - ry
            segment_dist = math.hypot(dx, dy)
            if segment_dist < 0.5:
                idx += 1
                continue

            # target heading (degrees)
            target_heading = math.degrees(math.atan2(dy, dx))
            delta = normalize_deg(target_heading - current_heading)

            # turn
            if abs(delta) >= 1.0:
                if delta > 0:
                    turn_dir = 2
                else:
                    turn_dir = 3
                degs_motor = max(1, min(360, int(round(abs(delta) * CORRECTION))))
                safe_turn(turn_dir, TURN_SPEED_DEG_S, degs_motor)
                with robot_state_lock:
                    robot_state['heading_deg'] = normalize_deg(current_heading + delta)
                    current_heading = robot_state['heading_deg']
                    robot_state['status_text'] = f"turned {delta:.1f} degrees"

            # move forward in chunks
            remaining = float(segment_dist)
            stopped = False
            while remaining > 0.49:
                chunk = min(MAX_MOVE_CHUNK_CM, remaining)
                chunk_int = int(round(chunk))
                if chunk_int <= 0:
                    break

                dist_read = read_distance()
                if dist_read >= 0 and dist_read <= STOP_DISTANCE_CM:
                    # obstacle detected before moving — attempt avoidance
                    with robot_state_lock:
                        robot_state['status_text'] = f"detected obstacle before move ({dist_read} cm)"

                    succeeded, obs = attempt_avoidance_and_map(rx, ry, current_heading, scale_cm_per_px, max(0, idx - 1), idx)
                    if succeeded:
                        # continue main loop; do NOT increment idx — new waypoints inserted at idx
                        stopped = False
                        break
                    else:
                        # record last_stop and stop execution
                        with robot_state_lock:
                            robot_state['last_stop'] = {
                                'reason': 'obstacle_before_move',
                                'distance_cm': dist_read,
                                'x_cm': robot_state['x_cm'],
                                'y_cm': robot_state['y_cm'],
                                'x_px': robot_state['x_cm'] / scale_cm_per_px,
                                'y_px': robot_state['y_cm'] / scale_cm_per_px,
                                'stop_segment_from_idx': max(0, idx - 1),
                                'stop_segment_to_idx': idx
                            }
                            robot_state['executing'] = False
                        play_tts(f"Ik heb een obstakel {dist_read} centimeter voor mij gedetecteerd! Ik doe niet meer mee.", wait=False)
                        robot.transform_set_chassis_height(2)
                        robot.screen_print_text_newline("Obstakel!", 3)
                        return

                # perform movement
                safe_move(0, FORWARD_SPEED_CM_S, chunk_int)

                # update pose
                rad = math.radians(current_heading)
                dx_ex = chunk_int * math.cos(rad)
                dy_ex = chunk_int * math.sin(rad)
                with robot_state_lock:
                    robot_state['x_cm'] += dx_ex
                    robot_state['y_cm'] += dy_ex
                    robot_state['status_text'] = f"moved {chunk_int} cm"

                # sensor after motion
                dist_read = read_distance()
                if dist_read >= 0 and dist_read <= STOP_DISTANCE_CM:
                    # obstacle during move — try avoidance
                    try:
                        robot.transform_stop()
                    except Exception:
                        pass

                    with robot_state_lock:
                        rx = robot_state['x_cm']
                        ry = robot_state['y_cm']

                    succeeded, obs = attempt_avoidance_and_map(rx, ry, current_heading, scale_cm_per_px, max(0, idx - 1), idx)
                    if succeeded:
                        # continue with newly inserted waypoints
                        stopped = False
                        break
                    else:
                        with robot_state_lock:
                            robot_state['status_text'] = f"stopped (obstacle {dist_read} cm)"
                            robot_state['last_stop'] = {
                                'reason': 'obstacle_during_move',
                                'distance_cm': dist_read,
                                'x_cm': robot_state['x_cm'],
                                'y_cm': robot_state['y_cm'],
                                'x_px': robot_state['x_cm'] / scale_cm_per_px,
                                'y_px': robot_state['y_cm'] / scale_cm_per_px,
                                'stop_segment_from_idx': max(0, idx - 1),
                                'stop_segment_to_idx': idx
                            }
                            robot_state['executing'] = False
                        play_tts(f"Ik heb een obstakel {dist_read} centimeter voor mij gedetecteerd! Ik doe niet meer mee.", wait=False)
                        robot.transform_set_chassis_height(2)
                        robot.screen_print_text_newline("Obstakel!", 3)
                        return

                remaining -= chunk_int
                # small pause
                time.sleep(0.06)

            # proceed to next segment
            idx += 1

        # finished entire route
        with robot_state_lock:
            robot_state['status_text'] = "route complete"
            robot_state['executing'] = False
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


# ---------------- New: reset endpoint ----------------
def _reset_robot_hardware_safely():
    """
    Best-effort attempts to stop/reset robot hardware state without throwing.
    Keep it non-fatal so the endpoint always returns successfully even if hardware isn't available.
    """
    try:
        # stop any active motion
        try:
            robot.transform_stop()
        except Exception:
            pass

        # return arms/chassis to safe defaults where available
        try:
            robot.transform_set_chassis_height(2)
        except Exception:
            pass

        try:
            robot.transform_restory()
        except Exception:
            pass
    except Exception as e:
        # swallow exceptions but log
        print("Error during hardware reset (ignored):", e)


@app.route('/reset', methods=['POST', 'GET'])
def reset():
    """
    Reset in-memory robot state to safe defaults and attempt a best-effort hardware reset.
    Frontend can call this when clearing or reloading to avoid inconsistent memory.
    """
    # Best-effort hardware stop/reset
    _reset_robot_hardware_safely()

    # Reset shared robot_state under lock
    with robot_state_lock:
        robot_state["x_cm"] = 0.0
        robot_state["y_cm"] = 0.0
        robot_state["heading_deg"] = 0.0
        robot_state["executing"] = False
        robot_state["status_text"] = "idle"
        robot_state["last_stop"] = None
        robot_state["obstacles"] = []  # clear persisted obstacle memory

    return jsonify({
        "ok": True,
        "message": "robot memory and status reset to defaults",
        "state": {
            "x_cm": robot_state["x_cm"],
            "y_cm": robot_state["y_cm"],
            "heading_deg": robot_state["heading_deg"],
            "executing": robot_state["executing"],
            "status_text": robot_state["status_text"],
            "last_stop": robot_state["last_stop"],
            "obstacles": robot_state["obstacles"]
        }
    })


@app.route('/status', methods=['GET'])
def status():
    with robot_state_lock:
        # Return obstacles as list (already in px and cm)
        return jsonify({
            "x_cm": robot_state["x_cm"],
            "y_cm": robot_state["y_cm"],
            "heading_deg": robot_state["heading_deg"],
            "executing": robot_state["executing"],
            "status_text": robot_state["status_text"],
            "last_stop": robot_state["last_stop"],
            "obstacles": robot_state["obstacles"]
        })

if __name__ == '__main__':
    # Serve on all interfaces by default for demo laptop; change if needed.
    app.run(host='0.0.0.0', port=5000, debug=True)