# app.py
# Improved split-in-segment detour insertion with capped in-place turn and better width estimate.

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
SCALE_CM_PER_PX = 0.5         # centimeters per canvas pixel (frontend must match)
FORWARD_SPEED_CM_S = 80       # [5-80] cm/s for transform_move_speed_times
TURN_SPEED_DEG_S = 120        # [5-280] deg/s for transform_turn_speed_times
MAX_MOVE_CHUNK_CM = 30        # chunk moves into <= this many cm per transform_move_speed_times call
SENSOR_ID = 21
STOP_DISTANCE_CM = 40         # if sensor <= this, stop immediately (detection)
AVOID_CLEARANCE_EXTRA_CM = 50  # extra buffer to pass around obstacle
MIN_LATERAL_OFFSET_CM = 40    # minimum lateral offset when performing deterministic detour
MAX_AVOID_ATTEMPTS = 2        # how many times to try avoidance for the same obstacle
# when doing the V split, use a smaller lateral offset so the inserted WP is not huge
SPLIT_LATERAL_OFFSET_CM = 50
SPLIT_FORWARD_OFFSET_CM = 8   # a small forward step from the current pose to place the split wp
MAX_IMMEDIATE_TURN_DEG = 55   # cap in-place rotation when orienting toward inserted WP
# -----------------------------------------------------------

# UGOT robot connection and initialization
robot = ugot.UGOT()
try:
    robot.initialize(DEVICE_IP)
    robot.transform_adaption_control(False)
    robot.transform_set_chassis_height(2)
    robot.open_camera()
    try:
        robot.transform_set_chassis_height(5)
    except Exception:
        pass
except Exception as e:
    print("Robot init error (ignored for now):", e)

# Robot runtime state (shared)
robot_state_lock = threading.Lock()
robot_state = {
    "x_cm": 0.0,
    "y_cm": 0.0,
    "heading_deg": 0.0,    # 0 degrees => +X
    "executing": False,
    "status_text": "idle",
    "last_stop": None,
    # persistent obstacles: each obstacle is a dict with center, width, x_px/y_px, attempts...
    "obstacles": []
}

# normalize degrees to [-180,180)
def normalize_deg(a):
    return (a + 180) % 360 - 180

# camera stream generator
def gen_frames():
    while True:
        try:
            frame_data = robot.read_camera_data()
            if frame_data is not None:
                nparr = np.frombuffer(frame_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                ret, buffer = cv2.imencode('.jpg', img)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                time.sleep(0.05)
        except Exception as e:
            print(f"Camera stream error: {e}")
            break

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Safe wrappers
def safe_turn(direction, speed, degrees):
    """direction: 2=left,3=right ; degrees: positive integer. Best-effort; does not alter robot_state here."""
    try:
        robot.transform_turn_speed_times(direction, speed, int(degrees), 2)
    except Exception as e:
        # fallback (timed)
        try:
            robot.transform_turn_speed(direction, speed)
            time.sleep(max(0.01, float(degrees) / max(1, speed)))
            robot.transform_stop()
        except Exception:
            print("turn fallback failed:", e)

def safe_move(direction, speed, cm):
    """direction 0 = forward. Blocks until done (best-effort)."""
    try:
        robot.transform_move_speed_times(direction, speed, int(cm), 1)
    except Exception as e:
        try:
            robot.transform_move_speed(direction, speed)
            time.sleep(max(0.01, float(cm) / max(1, speed)))
            robot.transform_stop()
        except Exception:
            print("move fallback failed:", e)

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
        # robot.play_audio_tts(text, voice_type=voice_type, wait=wait)
        pass
    except Exception:
        pass

# Utility: find existing obstacle near location (within thresh_cm)
def find_nearby_obstacle(x_cm, y_cm, thresh_cm=60):
    with robot_state_lock:
        for obs in robot_state['obstacles']:
            dx = obs.get('x_cm', obs.get('center_x_cm', 1e9)) - x_cm
            dy = obs.get('y_cm', obs.get('center_y_cm', 1e9)) - y_cm
            if math.hypot(dx, dy) <= thresh_cm:
                return obs
    return None

def execute_route_thread(waypoints_px, scale_cm_per_px):
    with robot_state_lock:
        if robot_state["executing"]:
            return
        robot_state["executing"] = True
        robot_state["status_text"] = "starting route"

    try:
        # --- Robot init ---
        try:
            robot.initialize(DEVICE_IP)
            robot.transform_adaption_control(False)
        except Exception:
            pass

        play_tts("Ik ga de route uitvoeren!", wait=False)
        robot.transform_set_chassis_height(4)

        if not waypoints_px or len(waypoints_px) < 2:
            with robot_state_lock:
                robot_state["executing"] = False
                robot_state["status_text"] = "no route"
            return

        # --- Convert route to cm ---
        points_cm = [
            {
                "x": float(p["x"]) * scale_cm_per_px,
                "y": float(p["y"]) * scale_cm_per_px
            }
            for p in waypoints_px
        ]

        def split_segment_insert(start_x, start_y, target_x, target_y,
                         cur_heading, scale, from_idx, to_idx,
                         distance_cm=None):
            """
            Insert a single lateral detour waypoint between from_idx and to_idx.
            Returns (succeeded: bool, obstacle_dict or None)
            """

            # --- refuse repeated attempts at same place ---
            nearby = find_nearby_obstacle(start_x, start_y, thresh_cm=60)
            if nearby and nearby.get("avoid_attempts", 0) >= MAX_AVOID_ATTEMPTS:
                return False, None

            heading_rad = math.radians(cur_heading)
            fx = math.cos(heading_rad)
            fy = math.sin(heading_rad)

            perp_candidates = [
                ("left", (-fy, fx)),
                ("right", (fy, -fx))
            ]

            for side_name, (px, py) in perp_candidates:
                cand_x = start_x + fx * SPLIT_FORWARD_OFFSET_CM + px * SPLIT_LATERAL_OFFSET_CM
                cand_y = start_y + fy * SPLIT_FORWARD_OFFSET_CM + py * SPLIT_LATERAL_OFFSET_CM

                # skip if too close to known obstacle
                if find_nearby_obstacle(cand_x, cand_y, thresh_cm=30):
                    continue

                # insert waypoint
                points_cm.insert(to_idx, {
                    "x": cand_x,
                    "y": cand_y,
                    "_detour": True,
                    "desired_heading": normalize_deg(
                        math.degrees(math.atan2(target_y - cand_y, target_x - cand_x))
                    )
                })

                # estimate obstacle position
                obs_dist = min(max(distance_cm or 30, 10), STOP_DISTANCE_CM)
                obs_x = start_x + fx * obs_dist
                obs_y = start_y + fy * obs_dist

                obs = {
                    "timestamp": time.time(),
                    "reason": "split_inserted",
                    "x_cm": obs_x,
                    "y_cm": obs_y,
                    "x_px": obs_x / scale,
                    "y_px": obs_y / scale,
                    "width_cm": max(30.0, SPLIT_LATERAL_OFFSET_CM * 2),
                    "side_chosen": side_name,
                    "avoid_attempts": 0,
                    "stop_segment_from_idx": from_idx,
                    "stop_segment_to_idx": to_idx,
                    "detour_wp_x_cm": cand_x,
                    "detour_wp_y_cm": cand_y,
                    "detour_wp_x_px": cand_x / scale,
                    "detour_wp_y_px": cand_y / scale
                }

                with robot_state_lock:
                    robot_state["obstacles"].append(obs)

                return True, obs

            return False, None
        
        def attempt_avoidance_insert(cx, cy, cur_heading, scale, from_idx, to_idx):
            """
            Larger deterministic detour used as fallback.
            """

            nearby = find_nearby_obstacle(cx, cy, thresh_cm=80)
            if nearby and nearby.get("avoid_attempts", 0) >= MAX_AVOID_ATTEMPTS:
                return False, None

            lateral_offset = max(MIN_LATERAL_OFFSET_CM,
                                STOP_DISTANCE_CM + AVOID_CLEARANCE_EXTRA_CM)

            sides = [("left", 1), ("right", -1)]

            for side_name, sign in sides:
                perp_angle = math.radians(cur_heading + 90 * sign)
                px = math.cos(perp_angle)
                py = math.sin(perp_angle)

                wp_x = cx + px * lateral_offset
                wp_y = cy + py * lateral_offset

                if find_nearby_obstacle(wp_x, wp_y, thresh_cm=40):
                    continue

                points_cm.insert(to_idx, {
                    "x": wp_x,
                    "y": wp_y,
                    "_detour": True,
                    "desired_heading": normalize_deg(cur_heading)
                })

                obs = {
                    "timestamp": time.time(),
                    "reason": "fallback_detour",
                    "x_cm": cx,
                    "y_cm": cy,
                    "x_px": cx / scale,
                    "y_px": cy / scale,
                    "width_cm": lateral_offset * 2,
                    "side_chosen": side_name,
                    "avoid_attempts": 0,
                    "stop_segment_from_idx": from_idx,
                    "stop_segment_to_idx": to_idx
                }

                with robot_state_lock:
                    robot_state["obstacles"].append(obs)

                return True, obs

            return False, None



        # --- Initialize pose ---
        with robot_state_lock:
            robot_state["x_cm"] = points_cm[0]["x"]
            robot_state["y_cm"] = points_cm[0]["y"]
            current_heading = robot_state["heading_deg"]
            robot_state["status_text"] = "executing"

        # --- Navigation state ---
        idx = 1
        avoid_mode = False
        avoid_mode_until = 0.0
        current_active_obstacle = None

        # =========================
        # Main navigation loop
        # =========================
        while True:
            if idx >= len(points_cm):
                break

            # expire avoid mode
            if avoid_mode and time.time() > avoid_mode_until:
                avoid_mode = False
                current_active_obstacle = None

            with robot_state_lock:
                rx = robot_state["x_cm"]
                ry = robot_state["y_cm"]
                current_heading = robot_state["heading_deg"]

            tx = points_cm[idx]["x"]
            ty = points_cm[idx]["y"]

            dx = tx - rx
            dy = ty - ry
            dist_to_target = math.hypot(dx, dy)

            # --- Waypoint reached ---
            if dist_to_target < 0.5:
                idx += 1
                continue

            # --- Heading control ---
            target_heading = math.degrees(math.atan2(dy, dx))
            delta = normalize_deg(target_heading - current_heading)

            if abs(delta) > 1.0:
                turn_mag = min(abs(delta), MAX_IMMEDIATE_TURN_DEG)
                turn_dir = 2 if delta > 0 else 3
                safe_turn(turn_dir, TURN_SPEED_DEG_S, int(turn_mag))

                turn_amount = math.copysign(turn_mag, delta)
                with robot_state_lock:
                    robot_state["heading_deg"] = normalize_deg(
                        current_heading + turn_amount
                    )
                    current_heading = robot_state["heading_deg"]

            # --- Obstacle check ---
            dist_read = read_distance()

            active_stop_distance = STOP_DISTANCE_CM
            if avoid_mode:
                active_stop_distance = min(active_stop_distance, 12)

            if dist_read >= 0 and dist_read <= active_stop_distance:
                succeeded, obs = split_segment_insert(
                    rx, ry,
                    tx, ty,
                    current_heading,
                    scale_cm_per_px,
                    idx - 1, idx,
                    dist_read
                )

                if not succeeded:
                    succeeded, obs = attempt_avoidance_insert(
                        rx, ry,
                        current_heading,
                        scale_cm_per_px,
                        idx - 1, idx
                    )

                if succeeded:
                    # enter avoid mode
                    avoid_mode = True
                    avoid_mode_until = time.time() + 6.0
                    current_active_obstacle = obs

                    # immediately reorient toward the newly inserted waypoint
                    new_tx = points_cm[idx]["x"]
                    new_ty = points_cm[idx]["y"]
                    new_heading = math.degrees(
                        math.atan2(new_ty - ry, new_tx - rx)
                    )
                    delta2 = normalize_deg(new_heading - current_heading)

                    if abs(delta2) > 1.0:
                        turn_mag = min(abs(delta2), MAX_IMMEDIATE_TURN_DEG)
                        turn_dir = 2 if delta2 > 0 else 3
                        safe_turn(turn_dir, TURN_SPEED_DEG_S, int(turn_mag))

                        turn_amount = math.copysign(turn_mag, delta2)
                        with robot_state_lock:
                            robot_state["heading_deg"] = normalize_deg(
                                current_heading + turn_amount
                            )

                    with robot_state_lock:
                        obs["avoid_attempts"] = obs.get("avoid_attempts", 0) + 1

                    # HARD restart with new target
                    continue

                # --- Hard block ---
                with robot_state_lock:
                    robot_state["executing"] = False
                    robot_state["status_text"] = "blocked"
                robot.transform_stop()
                play_tts("Ik kan niet verder, het pad is geblokkeerd.", wait=False)
                return

            # --- Forward movement ---
            move_cm = int(min(MAX_MOVE_CHUNK_CM, dist_to_target))
            if move_cm <= 0:
                continue

            safe_move(0, FORWARD_SPEED_CM_S, move_cm)

            rad = math.radians(current_heading)
            with robot_state_lock:
                robot_state["x_cm"] += move_cm * math.cos(rad)
                robot_state["y_cm"] += move_cm * math.sin(rad)
                robot_state["status_text"] = f"moving to wp {idx}"

            time.sleep(0.05)

        # --- Route complete ---
        with robot_state_lock:
            robot_state["executing"] = False
            robot_state["status_text"] = "route complete"

        play_tts("Ik heb de route uitgevoerd.", wait=False)
        robot.transform_set_chassis_height(2)

    except Exception as e:
        print("Execution error:", e)
        with robot_state_lock:
            robot_state["executing"] = False
            robot_state["status_text"] = f"error: {e}"

# -------------------- Flask endpoints --------------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/execute_strict', methods=['POST'])
def execute_strict():
    req = request.json or {}
    waypoints = req.get('waypoints')
    scale = float(req.get('scale_cm_per_px', SCALE_CM_PER_PX))
    if not waypoints or len(waypoints) < 2:
        return jsonify({"ok": False, "message": "Need at least 2 waypoints"}), 400

    thread = threading.Thread(target=execute_route_thread, args=(waypoints, scale), daemon=True)
    thread.start()
    return jsonify({"ok": True, "message": "Execution started"})

@app.route('/execute_tail', methods=['POST'])
def execute_tail():
    req = request.json or {}
    tail_waypoints = req.get('waypoints')
    scale = float(req.get('scale_cm_per_px', SCALE_CM_PER_PX))

    if not tail_waypoints or len(tail_waypoints) < 1:
        return jsonify({"ok": False, "message": "Need at least 1 waypoint for tail execution"}), 400

    with robot_state_lock:
        if robot_state.get("executing"):
            return jsonify({"ok": False, "message": "Robot is already executing"}), 409
        start_px = {"x": robot_state["x_cm"] / scale, "y": robot_state["y_cm"] / scale}

    waypoints_to_execute = [start_px] + tail_waypoints
    thread = threading.Thread(target=execute_route_thread, args=(waypoints_to_execute, scale), daemon=True)
    thread.start()
    return jsonify({"ok": True, "message": "Tail execution started"})

# ---------------- New: reset endpoint ----------------
def _reset_robot_hardware_safely():
    try:
        try:
            robot.transform_stop()
        except Exception:
            pass
        try:
            robot.transform_set_chassis_height(2)
        except Exception:
            pass
        try:
            robot.transform_restory()
        except Exception:
            pass
    except Exception as e:
        print("Error during hardware reset (ignored):", e)

@app.route('/reset', methods=['POST', 'GET'])
def reset():
    _reset_robot_hardware_safely()
    with robot_state_lock:
        robot_state["x_cm"] = 0.0
        robot_state["y_cm"] = 0.0
        robot_state["heading_deg"] = 0.0
        robot_state["executing"] = False
        robot_state["status_text"] = "idle"
        robot_state["last_stop"] = None
        robot_state["obstacles"] = []
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
        # ensure obstacles have x_px/y_px for frontend convenience
        obs_copy = []
        for o in robot_state["obstacles"]:
            copy_o = dict(o)
            # Ensure obstacle center has pixel fields
            if "x_cm" in copy_o and "x_px" not in copy_o:
                copy_o["x_px"] = copy_o["x_cm"] / SCALE_CM_PER_PX
            if "y_cm" in copy_o and "y_px" not in copy_o:
                copy_o["y_px"] = copy_o["y_cm"] / SCALE_CM_PER_PX
            # Ensure detour waypoint pixel fields exist (if present)
            if "detour_wp_x_cm" in copy_o and "detour_wp_x_px" not in copy_o:
                copy_o["detour_wp_x_px"] = copy_o["detour_wp_x_cm"] / SCALE_CM_PER_PX
            if "detour_wp_y_cm" in copy_o and "detour_wp_y_px" not in copy_o:
                copy_o["detour_wp_y_px"] = copy_o["detour_wp_y_cm"] / SCALE_CM_PER_PX

            obs_copy.append(copy_o)


        return jsonify({
            "x_cm": robot_state["x_cm"],
            "y_cm": robot_state["y_cm"],
            "heading_deg": robot_state["heading_deg"],
            "executing": robot_state["executing"],
            "status_text": robot_state["status_text"],
            "last_stop": robot_state["last_stop"],
            "obstacles": obs_copy
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
