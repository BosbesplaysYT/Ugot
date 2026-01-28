# app.py
# Refactored, safer version of the route execution service.
# - Encapsulated robot interactions in RobotController
# - Robust camera stream that won't crash if robot isn't ready
# - Thread + stop-event management for execution
# - Better logging and error handling
# - Minor behavioral fixes and defensive checks

from flask import Flask, request, jsonify, render_template, Response
import math
import threading
import time
import logging
from typing import List, Dict, Optional, Tuple
import numpy as np
import cv2

# Put the real ugot import behind a try so the module can still run for testing.
try:
    from ugot import ugot
except Exception:
    ugot = None  # unit tests / dev machines may not have the robot package

# ---------------------- Configuration ----------------------
DEVICE_IP = "10.60.134.226"
SCALE_CM_PER_PX = 0.5
FORWARD_SPEED_CM_S = 80
TURN_SPEED_DEG_S = 120
MAX_MOVE_CHUNK_CM = 30
SENSOR_ID = 21
STOP_DISTANCE_CM = 40
AVOID_CLEARANCE_EXTRA_CM = 30
MIN_LATERAL_OFFSET_CM = 40
MAX_AVOID_ATTEMPTS = 2
SPLIT_LATERAL_OFFSET_CM = 50
SPLIT_FORWARD_OFFSET_CM = 8
MAX_IMMEDIATE_TURN_DEG = 55
# -----------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("route_exec")

app = Flask(__name__)

# ---------------- Robot controller (safe wrapper) ----------------
class RobotController:
    """
    Encapsulates robot calls and provides safe fallbacks.
    If ugot is None or initialization fails, methods degrade gracefully.
    """
    def __init__(self):
        self._robot = None
        self._initialized = False
        self._camera_open = False

    def ensure_robot(self, device_ip: str = DEVICE_IP):
        """Lazy initialize the robot instance. Safe to call repeatedly."""
        if self._initialized:
            return True

        if ugot is None:
            logger.warning("ugot library not available; robot disabled.")
            return False

        try:
            self._robot = ugot.UGOT()
            self._robot.initialize(device_ip)
            # try several safe configuration calls; ignore failures
            try:
                self._robot.transform_adaption_control(False)
            except Exception:
                pass
            try:
                self._robot.transform_set_chassis_height(2)
            except Exception:
                pass
            try:
                self._robot.open_camera()
                self._camera_open = True
            except Exception:
                self._camera_open = False
            # mark as initialized even if some optional calls failed
            self._initialized = True
            logger.info("Robot initialized (best-effort).")
            return True
        except Exception as e:
            logger.exception("Robot initialization failed: %s", e)
            self._robot = None
            self._initialized = False
            return False

    def safe_turn(self, direction: int, speed: float, degrees: float):
        """Best-effort turn. direction: 2=left,3=right"""
        if not self.ensure_robot():
            return
        try:
            # prefer timed precise API if available
            try:
                self._robot.transform_turn_speed_times(direction, speed, int(degrees), 2)
                return
            except Exception:
                pass

            # fallback: start continuous turn, sleep proportional to degrees/speed
            try:
                self._robot.transform_turn_speed(direction, speed)
                time.sleep(max(0.01, abs(degrees) / max(1.0, float(speed))))
                self._robot.transform_stop()
            except Exception:
                logger.exception("Turn fallback failed")
        except Exception:
            logger.exception("Unexpected exception in safe_turn")

    def safe_move(self, direction: int, speed: float, cm: float):
        """Best-effort move. direction 0 = forward"""
        if not self.ensure_robot():
            return
        try:
            try:
                self._robot.transform_move_speed_times(direction, speed, int(cm), 1)
                return
            except Exception:
                pass

            try:
                self._robot.transform_move_speed(direction, speed)
                time.sleep(max(0.01, abs(cm) / max(1.0, float(speed))))
                self._robot.transform_stop()
            except Exception:
                logger.exception("Move fallback failed")
        except Exception:
            logger.exception("Unexpected exception in safe_move")

    def read_distance(self, sensor_id: int = SENSOR_ID) -> int:
        """Return distance in cm or -1 on failure."""
        if not self.ensure_robot():
            return -1
        try:
            d = self._robot.read_distance_data(sensor_id)
            if d is None:
                return -1
            return int(d)
        except Exception:
            logger.exception("Distance read error")
            return -1

    def play_tts(self, text: str, voice_type: int = 0, wait: bool = False):
        if not self.ensure_robot():
            return
        try:
            self._robot.play_audio_tts(text, voice_type=voice_type, wait=wait)
        except Exception:
            logger.exception("TTS failed (ignored)")

    def stop(self):
        """Attempt to stop all robot motion."""
        if not self.ensure_robot():
            return
        try:
            self._robot.transform_stop()
        except Exception:
            logger.exception("transform_stop failed (ignored)")

    def set_chassis_height(self, height: int):
        if not self.ensure_robot():
            return
        try:
            self._robot.transform_set_chassis_height(height)
        except Exception:
            logger.exception("set_chassis_height failed (ignored)")

    def reset_hardware_safely(self):
        """Try to restore robot to safe state using best available calls."""
        if not self.ensure_robot():
            # nothing to do
            return
        try:
            try:
                self._robot.transform_stop()
            except Exception:
                pass
            try:
                self._robot.transform_set_chassis_height(2)
            except Exception:
                pass
            try:
                self._robot.transform_adaption_control(False)
            except Exception:
                pass
            # safe call: if robot implements a restore/reset method, call it
            for candidate in ("transform_restore", "transform_restory", "transform_reset", "restore"):
                fn = getattr(self._robot, candidate, None)
                if callable(fn):
                    try:
                        fn()
                        logger.info("Called robot.%s()", candidate)
                    except Exception:
                        logger.exception("Calling robot.%s() failed", candidate)
        except Exception:
            logger.exception("Error during hardware reset (ignored)")

    def read_camera_frame(self) -> Optional[bytes]:
        """
        Try to read a JPEG-encoded camera frame from the robot.
        Return raw jpeg bytes or None if not available.
        """
        if not self.ensure_robot() or not self._camera_open:
            return None
        try:
            frame_data = self._robot.read_camera_data()
            if frame_data is None:
                return None
            # The robot returns a byte buffer (assumed jpg/encoded). Some devices return raw bytes that must be decoded.
            # Try to decode first; if decoding produces an image, re-encode to JPEG for the web streamer.
            nparr = np.frombuffer(frame_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                # maybe already a JPEG; return raw
                return frame_data
            ret, buffer = cv2.imencode('.jpg', img)
            if not ret:
                return None
            return buffer.tobytes()
        except Exception:
            logger.exception("Camera read error")
            return None

# single controller instance
robot_ctrl = RobotController()

# ---------------- Robot runtime state (shared) ----------------
robot_state_lock = threading.Lock()
robot_state: Dict = {
    "x_cm": 0.0,
    "y_cm": 0.0,
    "heading_deg": 0.0,
    "executing": False,
    "status_text": "idle",
    "last_stop": None,
    "obstacles": []
}

# execution thread control
_execution_thread: Optional[threading.Thread] = None
_execution_stop_event: Optional[threading.Event] = None

# ---------------- Utility functions ----------------
def normalize_deg(a: float) -> float:
    """Normalize degrees to [-180, 180)."""
    return (a + 180.0) % 360.0 - 180.0

def find_nearby_obstacle(x_cm: float, y_cm: float, thresh_cm: float = 60.0) -> Optional[Dict]:
    with robot_state_lock:
        for obs in robot_state["obstacles"]:
            ox = obs.get("x_cm", obs.get("center_x_cm", None))
            oy = obs.get("y_cm", obs.get("center_y_cm", None))
            if ox is None or oy is None:
                continue
            if math.hypot(ox - x_cm, oy - y_cm) <= thresh_cm:
                return obs
    return None

# ---------------- Camera feed ----------------
def gen_frames():
    """Yield multipart MJPEG frames. Never raises; yields a placeholder on errors."""
    # prepare a small black JPEG placeholder to avoid client disconnects
    placeholder = None
    try:
        black = np.zeros((120, 160, 3), dtype=np.uint8)
        ret, buf = cv2.imencode(".jpg", black)
        if ret:
            placeholder = buf.tobytes()
    except Exception:
        placeholder = b""

    while True:
        try:
            frame = robot_ctrl.read_camera_frame()
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                # no frame: yield placeholder every so often and sleep briefly
                if placeholder:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + placeholder + b'\r\n')
                time.sleep(0.05)
        except Exception:
            logger.exception("Camera generator exception; continuing.")
            time.sleep(0.2)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ---------------- Execution logic ----------------
def execute_route_thread(waypoints_px: List[Dict], scale_cm_per_px: float, stop_event: threading.Event):
    """
    Core navigation loop. This function is run in a separate thread.
    stop_event is used to interrupt execution early.
    """
    global _execution_thread, _execution_stop_event

    # update executing state
    with robot_state_lock:
        if robot_state["executing"]:
            logger.info("execute_route_thread called but robot already executing")
            return
        robot_state["executing"] = True
        robot_state["status_text"] = "starting route"

    try:
        robot_ctrl.ensure_robot()
        robot_ctrl.play_tts("Ik ga de route uitvoeren!", wait=False)
        robot_ctrl.set_chassis_height(4)

        if not waypoints_px or len(waypoints_px) < 2:
            with robot_state_lock:
                robot_state["executing"] = False
                robot_state["status_text"] = "no route"
            return

        # convert to cm coordinates
        points_cm = [{"x": float(p["x"]) * scale_cm_per_px, "y": float(p["y"]) * scale_cm_per_px} for p in waypoints_px]

        # helper: insert a small lateral detour (split) between from_idx and to_idx
        def split_segment_insert(start_x, start_y, target_x, target_y, cur_heading, scale, from_idx, to_idx, distance_cm=None) -> Tuple[bool, Optional[Dict]]:
            nearby = find_nearby_obstacle(start_x, start_y, thresh_cm=60.0)
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

                if find_nearby_obstacle(cand_x, cand_y, thresh_cm=30.0):
                    continue

                # desired heading to the original target from the detour candidate
                desired_heading = normalize_deg(math.degrees(math.atan2(target_y - cand_y, target_x - cand_x)))

                points_cm.insert(to_idx, {
                    "x": cand_x,
                    "y": cand_y,
                    "_detour": True,
                    "desired_heading": desired_heading
                })

                # estimate obstacle center a short distance ahead
                obs_dist = min(max((distance_cm or 30.0), 10.0), float(STOP_DISTANCE_CM))
                obs_x = start_x + fx * obs_dist
                obs_y = start_y + fy * obs_dist

                obs = {
                    "timestamp": time.time(),
                    "reason": "split_inserted",
                    "x_cm": obs_x,
                    "y_cm": obs_y,
                    "x_px": obs_x / scale,
                    "y_px": obs_y / scale,
                    "width_cm": max(30.0, SPLIT_LATERAL_OFFSET_CM * 2.0),
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

        def attempt_avoidance_insert(cx, cy, cur_heading, scale, from_idx, to_idx) -> Tuple[bool, Optional[Dict]]:
            nearby = find_nearby_obstacle(cx, cy, thresh_cm=80.0)
            if nearby and nearby.get("avoid_attempts", 0) >= MAX_AVOID_ATTEMPTS:
                return False, None

            lateral_offset = max(MIN_LATERAL_OFFSET_CM, STOP_DISTANCE_CM + AVOID_CLEARANCE_EXTRA_CM)
            # try left then right (consistent with earlier code)
            sides = [("left", 1), ("right", -1)]

            for side_name, sign in sides:
                perp_angle = math.radians(cur_heading + 90.0 * sign)
                px = math.cos(perp_angle)
                py = math.sin(perp_angle)

                wp_x = cx + px * lateral_offset
                wp_y = cy + py * lateral_offset

                if find_nearby_obstacle(wp_x, wp_y, thresh_cm=40.0):
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
                    "width_cm": lateral_offset * 2.0,
                    "side_chosen": side_name,
                    "avoid_attempts": 0,
                    "stop_segment_from_idx": from_idx,
                    "stop_segment_to_idx": to_idx
                }

                with robot_state_lock:
                    robot_state["obstacles"].append(obs)

                return True, obs

            return False, None
        
        def _resolve_obstacles_for_detour(detour_x, detour_y, scale, tol_cm=15.0):
            """Remove (or mark) obstacle entries that correspond to the given detour waypoint.
            Use a small tolerance so we only resolve obstacles associated with this detour.
            """
            with robot_state_lock:
                remaining = []
                for o in robot_state["obstacles"]:
                    ox = o.get("detour_wp_x_cm") or o.get("x_cm")
                    oy = o.get("detour_wp_y_cm") or o.get("y_cm")
                    if ox is None or oy is None:
                        remaining.append(o)
                        continue
                    if math.hypot(ox - detour_x, oy - detour_y) <= tol_cm:
                        # mark as resolved for debugging, but drop from active list
                        logger.info("Resolving obstacle created for detour at (%.1f,%.1f)", detour_x, detour_y)
                        continue
                    remaining.append(o)
                robot_state["obstacles"] = remaining


        # initialize pose
        with robot_state_lock:
            robot_state["x_cm"] = points_cm[0]["x"]
            robot_state["y_cm"] = points_cm[0]["y"]
            current_heading = robot_state["heading_deg"]
            robot_state["status_text"] = "executing"

        idx = 1
        avoid_mode = False
        avoid_mode_until = 0.0
        current_active_obstacle = None

        while True:
            # check for external stop
            if stop_event.is_set():
                logger.info("Execution stopped by external request.")
                with robot_state_lock:
                    robot_state["executing"] = False
                    robot_state["status_text"] = "stopped"
                robot_ctrl.stop()
                return

            if idx >= len(points_cm):
                break

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

            # reached waypoint?
            if dist_to_target < 0.5:
                # if we just arrived at a detour or clearance waypoint, resolve obstacles
                if points_cm[idx].get("_detour") or points_cm[idx].get("_clearance"):
                    _resolve_obstacles_for_detour(points_cm[idx]["x"], points_cm[idx]["y"], scale_cm_per_px)
                idx += 1
                continue


            # heading control
            target_heading = math.degrees(math.atan2(dy, dx))
            delta = normalize_deg(target_heading - current_heading)

            if abs(delta) > 1.0:
                turn_mag = min(abs(delta), MAX_IMMEDIATE_TURN_DEG)
                turn_dir = 2 if delta > 0 else 3
                robot_ctrl.safe_turn(turn_dir, TURN_SPEED_DEG_S, float(turn_mag))

                turn_amount = math.copysign(turn_mag, delta)
                with robot_state_lock:
                    robot_state["heading_deg"] = normalize_deg(current_heading + turn_amount)
                    current_heading = robot_state["heading_deg"]

            # read distance sensor
            dist_read = robot_ctrl.read_distance()

            active_stop_distance = STOP_DISTANCE_CM
            if avoid_mode:
                active_stop_distance = min(active_stop_distance, 12.0)

            if dist_read >= 0 and dist_read <= active_stop_distance:
                # attempt split insertion first, then fallback detour
                succeeded, obs = split_segment_insert(rx, ry, tx, ty, current_heading, scale_cm_per_px, idx - 1, idx, dist_read)
                if not succeeded:
                    succeeded, obs = attempt_avoidance_insert(rx, ry, current_heading, scale_cm_per_px, idx - 1, idx)

                if succeeded:
                    avoid_mode = True
                    # give more time to clear obstacle
                    avoid_mode_until = time.time() + 10.0
                    current_active_obstacle = obs

                    # immediately orient toward inserted waypoint
                    new_tx = points_cm[idx]["x"]
                    new_ty = points_cm[idx]["y"]
                    new_heading = math.degrees(math.atan2(new_ty - ry, new_tx - rx))
                    delta2 = normalize_deg(new_heading - current_heading)
                    if abs(delta2) > 1.0:
                        turn_mag = min(abs(delta2), MAX_IMMEDIATE_TURN_DEG)
                        turn_dir = 2 if delta2 > 0 else 3
                        robot_ctrl.safe_turn(turn_dir, TURN_SPEED_DEG_S, float(turn_mag))
                        turn_amount = math.copysign(turn_mag, delta2)
                        with robot_state_lock:
                            robot_state["heading_deg"] = normalize_deg(current_heading + turn_amount)

                    with robot_state_lock:
                        # increment attempts on the obstacle we just created or found
                        if current_active_obstacle is not None:
                            current_active_obstacle["avoid_attempts"] = current_active_obstacle.get("avoid_attempts", 0) + 1

                    # insert a small forward "clearance" waypoint after the detour so robot passes the obstacle
                    rad_det = math.radians(new_heading)
                    clearance_x = new_tx + math.cos(rad_det) * (STOP_DISTANCE_CM + 10.0)
                    clearance_y = new_ty + math.sin(rad_det) * (STOP_DISTANCE_CM + 10.0)
                    # insert clearance after the detour (detour is at index idx)
                    points_cm.insert(idx + 1, {"x": clearance_x, "y": clearance_y, "_clearance": True})

                    # re-evaluate (HARD restart of loop)
                    continue


                # hard block: couldn't insert detour
                with robot_state_lock:
                    robot_state["executing"] = False
                    robot_state["status_text"] = "blocked"
                robot_ctrl.stop()
                robot_ctrl.play_tts("Ik kan niet verder, het pad is geblokkeerd.", wait=False)
                return

            # move forward in chunks
            move_cm = min(MAX_MOVE_CHUNK_CM, dist_to_target)  # float, not int
            if move_cm <= 0:
                # might be a rounding issue; skip iteration briefly
                time.sleep(0.05)
                continue

            robot_ctrl.safe_move(0, FORWARD_SPEED_CM_S, float(move_cm))

            logger.info("Moved %.1f cm toward wp %d; new pose = (%.2f, %.2f), target = (%.2f, %.2f), dist_to_target=%.2f",
            move_cm, idx, robot_state["x_cm"], robot_state["y_cm"], tx, ty, dist_to_target)

            # update internal pose estimate
            rad = math.radians(current_heading)
            with robot_state_lock:
                robot_state["x_cm"] += move_cm * math.cos(rad)
                robot_state["y_cm"] += move_cm * math.sin(rad)
                robot_state["status_text"] = f"moving to wp {idx}"

            time.sleep(0.05)

        # finished properly
        with robot_state_lock:
            robot_state["executing"] = False
            robot_state["status_text"] = "route complete"

        robot_ctrl.play_tts("Ik heb de route uitgevoerd.", wait=False)
        robot_ctrl.set_chassis_height(2)

    except Exception as e:
        logger.exception("Execution error: %s", e)
        with robot_state_lock:
            robot_state["executing"] = False
            robot_state["status_text"] = f"error: {e}"

# -------------------- Flask endpoints --------------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/execute_strict', methods=['POST'])
def execute_strict():
    global _execution_thread, _execution_stop_event

    req = request.json or {}
    waypoints = req.get('waypoints')
    scale = float(req.get('scale_cm_per_px', SCALE_CM_PER_PX))

    if not waypoints or len(waypoints) < 2:
        return jsonify({"ok": False, "message": "Need at least 2 waypoints"}), 400

    with robot_state_lock:
        if robot_state.get("executing"):
            return jsonify({"ok": False, "message": "Robot is already executing"}), 409

    # prepare stop event and thread
    stop_event = threading.Event()
    thread = threading.Thread(target=execute_route_thread, args=(waypoints, scale, stop_event), daemon=True)
    _execution_thread = thread
    _execution_stop_event = stop_event
    thread.start()

    return jsonify({"ok": True, "message": "Execution started"})

@app.route('/execute_tail', methods=['POST'])
def execute_tail():
    global _execution_thread, _execution_stop_event

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

    stop_event = threading.Event()
    thread = threading.Thread(target=execute_route_thread, args=(waypoints_to_execute, scale, stop_event), daemon=True)
    _execution_thread = thread
    _execution_stop_event = stop_event
    thread.start()
    return jsonify({"ok": True, "message": "Tail execution started"})

@app.route('/stop', methods=['POST'])
def stop_execution():
    """Request an immediate stop of the running execution thread (if any)."""
    global _execution_thread, _execution_stop_event
    with robot_state_lock:
        if not robot_state.get("executing"):
            return jsonify({"ok": False, "message": "Robot is not executing"}), 409

    if _execution_stop_event is not None:
        _execution_stop_event.set()

    # also ask robot to stop immediately
    robot_ctrl.stop()

    return jsonify({"ok": True, "message": "Stop requested"})

@app.route('/reset', methods=['POST', 'GET'])
def reset():
    robot_ctrl.reset_hardware_safely()
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
        obs_copy = []
        for o in robot_state["obstacles"]:
            copy_o = dict(o)
            if "x_cm" in copy_o and "x_px" not in copy_o:
                copy_o["x_px"] = copy_o["x_cm"] / SCALE_CM_PER_PX
            if "y_cm" in copy_o and "y_px" not in copy_o:
                copy_o["y_px"] = copy_o["y_cm"] / SCALE_CM_PER_PX
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
    # on startup, we do not force robot initialization; let endpoints trigger it lazily
    logger.info("Starting Flask server for route execution")
    app.run(host='0.0.0.0', port=5000, debug=False)
