#!/usr/bin/env python3
"""
WRO Future Engineers 2026 - Bench Test & Teleop Dashboard
=========================================================
"""
import argparse
import json
import logging
import math
import os
import threading
import time
from dataclasses import dataclass, field

# Keep the BLAS libraries single-threaded. numpy/OpenCV spawning their own
# thread pools on top of our threads is a common source of instability and
# jitter on a Raspberry Pi. Must be set before numpy/cv2 are imported.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2
import numpy as np
import flask.cli
from flask import Flask, Response, jsonify, render_template_string, request

# ============================================================================
# SECTION 1 - TUNABLE PARAMETERS
# ============================================================================

# ---------------------------------------------------------------- 1.1 chassis
WHEELBASE_MM = 125.0             # front axle centre to rear axle centre
FRONT_TRACK_MM = 140.0           # left to right front wheel centre
REAR_TRACK_MM = 125.0            # left to right rear wheel centre
WHEEL_DIAMETER_MM = 68.0         # measured with the tyre fitted, not the rim
ROBOT_LENGTH_MM = 250.0          # used only to sanity-check rule 11.1 / 9.17
ROBOT_WIDTH_MM = 170.0           # limit is 300 x 200 mm, height 300 mm

# ------------------------------------------------------- 1.2 steering servo
STEER_CENTRE_CMD = 100           # servo command that points the wheels straight
STEER_CMD_MIN = -140                # Fixed to 0 for proper mapping
STEER_CMD_MAX = 140              # servo command at full right lock
STEER_MAX_LEFT_DEG = 30.0        # MEASURE: real wheel angle at full left
STEER_MAX_RIGHT_DEG = 30.0       # MEASURE: real wheel angle at full right

# ---------------------------------------------------------- 1.3 drive motor
DRIVE_CMD_MAX = 255              # ESP32 accepts -255..255
TICKS_PER_WHEEL_REV = 2000       # MEASURE: roll the wheel 10 turns by hand,
THROTTLE_POLARITY = -1           # set to -1 if a positive command drives the car backwards.

# Slew-rate limits.
THROTTLE_RAMP_PER_SEC = 400.0    # throttle units per second (0->255 in ~0.6 s)
STEER_RAMP_PER_SEC = 180.0       # degrees per second
RAMP_LOOP_HZ = 50                # how often the ramp loop ticks

# ----------------------------------------------------------- 1.4 serial link
SERIAL_PORT = "/dev/esp32"       # or COM5 on Windows
SERIAL_BAUD = 115200
SERIAL_RECONNECT_SEC = 2.0
COMMAND_WATCHDOG_SEC = 0.5       # stop the motor if no command arrives in time

# ---------------------------------------------------------------- 1.5 camera
CAMERA_DEVICE = "/dev/webcam"    # or an integer index such as 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
JPEG_QUALITY = 70                # lower = less WiFi bandwidth on the Pi

CAMERA_HEIGHT_MM = 160.0         # lens centre above the mat
CAMERA_TILT_DEG = 0.0            # positive = tilted down toward the mat
CAMERA_FOCAL_PX = 600.0          # pinhole focal length in pixels

# ------------------------------------------------------- 1.6 pillar geometry
PILLAR_WIDTH_MM = 50             # rule 13.19: 50 x 50 x 100 mm
PILLAR_HEIGHT_MM = 100
PILLAR_ASPECT_NOMINAL = PILLAR_WIDTH_MM / PILLAR_HEIGHT_MM   # 0.5

# ------------------------------------------------------ 1.7 detection backend
DETECTION_BACKEND = "auto"

# ---------------------------------------------------------- 1.8 ONNX detector
ONNX_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.onnx")
ONNX_INPUT_SIZE = 416            # confirmed from best.onnx's own input tensor
ONNX_CONF_THRESHOLD = 0.45
ONNX_NMS_IOU_THRESHOLD = 0.45
ONNX_EVERY_N_FRAMES = 2          # run inference on every Nth frame (CPU budget)
ONNX_CLASS_NAMES = ["greenbox", "redbox", "xparking"]
CLASS_NAME_TO_COLOUR = {
    "greenbox": "GREEN",
    "redbox": "RED",
}

# ----------------------------------------------------------- 1.9 HSV detector
# Minimum value (brightness) set to 100 to completely block black walls!
HSV_RED_RANGES = [((0, 120, 100), (10, 255, 255)), ((170, 120, 100), (179, 255, 255))]
HSV_GREEN_RANGE = ((40, 120, 100), (85, 255, 255))
HSV_MORPH_KERNEL = 5             # noise clean-up kernel, odd numbers only
HSV_MIN_AREA_PX = 300            # reject specks
HSV_MAX_AREA_PX = 150000         # INCREASED: detect closer/larger boxes
HSV_MIN_ASPECT = 0.25            # a pillar is 0.5 wide/high; allow for angle
HSV_MAX_ASPECT = 0.90
RULEBOOK_RED_RGB = (238, 39, 55)
RULEBOOK_GREEN_RGB = (68, 214, 44)

# ----------------------------------------------------------------- 1.10 LiDAR
LIDAR_PORT = "/dev/rplidar"
LIDAR_BAUD = 256000              # RPLidar A2M12 default
LIDAR_HEIGHT_MM = 80.0           # mounted at the nose of the chassis
LIDAR_RECONNECT_SEC = 2.0

LIDAR_FRONT_DEG = 0              # which raw LiDAR angle points at the nose
LIDAR_ANGLE_CW = True            # True if raw angles grow clockwise from above

LIDAR_MIN_RANGE_MM = 150         # ignore returns closer than this (own body)
LIDAR_MAX_RANGE_MM = 2200        # the mat is 3200 mm across (rule 13.1)
# TRUE: Ignore the back 180 degrees to save CPU and filter chassis noise!
LIDAR_USE_FRONT_HALF_ONLY = True  
LIDAR_BLOCKED_RAW_RANGES = []    # e.g. [(160, 200)]

LIDAR_SECTOR_PERCENTILE = 15     # 0 = pure minimum, 50 = median
LIDAR_SECTOR_MIN_POINTS = 3      # fewer points than this and we report "unknown"
# EXPANDED SECTORS: Perfectly meshes a 180-degree coverage array
SECTOR_FRONT_WIDTH_DEG = 40      # -20 to 20 straight ahead
SECTOR_DIAGONAL_WIDTH_DEG = 50   # 50 degree sweep for corners
SECTOR_DIAGONAL_CENTRE_DEG = 45  # Sitting perfectly at 45 degree angles
SECTOR_SIDE_WIDTH_DEG = 90       # Enormous 90 degree wedge looking full left/right

# ------------------------------------------------------- 1.11 driving control
DRIVE_CRUISE_SPEED = 190         # throttle while lane following (0-255)
DRIVE_CORNER_SPEED = 190         # INCREASED: Give motor torque to push through corners!
DRIVE_PILLAR_SPEED = 200         # throttle while passing a pillar
DRIVE_MIN_SPEED = 180            # motor needs at least 180 to push the car smoothly

CENTRE_STEER_GAIN_DEG_PER_MM = 0.020   # deg of steer per mm of imbalance
CENTRE_STEER_MAX_DEG = 14.0            # cap so it cannot fight a corner
CENTRE_DEADBAND_MM = 25                # ignore imbalance smaller than this

# Obstacle Detection & Escape Logic
FRONT_OBSTACLE_MM = 700          # EXACTLY 70cm for front triggering
SIDE_OBSTACLE_MM = 500           # EXACTLY 50cm for left/right triggering
ESCAPE_REVERSE_MM = 350          # Safe Mode: If closer than 35cm to front, go backward!

CORNER_EXIT_MM = 2000             # Distance to return to straight cruising
CORNER_STEER_DEG = 26.0          # Maximum steering angle for tightest part of corner
CORNER_MIN_HOLD_SEC = 0.15       # do not leave the corner state before this
TRACK_DIRECTION = "auto"         # "auto" | "cw" | "ccw"

# ------------------------------------------------------- 1.12 pillar passing
PILLAR_TARGET_X_RED = 0.18       # push red pillars to the left of the frame
PILLAR_TARGET_X_GREEN = 0.82     # push green pillars to the right of the frame
PILLAR_STEER_GAIN_DEG_PER_PX = 0.060   # proportional gain on the pixel error
PILLAR_STEER_MAX_DEG = 26.0            # cap on the pillar steering command
PILLAR_ENGAGE_MM = 2000          # increased to detect pillars from further away!
PILLAR_RELEASE_MM = 260          # too close to see - hold the last command
PILLAR_MIN_CONFIDENCE = 0.50     # ignore detections weaker than this
PILLAR_DECISION_TIMEOUT_SEC = 1.5   # increased so it remembers the box longer

# -------------------------------------------------------------- 1.13 web page
WEB_HOST = "0.0.0.0"
WEB_PORT = 5000
TELEMETRY_POLL_MS = 200          # how often the page asks for numbers
RADAR_POLL_MS = 250              # how often the page asks for LiDAR points
CONTROL_LOOP_HZ = 20             # autonomous decision rate

# ============================================================================
# SECTION 2 - SMALL MATHS HELPERS
# ============================================================================

WHEEL_CIRCUMFERENCE_MM = math.pi * WHEEL_DIAMETER_MM

def clamp(value, low, high):
    """Keep value inside [low, high]."""
    return max(low, min(high, value))

def wrap_180(angle_deg):
    """Fold any angle into the range -180..180."""
    return (angle_deg + 180.0) % 360.0 - 180.0

def steer_deg_to_cmd(angle_deg):
    """Convert a wheel angle in degrees to a servo command."""
    angle_deg = clamp(angle_deg, -STEER_MAX_LEFT_DEG, STEER_MAX_RIGHT_DEG)
    if angle_deg >= 0:
        span = STEER_MAX_RIGHT_DEG or 1e-6
        cmd = STEER_CENTRE_CMD + (angle_deg / span) * (STEER_CMD_MAX - STEER_CENTRE_CMD)
    else:
        span = STEER_MAX_LEFT_DEG or 1e-6
        cmd = STEER_CENTRE_CMD + (angle_deg / span) * (STEER_CENTRE_CMD - STEER_CMD_MIN)
    return int(clamp(round(cmd), STEER_CMD_MIN, STEER_CMD_MAX))

def steer_cmd_to_deg(steer_cmd):
    """Inverse of steer_deg_to_cmd, used for telemetry and the ramp loop."""
    if steer_cmd >= STEER_CENTRE_CMD:
        span = (STEER_CMD_MAX - STEER_CENTRE_CMD) or 1
        return (steer_cmd - STEER_CENTRE_CMD) / span * STEER_MAX_RIGHT_DEG
    span = (STEER_CENTRE_CMD - STEER_CMD_MIN) or 1
    return -(STEER_CENTRE_CMD - steer_cmd) / span * STEER_MAX_LEFT_DEG

def pillar_distance_mm(box_height_px):
    """Estimate range to a pillar from its height in the image (pinhole model)."""
    if box_height_px <= 1:
        return float("inf")
    return CAMERA_FOCAL_PX * PILLAR_HEIGHT_MM / float(box_height_px)

# ============================================================================
# SECTION 3 - ROBOT LINK (serial to the ESP32)
# ============================================================================

class RobotLink:
    """Owns the serial port, the command ramp, and the odometry maths."""

    def __init__(self, port=SERIAL_PORT, baud=SERIAL_BAUD):
        self.port = port
        self.baud = baud
        self._serial = None
        self._lock = threading.RLock()

        self.connected = False
        self.running = False

        self.encoder_ticks = 0
        self.linear_velocity_mmps = 0.0
        self.throttle_cmd = 0
        self.steer_cmd = STEER_CENTRE_CMD

        self.target_throttle = 0
        self.target_steer_deg = 0.0
        self.last_command_time = 0.0

        self._prev_ticks = 0
        self._prev_time = time.time()

    def start(self):
        if self.running:
            return
        self.running = True
        threading.Thread(target=self._serial_loop, daemon=True).start()
        threading.Thread(target=self._watchdog_loop, daemon=True).start()
        threading.Thread(target=self._ramp_loop, daemon=True).start()

    def stop(self):
        self.running = False
        self._write("M,0")
        self._write("S,%d" % STEER_CENTRE_CMD)
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:
                pass
        self._serial = None
        self.connected = False

    def set_target_drive(self, throttle):
        with self._lock:
            self.target_throttle = int(clamp(throttle, -DRIVE_CMD_MAX, DRIVE_CMD_MAX))
            self.last_command_time = time.time()

    def set_target_steer_deg(self, angle_deg):
        with self._lock:
            self.target_steer_deg = clamp(angle_deg, -STEER_MAX_LEFT_DEG, STEER_MAX_RIGHT_DEG)
            self.last_command_time = time.time()

    def emergency_stop(self):
        with self._lock:
            self.target_throttle = 0
        self._set_drive_now(0)

    def angular_velocity_radps(self):
        with self._lock:
            v_mps = self.linear_velocity_mmps / 1000.0
            steer_cmd = self.steer_cmd
        wheelbase_m = WHEELBASE_MM / 1000.0
        if wheelbase_m <= 0:
            return 0.0
        return v_mps * math.tan(math.radians(steer_cmd_to_deg(steer_cmd))) / wheelbase_m

    def state(self):
        with self._lock:
            return {
                "connected": self.connected,
                "encoder_ticks": self.encoder_ticks,
                "linear_vel_mmps": round(self.linear_velocity_mmps, 1),
                "angular_vel_radps": round(self.angular_velocity_radps(), 3),
                "throttle_cmd": self.throttle_cmd,
                "steer_cmd": self.steer_cmd,
                "steer_deg": round(steer_cmd_to_deg(self.steer_cmd), 1),
            }

    def _write(self, message):
        if self._serial is None:
            return
        try:
            self._serial.write((message + "\n").encode())
        except Exception:
            pass

    def _set_drive_now(self, throttle):
        throttle = int(clamp(throttle, -DRIVE_CMD_MAX, DRIVE_CMD_MAX))
        with self._lock:
            self.throttle_cmd = throttle
        self._write("M,%d" % (throttle * THROTTLE_POLARITY))

    def _set_steer_now(self, angle_deg):
        cmd = steer_deg_to_cmd(angle_deg)
        with self._lock:
            self.steer_cmd = cmd
        self._write("S,%d" % cmd)

    def _serial_loop(self):
        import serial
        while self.running:
            if self._serial is None:
                try:
                    self._serial = serial.Serial(self.port, self.baud, timeout=0.1)
                    time.sleep(2.0)
                    self.connected = True
                    logging.info("[robot] connected on %s", self.port)
                except Exception as exc:
                    self.connected = False
                    logging.warning("[robot] %s - retrying", exc)
                    time.sleep(SERIAL_RECONNECT_SEC)
                    continue
            try:
                line = self._serial.readline().decode(errors="ignore").strip()
            except Exception:
                self.connected = False
                try:
                    self._serial.close()
                except Exception:
                    pass
                self._serial = None
                continue
            if line.startswith("ENC,"):
                self._handle_encoder(line)

    def _handle_encoder(self, line):
        try:
            ticks = int(line.split(",")[1])
        except (IndexError, ValueError):
            return
        now = time.time()
        with self._lock:
            dt = now - self._prev_time
            if dt > 0:
                delta_ticks = ticks - self._prev_ticks
                distance_mm = (delta_ticks / TICKS_PER_WHEEL_REV) * WHEEL_CIRCUMFERENCE_MM
                self.linear_velocity_mmps = distance_mm / dt
            self._prev_ticks = ticks
            self._prev_time = now
            self.encoder_ticks = ticks

    def _watchdog_loop(self):
        while self.running:
            stale = time.time() - self.last_command_time > COMMAND_WATCHDOG_SEC
            if self._serial is not None and stale:
                with self._lock:
                    self.target_throttle = 0
                    self.target_steer_deg = 0.0
                self._write("M,0")
            time.sleep(0.1)

    def _ramp_loop(self):
        dt = 1.0 / RAMP_LOOP_HZ
        throttle_step = THROTTLE_RAMP_PER_SEC * dt
        steer_step = STEER_RAMP_PER_SEC * dt
        while self.running:
            with self._lock:
                want_throttle = self.target_throttle
                want_steer = self.target_steer_deg
                have_throttle = self.throttle_cmd
            have_steer = steer_cmd_to_deg(self.steer_cmd)

            # "MOTOR LISTENS TO SERVO" - Drop throttle to let wheels physically turn
            steer_error = abs(want_steer - have_steer)
            if steer_error > 5.0 and want_throttle > 0:
                want_throttle = int(want_throttle * 0.70)

            next_throttle = have_throttle + clamp(
                want_throttle - have_throttle, -throttle_step, throttle_step)
            next_steer = have_steer + clamp(
                want_steer - have_steer, -steer_step, steer_step)

            if round(next_throttle) != have_throttle:
                self._set_drive_now(int(round(next_throttle)))
            if abs(next_steer - have_steer) > 0.01:
                self._set_steer_now(next_steer)
            time.sleep(dt)

# ============================================================================
# SECTION 4 - LIDAR SENSOR
# ============================================================================

def raw_to_robot_angle(raw_deg):
    delta = raw_deg - LIDAR_FRONT_DEG
    if not LIDAR_ANGLE_CW:
        delta = -delta
    return wrap_180(delta)

def sector_distance_mm(points, centre_deg, width_deg):
    half = width_deg / 2.0
    inside = [d for angle, d in points if abs(wrap_180(angle - centre_deg)) <= half]
    if len(inside) < LIDAR_SECTOR_MIN_POINTS:
        return None
    return float(np.percentile(inside, LIDAR_SECTOR_PERCENTILE))

@dataclass
class LidarView:
    front: float = None
    front_left: float = None
    front_right: float = None
    left: float = None
    right: float = None
    point_count: int = 0

    def as_dict(self):
        def fmt(v):
            return None if v is None else round(v)
        return {
            "front": fmt(self.front),
            "front_left": fmt(self.front_left),
            "front_right": fmt(self.front_right),
            "left": fmt(self.left),
            "right": fmt(self.right),
            "point_count": self.point_count,
        }

class LidarSensor:
    def __init__(self, port=LIDAR_PORT, baud=LIDAR_BAUD):
        self.port = port
        self.baud = baud
        self.connected = False
        self.running = False
        self._lock = threading.Lock()
        self._points = []
        self._raw_points = []

    def start(self):
        if self.running:
            return
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self.running = False
        self.connected = False

    def points(self):
        with self._lock:
            return list(self._points)

    def raw_points(self):
        with self._lock:
            return list(self._raw_points)

    def view(self):
        pts = self.points()
        return LidarView(
            front=sector_distance_mm(pts, 0.0, SECTOR_FRONT_WIDTH_DEG),
            front_left=sector_distance_mm(pts, -SECTOR_DIAGONAL_CENTRE_DEG, SECTOR_DIAGONAL_WIDTH_DEG),
            front_right=sector_distance_mm(pts, +SECTOR_DIAGONAL_CENTRE_DEG, SECTOR_DIAGONAL_WIDTH_DEG),
            left=sector_distance_mm(pts, -90.0, SECTOR_SIDE_WIDTH_DEG),
            right=sector_distance_mm(pts, +90.0, SECTOR_SIDE_WIDTH_DEG),
            point_count=len(pts),
        )

    @staticmethod
    def _raw_angle_blocked(raw_deg):
        return any(lo <= raw_deg <= hi for lo, hi in LIDAR_BLOCKED_RAW_RANGES)

    def _accept(self, raw_deg, distance_mm):
        if not (LIDAR_MIN_RANGE_MM <= distance_mm <= LIDAR_MAX_RANGE_MM):
            return False
        if self._raw_angle_blocked(raw_deg):
            return False
        if LIDAR_USE_FRONT_HALF_ONLY and abs(raw_to_robot_angle(raw_deg)) > 90.0:
            return False
        return True

    def _loop(self):
        from rplidar import RPLidar, RPLidarException
        while self.running:
            lidar = None
            try:
                lidar = RPLidar(self.port, baudrate=self.baud, timeout=3)
                lidar.stop()
                lidar.stop_motor()
                time.sleep(0.3)
                lidar.clean_input()
                lidar.start_motor()
                time.sleep(1.0)
                self.connected = True
                logging.info("[lidar] scanning on %s", self.port)
                # FIX: Increased buffer size from 3000 to 8000 to prevent CPU spikes from crashing it
                for scan in lidar.iter_scans(max_buf_meas=8000):
                    if not self.running:
                        break
                    raw = [(m[1], m[2]) for m in scan if self._accept(m[1], m[2])]
                    with self._lock:
                        self._raw_points = raw
                        self._points = [(raw_to_robot_angle(a), d) for a, d in raw]
            except (RPLidarException, Exception) as exc:
                self.connected = False
                logging.warning("[lidar] %s - retrying", exc)
            finally:
                if lidar is not None:
                    try:
                        lidar.stop()
                        lidar.stop_motor()
                        lidar.disconnect()
                    except Exception:
                        pass
            if self.running:
                time.sleep(LIDAR_RECONNECT_SEC)

# ============================================================================
# SECTION 5 - VISION
# ============================================================================

@dataclass
class Detection:
    colour: str                 # "RED" or "GREEN"
    confidence: float
    box: tuple                  # (x1, y1, x2, y2) in frame pixels
    timestamp: float = field(default_factory=time.time)

    @property
    def cx(self):
        return (self.box[0] + self.box[2]) // 2

    @property
    def cy(self):
        return (self.box[1] + self.box[3]) // 2

    @property
    def width_px(self):
        return self.box[2] - self.box[0]

    @property
    def height_px(self):
        return self.box[3] - self.box[1]

    @property
    def distance_mm(self):
        return pillar_distance_mm(self.height_px)

    def as_dict(self):
        return {
            "colour": self.colour,
            "conf": round(self.confidence, 2),
            "cx": self.cx,
            "cy": self.cy,
            "distance_mm": round(self.distance_mm) if math.isfinite(self.distance_mm) else None,
        }

class OnnxPillarDetector:
    name = "onnx"

    def __init__(self, model_path=ONNX_MODEL_PATH, input_size=ONNX_INPUT_SIZE):
        self.input_size = input_size
        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        logging.info("[vision] ONNX model loaded: %s (%d px, classes=%s)",
                     model_path, input_size, ONNX_CLASS_NAMES)

    def detect(self, frame):
        height, width = frame.shape[:2]

        side = max(height, width)
        canvas = np.zeros((side, side, 3), dtype=np.uint8)
        canvas[:height, :width] = frame
        scale = side / float(self.input_size)

        blob = cv2.dnn.blobFromImage(
            canvas, 1 / 255.0, (self.input_size, self.input_size),
            swapRB=True, crop=False)
        self.net.setInput(blob)
        output = self.net.forward()

        rows = self._normalise_output(output)
        if rows is None:
            return []

        num_classes = len(ONNX_CLASS_NAMES)
        columns = rows.shape[1]
        has_objectness = (columns == 5 + num_classes)
        score_start = 5 if has_objectness else 4

        boxes, scores, class_ids = [], [], []
        for row in rows:
            class_scores = row[score_start:score_start + num_classes]
            class_id = int(np.argmax(class_scores))
            confidence = float(class_scores[class_id])
            if has_objectness:
                confidence *= float(row[4])
            if confidence < ONNX_CONF_THRESHOLD:
                continue
            cx, cy, w, h = row[0], row[1], row[2], row[3]
            boxes.append([int((cx - w / 2) * scale), int((cy - h / 2) * scale),
                          int(w * scale), int(h * scale)])
            scores.append(confidence)
            class_ids.append(class_id)

        if not boxes:
            return []

        keep = cv2.dnn.NMSBoxes(boxes, scores, ONNX_CONF_THRESHOLD,
                                ONNX_NMS_IOU_THRESHOLD)
        detections = []
        for index in np.array(keep).flatten():
            index = int(index)
            class_id = class_ids[index]
            if class_id >= len(ONNX_CLASS_NAMES):
                continue
            colour = CLASS_NAME_TO_COLOUR.get(ONNX_CLASS_NAMES[class_id])
            if colour is None:
                continue
            x, y, w, h = boxes[index]
            detections.append(Detection(
                colour=colour,
                confidence=scores[index],
                box=(max(0, x), max(0, y),
                     min(width, x + w), min(height, y + h)),
            ))
        return detections

    @staticmethod
    def _normalise_output(output):
        array = np.squeeze(output)
        if array.ndim != 2:
            return None
        if array.shape[0] < array.shape[1]:
            array = array.T
        return array

class HsvPillarDetector:
    name = "hsv"

    def __init__(self):
        self.kernel = np.ones((HSV_MORPH_KERNEL, HSV_MORPH_KERNEL), np.uint8)
        logging.info("[vision] HSV colour detector active")

    def detect(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        red_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for low, high in HSV_RED_RANGES:
            red_mask |= cv2.inRange(hsv, np.array(low), np.array(high))

        green_mask = cv2.inRange(hsv, np.array(HSV_GREEN_RANGE[0]),
                                 np.array(HSV_GREEN_RANGE[1]))

        detections = []
        for mask, colour in ((self._clean(red_mask), "RED"),
                             (self._clean(green_mask), "GREEN")):
            detections.extend(self._blobs(mask, colour))
        return detections

    def _clean(self, mask):
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)

    def _blobs(self, mask, colour):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        found = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if not (HSV_MIN_AREA_PX <= area <= HSV_MAX_AREA_PX):
                continue
            x, y, w, h = cv2.boundingRect(contour)
            aspect = (w / h) if h else 0.0
            if not (HSV_MIN_ASPECT <= aspect <= HSV_MAX_ASPECT):
                continue
            fill = area / float(max(1, w * h))
            shape = 1.0 - min(1.0, abs(aspect - PILLAR_ASPECT_NOMINAL) / PILLAR_ASPECT_NOMINAL)
            found.append(Detection(
                colour=colour,
                confidence=round(clamp(0.5 * fill + 0.5 * shape, 0.0, 1.0), 2),
                box=(x, y, x + w, y + h),
            ))
        return found

def build_detector():
    if DETECTION_BACKEND == "hsv":
        return HsvPillarDetector()
    try:
        return OnnxPillarDetector()
    except Exception as exc:
        if DETECTION_BACKEND == "auto":
            logging.warning("[vision] ONNX unavailable (%s) - using HSV instead", exc)
            return HsvPillarDetector()
        logging.error("[vision] ONNX failed to load: %s - detection disabled", exc)
        return None

def choose_pass_side(detection):
    if detection is None:
        return None
    return "RIGHT" if detection.colour == "RED" else "LEFT"

class Camera:
    def __init__(self, device=CAMERA_DEVICE):
        self.device = device
        self.connected = False
        self.running = False
        self.fps = 0.0
        self._lock = threading.Lock()
        self._jpeg = None
        self._detections = []
        self._detector = None

    def start(self):
        if self.running:
            return
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self.running = False
        self.connected = False

    def detections(self):
        with self._lock:
            return list(self._detections)

    def best_pillar(self):
        now = time.time()
        candidates = [
            d for d in self.detections()
            if d.confidence >= PILLAR_MIN_CONFIDENCE
            and now - d.timestamp <= PILLAR_DECISION_TIMEOUT_SEC
            and d.distance_mm <= PILLAR_ENGAGE_MM
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda d: d.distance_mm)

    def jpeg(self):
        with self._lock:
            return self._jpeg

    def state(self):
        best = self.best_pillar()
        return {
            "connected": self.connected,
            "fps": round(self.fps, 1),
            "backend": self._detector.name if self._detector else "none",
            "detections": [d.as_dict() for d in self.detections()],
            "decision": None if best is None else {
                "colour": best.colour,
                "pass_side": choose_pass_side(best),
                "conf": round(best.confidence, 2),
                "distance_mm": round(best.distance_mm) if math.isfinite(best.distance_mm) else None,
            },
        }

    def _open(self):
        target = int(self.device) if str(self.device).isdigit() else self.device
        if isinstance(target, str) and not os.path.exists(target):
            return None
        capture = cv2.VideoCapture(target, cv2.CAP_V4L2)
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not capture.isOpened() or not capture.read()[0]:
            capture.release()
            return None
        logging.info("[camera] opened %dx%d @ %.0f fps (asked for %dx%d)",
                     int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                     capture.get(cv2.CAP_PROP_FPS), FRAME_WIDTH, FRAME_HEIGHT)
        return capture

    def _loop(self):
        self._detector = build_detector()
        capture = None
        frame_index = 0
        latest = []
        frames_this_second, second_started = 0, time.time()

        while self.running:
            if capture is None:
                capture = self._open()
                if capture is None:
                    self.connected = False
                    time.sleep(2.0)
                    continue
                self.connected = True

            ok, frame = capture.read()
            if not ok:
                capture.release()
                capture = None
                self.connected = False
                continue

            frame_index += 1
            if self._detector is not None:
                cheap = isinstance(self._detector, HsvPillarDetector)
                if cheap or frame_index % ONNX_EVERY_N_FRAMES == 0:
                    try:
                        latest = self._detector.detect(frame)
                    except Exception as exc:
                        logging.warning("[vision] detect failed: %s", exc)
                        latest = []

            self._draw_overlay(frame, latest)

            ok, buffer = cv2.imencode(".jpg", frame,
                                      [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if not ok:
                continue

            frames_this_second += 1
            now = time.time()
            with self._lock:
                self._jpeg = buffer.tobytes()
                self._detections = latest
                if now - second_started >= 1.0:
                    self.fps = frames_this_second / (now - second_started)
                    frames_this_second, second_started = 0, now

        if capture is not None:
            capture.release()

    @staticmethod
    def _draw_overlay(frame, detections):
        height, width = frame.shape[:2]
        for fraction, colour in ((PILLAR_TARGET_X_RED, (0, 0, 255)),
                                 (PILLAR_TARGET_X_GREEN, (0, 255, 0))):
            x = int(width * fraction)
            for y in range(0, height, 16):
                cv2.line(frame, (x, y), (x, y + 8), colour, 1)

        for detection in detections:
            x1, y1, x2, y2 = detection.box
            colour = (0, 0, 255) if detection.colour == "RED" else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
            distance = detection.distance_mm
            label = "%s %.2f  %s" % (
                detection.colour, detection.confidence,
                "%d mm" % round(distance) if math.isfinite(distance) else "-")
            cv2.putText(frame, label, (x1, max(12, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)

# ============================================================================
# SECTION 6 - DRIVER (the sensor fusion state machine)
# ============================================================================

STATE_IDLE = "IDLE"
STATE_LANE_FOLLOW = "LANE_FOLLOW"
STATE_PASS_PILLAR = "PASS_PILLAR"
STATE_CORNER = "CORNER"
STATE_REVERSE_ESCAPE = "REVERSE_ESCAPE"

class Driver:
    """Reactive autonomous loop for bench testing."""

    def __init__(self, robot, lidar, camera):
        self.robot = robot
        self.lidar = lidar
        self.camera = camera
        self.active = threading.Event()
        self.state = STATE_IDLE
        self.reason = "stopped"
        self._corner_entered_at = 0.0
        self._corner_direction = 0      # -1 left, +1 right
        self._thread = None

    def start(self):
        if self.active.is_set():
            return
        self.active.set()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.active.clear()
        self.state = STATE_IDLE
        self.reason = "stopped"
        if self.robot is not None:
            self.robot.set_target_steer_deg(0.0)
            self.robot.emergency_stop()

    def status(self):
        return {"active": self.active.is_set(), "state": self.state, "reason": self.reason}

    def _loop(self):
        period = 1.0 / CONTROL_LOOP_HZ
        while self.active.is_set():
            view = self.lidar.view() if self.lidar is not None else LidarView()
            pillar = self.camera.best_pillar() if self.camera is not None else None

            speed, steer = self._decide(view, pillar)
            speed, steer = self._apply_safety(view, speed, steer)

            self.robot.set_target_drive(speed)
            self.robot.set_target_steer_deg(steer)
            time.sleep(period)

        self.robot.set_target_drive(0)
        self.robot.set_target_steer_deg(0.0)

    def _decide(self, view, pillar):
        """Pick a state and return (speed, steer_deg). Priority order matters."""
        front_dist = view.front if view.front is not None else 3000

        # 1. SAFE MODE: Escape Backward if too near to a front object!
        if front_dist < ESCAPE_REVERSE_MM:
            self.state = STATE_REVERSE_ESCAPE
            self.reason = "too near front (%d mm) - reversing safely" % int(front_dist)
            # When backing up, we set steering to 0 to reverse out straight
            return -120, 0.0  

        # 2. CAMERA PRIORITY: If camera sees a pillar, it overrides the walls!
        if pillar is not None and pillar.distance_mm >= PILLAR_RELEASE_MM:
            self.state = STATE_PASS_PILLAR
            self.reason = "%s pillar at %d mm, pass %s" % (
                pillar.colour, round(pillar.distance_mm), choose_pass_side(pillar))
            return DRIVE_PILLAR_SPEED, self._pillar_steer(pillar)

        # 3. Cornering (Front detection at exact parameter setting)
        if self._corner_active(view):
            self.state = STATE_CORNER
            
            # Dynamic snake steering: Angle drops smoothly as wall moves away
            urgency = clamp(1.0 - (front_dist / FRONT_OBSTACLE_MM), 0.0, 1.0)
            dynamic_steer = 10.0 + (urgency * (CORNER_STEER_DEG - 10.0))
            
            return DRIVE_CORNER_SPEED, self._corner_direction * dynamic_steer

        # 4. Nothing from the camera: LiDAR-only corridor centring.
        self.state = STATE_LANE_FOLLOW
        return DRIVE_CRUISE_SPEED, self._centring_steer(view)

    def _pillar_steer(self, pillar):
        fraction = (PILLAR_TARGET_X_RED if pillar.colour == "RED"
                    else PILLAR_TARGET_X_GREEN)
        target_x = FRAME_WIDTH * fraction
        error_px = pillar.cx - target_x
        steer = error_px * PILLAR_STEER_GAIN_DEG_PER_PX
        return clamp(steer, -PILLAR_STEER_MAX_DEG, PILLAR_STEER_MAX_DEG)

    def _centring_steer(self, view):
        if view.left is None and view.right is None:
            self.reason = "no walls in view - holding straight"
            return 0.0
        if view.left is None:
            self.reason = "left wall unknown - hugging the right wall"
            return -CENTRE_STEER_MAX_DEG * 0.4
        if view.right is None:
            self.reason = "right wall unknown - hugging the left wall"
            return +CENTRE_STEER_MAX_DEG * 0.4

        imbalance_mm = view.right - view.left
        if abs(imbalance_mm) < CENTRE_DEADBAND_MM:
            self.reason = "centred (L %d / R %d mm)" % (round(view.left), round(view.right))
            return 0.0
        self.reason = "centring (L %d / R %d mm)" % (round(view.left), round(view.right))
        steer = imbalance_mm * CENTRE_STEER_GAIN_DEG_PER_MM
        return clamp(steer, -CENTRE_STEER_MAX_DEG, CENTRE_STEER_MAX_DEG)

    def _corner_active(self, view):
        front = view.front if view.front is not None else LIDAR_MAX_RANGE_MM
        now = time.time()

        if self.state == STATE_CORNER:
            held_long_enough = now - self._corner_entered_at >= CORNER_MIN_HOLD_SEC
            if held_long_enough and front > CORNER_EXIT_MM:
                return False
            self.reason = "turning %s (front %d mm)" % (
                "right" if self._corner_direction > 0 else "left", round(front))
            return True

        if front > FRONT_OBSTACLE_MM:
            return False

        self._corner_direction = self._corner_turn_direction(view)
        self._corner_entered_at = now
        return True

    @staticmethod
    def _corner_turn_direction(view):
        if TRACK_DIRECTION == "cw":
            return +1
        if TRACK_DIRECTION == "ccw":
            return -1
        left = view.front_left if view.front_left is not None else 0.0
        right = view.front_right if view.front_right is not None else 0.0
        return +1 if right >= left else -1

    def _apply_safety(self, view, speed, steer):
        """Runs after every state. The LiDAR always has the final word."""
        # Check if we are in our new backward safe mode
        is_reversing = speed < 0
        
        # Guard against Side Obstacles (40cm / 400mm threshold)
        if view.right is not None and view.right < SIDE_OBSTACLE_MM:
            steer = max(steer, 0.0) if is_reversing else min(steer, 0.0)
        if view.left is not None and view.left < SIDE_OBSTACLE_MM:
            steer = min(steer, 0.0) if is_reversing else max(steer, 0.0)

        # --- DYNAMIC FRICTION BOOST ---
        # Provide extra push if the wheels are turned and dragging on the ground!
        if speed > 0:
            steer_intensity = abs(steer)
            friction_boost = steer_intensity * 1.5 
            speed = speed + friction_boost
            speed = min(speed, DRIVE_CMD_MAX)
        # -----------------------------------

        # Ensure the motor has enough power whether going forward OR backward
        if 0 < speed < DRIVE_MIN_SPEED:
            speed = DRIVE_MIN_SPEED
        elif -DRIVE_MIN_SPEED < speed < 0:
            speed = -DRIVE_MIN_SPEED

        return int(speed), clamp(steer, -STEER_MAX_LEFT_DEG, STEER_MAX_RIGHT_DEG)

# ============================================================================
# SECTION 7 - WEB DASHBOARD
# ============================================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
logging.getLogger("werkzeug").setLevel(logging.ERROR)
flask.cli.show_server_banner = lambda *a, **k: None

app = Flask(__name__)

robot = None     # RobotLink
lidar = None     # LidarSensor
camera = None    # Camera
driver = None    # Driver
WORKERS = {}     # {"motor": robot, "lidar": lidar, "camera": camera}

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>WRO bench dashboard</title>
<style>
  :root { --ink:#e8e6e1; --dim:#8d8a84; --panel:#1c1e21; --edge:#32363b;
          --ok:#5fbf7a; --bad:#e0705f; --accent:#4fa3ff; }
  * { box-sizing:border-box; }
  body { margin:0; padding:20px; background:#131518; color:var(--ink);
         font-family:"DejaVu Sans Mono", ui-monospace, monospace; font-size:13px; }
  .wrap { max-width:1180px; margin:0 auto; }
  h1 { font-size:15px; letter-spacing:.12em; text-transform:uppercase; margin:0 0 4px; }
  .sub { color:var(--dim); margin:0 0 14px; }
  .row { display:flex; gap:14px; flex-wrap:wrap; }
  .panel { background:var(--panel); border:1px solid var(--edge); border-radius:4px; padding:10px; }
  .video { flex:2 1 460px; }
  .video img { width:100%; display:block; background:#000; }
  .radar { flex:1 1 340px; text-align:center; }
  canvas { max-width:100%; background:#000; }
  button { background:var(--panel); color:var(--ink); border:1px solid var(--edge);
           padding:6px 12px; margin:0 8px 8px 0; cursor:pointer; font-family:inherit; }
  button.on { border-color:var(--ok); color:var(--ok); }
  button.off { border-color:var(--bad); color:var(--bad); }
  .pad { flex:0 0 auto; text-align:center; }
  .pad h3, .cal h3 { margin:0 0 8px; font-size:11px; letter-spacing:.1em;
                     text-transform:uppercase; color:var(--dim); font-weight:normal; }
  .grid { display:grid; grid-template-columns:repeat(3,54px); grid-template-rows:repeat(3,54px); gap:6px; }
  .key { font-size:19px; background:var(--panel); border:1px solid var(--edge); color:var(--ink);
         cursor:pointer; user-select:none; touch-action:none; border-radius:4px; }
  .key:active { background:#2a2d31; }
  .key.stop { color:var(--bad); border-color:var(--bad); font-size:11px; }
  #pad.locked .key { opacity:.3; pointer-events:none; }
  .fields { display:grid; grid-template-columns:repeat(auto-fill,minmax(150px,1fr)); gap:8px; margin-top:14px; }
  .field { border-top:1px solid var(--edge); padding-top:4px; }
  .field .k { display:block; color:var(--dim); font-size:10px; letter-spacing:.08em; text-transform:uppercase; }
  .field .v { font-size:15px; }
  .state { font-size:16px; margin-top:10px; }
  .state b { color:var(--accent); }
  .cal { margin-top:14px; }
  .cal input[type=range] { width:100%; }
  .cal code { display:block; background:#0e1013; border:1px solid var(--edge);
              padding:8px; margin-top:8px; color:var(--ok); white-space:pre; overflow-x:auto; }
  footer { color:var(--dim); margin-top:16px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>WRO bench test &amp; teleop</h1>
  <p class="sub">Practice tool. Rule 11.10 forbids WiFi during a timed round.</p>

  <div>
    <button id="btn-camera" onclick="toggle('camera')">Camera</button>
    <button id="btn-lidar" onclick="toggle('lidar')">LiDAR</button>
    <button id="btn-motor" onclick="toggle('motor')">Motor</button>
  </div>
  <div>
    <button onclick="autoStart()">&#9654; Start autonomous</button>
    <button onclick="autoStop()">&#9632; Stop autonomous</button>
    <span id="mode">manual</span>
  </div>

  <div class="row" style="margin-top:14px;">
    <div class="panel video">
      <img src="/stream" alt="camera feed">
      <div class="state" id="decision">no pillar detected</div>
    </div>

    <div class="panel radar">
      <canvas id="radar" width="340" height="340"></canvas>
      <div class="cal">
        <h3>LiDAR calibration</h3>
        <label>Front angle: <span id="frontDegLabel">0</span>&deg;</label>
        <input type="range" id="frontDeg" min="0" max="359" step="1" value="0"
               oninput="pushCalibration()">
        <button id="cwBtn" onclick="flipDirection()">Angles grow: clockwise</button>
        <code id="calSnippet">LIDAR_FRONT_DEG = 0
LIDAR_ANGLE_CW = True</code>
        <p style="color:var(--dim);margin:8px 0 0;">Put a box 30-50 cm in front of
        the nose, then drag until the white wedge sits on it. Paste the two
        lines into SECTION 1.10.</p>
      </div>
    </div>

    <div class="panel pad" id="pad">
      <h3>Manual drive</h3>
      <div class="grid">
        <div></div><button class="key" data-key="w">&#9650;</button><div></div>
        <button class="key" data-key="a">&#9664;</button>
        <button class="key stop" id="stopBtn">STOP</button>
        <button class="key" data-key="d">&#9654;</button>
        <div></div><button class="key" data-key="s">&#9660;</button><div></div>
      </div>
    </div>
  </div>

  <div class="panel fields">
    <div class="field"><span class="k">State</span><span class="v" id="state">-</span></div>
    <div class="field"><span class="k">Front</span><span class="v" id="sFront">-</span></div>
    <div class="field"><span class="k">Front left</span><span class="v" id="sFrontLeft">-</span></div>
    <div class="field"><span class="k">Front right</span><span class="v" id="sFrontRight">-</span></div>
    <div class="field"><span class="k">Left</span><span class="v" id="sLeft">-</span></div>
    <div class="field"><span class="k">Right</span><span class="v" id="sRight">-</span></div>
    <div class="field"><span class="k">Linear vel</span><span class="v" id="linVel">-</span></div>
    <div class="field"><span class="k">Angular vel</span><span class="v" id="angVel">-</span></div>
    <div class="field"><span class="k">Encoder</span><span class="v" id="ticks">-</span></div>
    <div class="field"><span class="k">Throttle</span><span class="v" id="throttle">-</span></div>
    <div class="field"><span class="k">Steer</span><span class="v" id="steer">-</span></div>
    <div class="field"><span class="k">Motor link</span><span class="v" id="motorStatus">-</span></div>
    <div class="field"><span class="k">Camera</span><span class="v" id="cameraStatus">-</span></div>
    <div class="field"><span class="k">Detector</span><span class="v" id="backend">-</span></div>
    <div class="field"><span class="k">LiDAR</span><span class="v" id="lidarStatus">-</span></div>
    <div class="field"><span class="k">Sent</span><span class="v" id="sent">-</span></div>
  </div>

  <footer id="why">Drive with the pad, or W/S/A/D on a keyboard. Space centres
  the steering. Autonomous mode locks the pad.</footer>
</div>

<script>
// ---- manual drive -------------------------------------------------------
const cmd = {linear:0, angular:0};
const held = new Set();
let autonomous = false;
let calibration = {front_deg:0, clockwise:true, front_width:30};

addEventListener('keydown', e => {
  const k = e.key.toLowerCase();
  if (['w','a','s','d',' '].includes(k)) e.preventDefault();
  if (k === ' ') { held.delete('a'); held.delete('d'); } else held.add(k);
});
addEventListener('keyup', e => held.delete(e.key.toLowerCase()));

document.querySelectorAll('.key[data-key]').forEach(btn => {
  const k = btn.dataset.key;
  const down = e => { e.preventDefault(); held.add(k); };
  const up = e => { e.preventDefault(); held.delete(k); };
  btn.addEventListener('pointerdown', down);
  btn.addEventListener('pointerup', up);
  btn.addEventListener('pointerleave', up);
  btn.addEventListener('pointercancel', up);
});
document.getElementById('stopBtn').addEventListener('pointerdown', e => {
  e.preventDefault(); held.clear(); send();
});

async function send() {
  cmd.linear  = held.has('w') ? 1 : (held.has('s') ? -1 : 0);
  cmd.angular = held.has('d') ? 1 : (held.has('a') ? -1 : 0);
  document.getElementById('sent').textContent =
    'lin ' + cmd.linear.toFixed(1) + '  ang ' + cmd.angular.toFixed(1);
  if (autonomous) return;
  try {
    await fetch('/cmd', {method:'POST', headers:{'Content-Type':'application/json'},
                         body:JSON.stringify(cmd)});
  } catch (e) {}
}
setInterval(send, 100);

// ---- buttons ------------------------------------------------------------
async function toggle(name) { await fetch('/toggle/' + name, {method:'POST'}); }
async function autoStart() {
  const r = await fetch('/autonomous/start', {method:'POST'});
  const d = await r.json();
  if (!d.ok) alert(d.error || 'could not start autonomous mode');
}
async function autoStop() { await fetch('/autonomous/stop', {method:'POST'}); }

// ---- live LiDAR calibration --------------------------------------------
function flipDirection() {
  calibration.clockwise = !calibration.clockwise;
  pushCalibration();
}
async function pushCalibration() {
  calibration.front_deg = parseInt(document.getElementById('frontDeg').value, 10);
  document.getElementById('frontDegLabel').textContent = calibration.front_deg;
  document.getElementById('cwBtn').textContent =
    'Angles grow: ' + (calibration.clockwise ? 'clockwise' : 'counter-clockwise');
  document.getElementById('calSnippet').textContent =
    'LIDAR_FRONT_DEG = ' + calibration.front_deg + '\\n' +
    'LIDAR_ANGLE_CW = ' + (calibration.clockwise ? 'True' : 'False');
  try {
    await fetch('/calibration', {method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({front_deg:calibration.front_deg, clockwise:calibration.clockwise})});
  } catch (e) {}
}

// ---- telemetry ----------------------------------------------------------
const mm = v => (v === null || v === undefined) ? '--' : v + ' mm';

async function poll() {
  try {
    const d = await (await fetch('/telemetry', {cache:'no-store'})).json();

    document.getElementById('linVel').textContent = d.robot.linear_vel_mmps + ' mm/s';
    document.getElementById('angVel').textContent = d.robot.angular_vel_radps + ' rad/s';
    document.getElementById('ticks').textContent = d.robot.encoder_ticks;
    document.getElementById('throttle').textContent = d.robot.throttle_cmd;
    document.getElementById('steer').textContent =
      d.robot.steer_cmd + '  (' + d.robot.steer_deg + '\\u00b0)';

    setBtn('btn-motor', d.robot.connected);
    setBtn('btn-camera', d.camera.connected);
    setBtn('btn-lidar', d.lidar.connected);

    document.getElementById('motorStatus').textContent = d.robot.connected ? 'connected' : 'offline';
    document.getElementById('cameraStatus').textContent =
      d.camera.connected ? d.camera.fps + ' fps' : 'offline';
    document.getElementById('backend').textContent = d.camera.backend;
    document.getElementById('lidarStatus').textContent =
      d.lidar.connected ? d.lidar.sectors.point_count + ' pts' : 'offline';

    const s = d.lidar.sectors;
    document.getElementById('sFront').textContent = mm(s.front);
    document.getElementById('sFrontLeft').textContent = mm(s.front_left);
    document.getElementById('sFrontRight').textContent = mm(s.front_right);
    document.getElementById('sLeft').textContent = mm(s.left);
    document.getElementById('sRight').textContent = mm(s.right);

    const dec = d.camera.decision;
    document.getElementById('decision').innerHTML = dec
      ? '<b>' + dec.colour + '</b> pillar at ' + dec.distance_mm + ' mm &rarr; pass '
        + dec.pass_side + ' (conf ' + dec.conf + ')'
      : 'no pillar detected &mdash; LiDAR is steering';

    autonomous = d.driver.active;
    document.getElementById('mode').textContent = autonomous ? 'AUTONOMOUS' : 'manual';
    document.getElementById('state').textContent = d.driver.state;
    document.getElementById('why').textContent = d.driver.reason;
    document.getElementById('pad').classList.toggle('locked', autonomous);
  } catch (e) {}
}
function setBtn(id, on) {
  const el = document.getElementById(id);
  el.classList.toggle('on', on);
  el.classList.toggle('off', !on);
}
setInterval(poll, __TELEMETRY_MS__);
poll();

// ---- radar --------------------------------------------------------------
function drawRadar(points, frontWidth) {
  const c = document.getElementById('radar'), ctx = c.getContext('2d');
  const W = c.width, H = c.height, cx = W/2, cy = H/2;
  const maxRange = 3000, scale = Math.min(W,H)/2/maxRange;

  ctx.fillStyle = '#000'; ctx.fillRect(0,0,W,H);

  ctx.strokeStyle = '#32363b'; ctx.fillStyle = '#8d8a84'; ctx.font = '10px monospace';
  for (let r = 500; r <= maxRange; r += 500) {
    ctx.beginPath(); ctx.arc(cx, cy, r*scale, 0, Math.PI*2); ctx.stroke();
    ctx.fillText(r + 'mm', cx + 4, cy - r*scale);
  }

  const half = frontWidth/2, reach = maxRange*scale;
  ctx.strokeStyle = '#ffffff'; ctx.setLineDash([4,4]);
  [-half, half].forEach(a => {
    const r = a*Math.PI/180;
    ctx.beginPath(); ctx.moveTo(cx, cy);
    ctx.lineTo(cx + reach*Math.sin(r), cy - reach*Math.cos(r)); ctx.stroke();
  });
  ctx.setLineDash([]);

  ctx.fillStyle = '#4fa3ff';
  ctx.beginPath(); ctx.arc(cx, cy, 4, 0, Math.PI*2); ctx.fill();

  for (const [angle, dist] of points) {
    const r = angle*Math.PI/180;
    const x = cx + dist*scale*Math.sin(r);
    const y = cy - dist*scale*Math.cos(r);
    ctx.fillStyle = Math.abs(angle) <= half ? '#ffffff' : '#e0705f';
    if (x >= 0 && x <= W && y >= 0 && y <= H) ctx.fillRect(x-1, y-1, 2, 2);
  }
}
async function pollRadar() {
  try {
    const d = await (await fetch('/lidar', {cache:'no-store'})).json();
    drawRadar(d.points, d.front_width_deg);
  } catch (e) {}
}
setInterval(pollRadar, __RADAR_MS__);

fetch('/calibration').then(r => r.json()).then(d => {
  calibration.clockwise = d.clockwise;
  document.getElementById('frontDeg').value = d.front_deg;
  pushCalibration();
  pollRadar();
});
</script>
</body>
</html>"""

@app.route("/")
def index():
    page = (PAGE
            .replace("__TELEMETRY_MS__", str(TELEMETRY_POLL_MS))
            .replace("__RADAR_MS__", str(RADAR_POLL_MS)))
    return render_template_string(page)

@app.route("/stream")
def stream():
    def frames():
        while True:
            if camera is None or not camera.connected:
                time.sleep(0.25)
                continue
            jpeg = camera.jpeg()
            if jpeg is None:
                time.sleep(0.05)
                continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            time.sleep(1 / 30.0)
    return Response(frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/lidar")
def lidar_points():
    points = lidar.points() if lidar is not None else []
    return jsonify({
        "points": [[round(a, 1), round(d)] for a, d in points],
        "front_width_deg": SECTOR_FRONT_WIDTH_DEG,
    })

@app.route("/telemetry")
def telemetry():
    empty_camera = {"connected": False, "fps": 0, "backend": "none",
                    "detections": [], "decision": None}
    return jsonify({
        "robot": robot.state() if robot is not None else {},
        "camera": camera.state() if camera is not None else empty_camera,
        "lidar": {
            "connected": lidar.connected if lidar is not None else False,
            "sectors": (lidar.view() if lidar is not None else LidarView()).as_dict(),
        },
        "driver": driver.status() if driver is not None else
                  {"active": False, "state": STATE_IDLE, "reason": "not started"},
    })

@app.route("/calibration", methods=["GET", "POST"])
def calibration():
    global LIDAR_FRONT_DEG, LIDAR_ANGLE_CW
    if request.method == "POST":
        data = request.get_json(force=True, silent=True) or {}
        if "front_deg" in data:
            LIDAR_FRONT_DEG = int(data["front_deg"]) % 360
        if "clockwise" in data:
            LIDAR_ANGLE_CW = bool(data["clockwise"])
    return jsonify({"front_deg": LIDAR_FRONT_DEG, "clockwise": LIDAR_ANGLE_CW})

@app.route("/cmd", methods=["POST"])
def manual_command():
    if driver is not None and driver.active.is_set():
        return jsonify({"ok": False, "error": "autonomous mode owns the motors"}), 409
    data = request.get_json(force=True, silent=True) or {}
    linear = clamp(float(data.get("linear", 0.0)), -1.0, 1.0)
    angular = clamp(float(data.get("angular", 0.0)), -1.0, 1.0)
    if robot is not None and robot.connected:
        robot.set_target_drive(int(linear * DRIVE_CMD_MAX))
        limit = STEER_MAX_RIGHT_DEG if angular >= 0 else STEER_MAX_LEFT_DEG
        robot.set_target_steer_deg(angular * limit)
    return jsonify({"ok": True})

@app.route("/autonomous/start", methods=["POST"])
def autonomous_start():
    if robot is None or not robot.connected:
        return jsonify({"ok": False, "error": "motor link not connected"}), 400
    driver.start()
    return jsonify({"ok": True})

@app.route("/autonomous/stop", methods=["POST"])
def autonomous_stop():
    if driver is not None:
        driver.stop()
    return jsonify({"ok": True})

@app.route("/toggle/<name>", methods=["POST"])
def toggle_worker(name):
    worker = WORKERS.get(name)
    if worker is None:
        return jsonify({"ok": False, "error": "unknown sensor"}), 404
    if getattr(worker, "running", False):
        worker.stop()
    else:
        worker.start()
    return jsonify({"ok": True, "running": getattr(worker, "running", False)})

# ============================================================================
# SECTION 8 - MAIN, CLI AND THE LIDAR CALIBRATION WIZARD
# ============================================================================

def calibrate_lidar(port, baud, samples=40):
    sensor = LidarSensor(port, baud)
    sensor.start()

    print("\nWaiting for the LiDAR to spin up...")
    deadline = time.time() + 20
    while not sensor.connected and time.time() < deadline:
        time.sleep(0.5)
    if not sensor.connected:
        print("LiDAR did not start. Check the port and the power.")
        sensor.stop()
        return

    def nearest_raw_angle(prompt):
        input(prompt)
        readings = []
        for _ in range(samples):
            points = sensor.raw_points()
            if points:
                readings.append(min(points, key=lambda p: p[1]))
            time.sleep(0.05)
        if not readings:
            return None, None
        angle, distance = min(readings, key=lambda p: p[1])
        print("  nearest return: %.1f deg at %d mm" % (angle, distance))
        return angle, distance

    print("\nStep 1 of 2")
    print("Put a flat box 30-50 cm DIRECTLY IN FRONT of the nose.")
    print("Clear everything else within 1.5 m.")
    front_deg, _ = nearest_raw_angle("Press Enter when the box is in place... ")
    if front_deg is None:
        print("No returns. Move the box closer and try again.")
        sensor.stop()
        return

    print("\nStep 2 of 2")
    print("Move the SAME box to the RIGHT-HAND SIDE of the car, 30-50 cm out.")
    right_deg, _ = nearest_raw_angle("Press Enter when the box is in place... ")
    sensor.stop()
    if right_deg is None:
        print("No returns on the right. Try again.")
        return

    clockwise = 0 < (right_deg - front_deg) % 360 < 180

    print("\n" + "=" * 58)
    print("Paste these two lines into SECTION 1.10:\n")
    print("LIDAR_FRONT_DEG = %d" % (round(front_deg) % 360))
    print("LIDAR_ANGLE_CW = %s" % clockwise)
    print("=" * 58)
    print("\nThen restart the dashboard and check that the white wedge on the")
    print("radar plot lands on an object you place in front of the car.\n")

def main():
    global robot, lidar, camera, driver

    parser = argparse.ArgumentParser(description="WRO bench test & teleop dashboard")
    parser.add_argument("--serial-port", default=SERIAL_PORT)
    parser.add_argument("--lidar-port", default=LIDAR_PORT)
    parser.add_argument("--camera", default=CAMERA_DEVICE)
    parser.add_argument("--web-port", type=int, default=WEB_PORT)
    parser.add_argument("--no-motor", action="store_true", help="do not open the ESP32 port")
    parser.add_argument("--no-lidar", action="store_true", help="do not start the LiDAR")
    parser.add_argument("--no-camera", action="store_true", help="do not start the camera")
    parser.add_argument("--calibrate-lidar", action="store_true",
                        help="run the LIDAR_FRONT_DEG wizard and exit")
    args = parser.parse_args()

    if args.calibrate_lidar:
        calibrate_lidar(args.lidar_port, LIDAR_BAUD)
        return

    robot = RobotLink(args.serial_port, SERIAL_BAUD)
    lidar = LidarSensor(args.lidar_port, LIDAR_BAUD)
    camera = Camera(args.camera)
    driver = Driver(robot, lidar, camera)
    WORKERS.update({"motor": robot, "lidar": lidar, "camera": camera})

    if not args.no_motor:
        robot.start()
    if not args.no_lidar:
        lidar.start()
    if not args.no_camera:
        camera.start()

    print("Dashboard on http://<pi-ip>:%d   (practice tool - not for a timed run)"
          % args.web_port)
    try:
        app.run(host=WEB_HOST, port=args.web_port, threaded=True, debug=False)
    except KeyboardInterrupt:
        pass
    finally:
        driver.stop()
        robot.stop()
        lidar.stop()
        camera.stop()

if __name__ == "__main__":
    main()