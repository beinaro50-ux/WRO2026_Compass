#!/usr/bin/env python3

import os

# Must be set before cv2 is imported, or OpenCV prints its own V4L2 warnings.
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
os.environ.setdefault("OPENCV_VIDEOIO_DEBUG", "0")

import logging
import sys
import threading
import time

import cv2
import numpy as np
import flask.cli
from flask import Flask, Response, jsonify, render_template_string

# ---------------------------------------------------------------------------
DEVICE = sys.argv[1] if len(sys.argv) > 1 else "/dev/webcam"
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
JPEG_QUALITY = 70          # lower = less bandwidth, blurrier
DEFAULT_BOX = 100
FAILS_BEFORE_RECONNECT = 5     # consecutive bad reads that count as "lost"
RETRY_DELAY = 2.0              # seconds between reconnection attempts
# ---------------------------------------------------------------------------

# Only our own messages and real errors reach the terminal. The Flask access
# log and startup banner are silenced.
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
logging.getLogger("werkzeug").setLevel(logging.ERROR)
flask.cli.show_server_banner = lambda *a, **k: None
log = logging.getLogger("camera")

NAMED_COLORS = {
    "Black": (0, 0, 0), "White": (255, 255, 255), "Gray": (128, 128, 128),
    "Light Gray": (211, 211, 211), "Dark Gray": (64, 64, 64), "Silver": (192, 192, 192),
    "Red": (255, 0, 0), "Dark Red": (139, 0, 0), "Crimson": (220, 20, 60),
    "Maroon": (128, 0, 0), "Firebrick": (178, 34, 34), "Tomato": (255, 99, 71),
    "Orange Red": (255, 69, 0), "Orange": (255, 165, 0), "Dark Orange": (255, 140, 0),
    "Gold": (255, 215, 0), "Yellow": (255, 255, 0), "Khaki": (240, 230, 140),
    "Olive": (128, 128, 0), "Yellow Green": (154, 205, 50), "Green": (0, 128, 0),
    "Lime": (0, 255, 0), "Dark Green": (0, 100, 0), "Forest Green": (34, 139, 34),
    "Sea Green": (46, 139, 87), "Spring Green": (0, 255, 127), "Teal": (0, 128, 128),
    "Turquoise": (64, 224, 208), "Cyan": (0, 255, 255), "Light Blue": (173, 216, 230),
    "Sky Blue": (135, 206, 235), "Steel Blue": (70, 130, 180), "Dodger Blue": (30, 144, 255),
    "Blue": (0, 0, 255), "Navy": (0, 0, 128), "Royal Blue": (65, 105, 225),
    "Indigo": (75, 0, 130), "Purple": (128, 0, 128), "Violet": (238, 130, 238),
    "Magenta": (255, 0, 255), "Orchid": (218, 112, 214), "Pink": (255, 192, 203),
    "Hot Pink": (255, 105, 180), "Deep Pink": (255, 20, 147), "Salmon": (250, 128, 114),
    "Coral": (255, 127, 80), "Brown": (165, 42, 42), "Chocolate": (210, 105, 30),
    "Saddle Brown": (139, 69, 19), "Sienna": (160, 82, 45), "Tan": (210, 180, 140),
    "Beige": (245, 245, 220), "Ivory": (255, 255, 240), "Wheat": (245, 222, 179),
}

# Precomputed arrays so matching is one vectorised operation per frame.
_NAMES = list(NAMED_COLORS.keys())
_VALUES = np.array([NAMED_COLORS[n] for n in _NAMES], dtype=np.int32)


def closest_color_name(rgb):
    """Return the nearest named colour to the given (R, G, B) tuple."""
    diff = _VALUES - np.array(rgb, dtype=np.int32)
    return _NAMES[int(np.argmin((diff * diff).sum(axis=1)))]


class CameraWorker:
    """Grabs frames in a background thread, reopening the device whenever it
    disappears. The HTTP handlers never block on the camera, and the server
    keeps running through a disconnect."""

    def __init__(self, device):
        self.device = device
        self.box_size = DEFAULT_BOX
        self.lock = threading.Lock()
        self.jpeg = None
        self.name = "-"
        self.rgb = (0, 0, 0)
        self.fps = 0.0
        self.connected = False
        self.status = "starting up"
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    # -- state shared with the web handlers --------------------------------
    def _set(self, **kw):
        with self.lock:
            for k, v in kw.items():
                setattr(self, k, v)

    def state(self):
        with self.lock:
            return {
                "name": self.name,
                "rgb": list(self.rgb),
                "hex": "#%02x%02x%02x" % self.rgb,
                "fps": round(self.fps, 1),
                "box": self.box_size,
                "connected": self.connected,
                "status": self.status,
            }

    def snapshot(self):
        with self.lock:
            return self.jpeg

    # -- capture ----------------------------------------------------------
    def _open(self):
        """Return an opened VideoCapture, or None if the device isn't there."""
        target = int(self.device) if str(self.device).isdigit() else self.device
        if isinstance(target, str) and not os.path.exists(target):
            return None

        cap = cv2.VideoCapture(target, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            cap.release()
            return None

        # An opened-but-dead node still reads False, so confirm one real frame
        # before declaring the camera healthy.
        ok, _ = cap.read()
        if not ok:
            cap.release()
            return None
        return cap

    def _loop(self):
        cap = None
        fails = 0
        announced_missing = False
        last, frames = time.time(), 0

        while self.running:
            # --- (re)connect ---
            if cap is None:
                cap = self._open()
                if cap is None:
                    if not announced_missing:
                        log.error("camera unavailable at %s - retrying every %.0fs",
                                  self.device, RETRY_DELAY)
                        announced_missing = True
                    self._set(connected=False, status="camera disconnected")
                    time.sleep(RETRY_DELAY)
                    continue
                log.info("camera connected on %s", self.device)
                announced_missing = False
                fails = 0
                self._set(connected=True, status="streaming")

            # --- grab ---
            ok, frame = cap.read()
            if not ok:
                fails += 1
                if fails >= FAILS_BEFORE_RECONNECT:
                    log.error("camera lost on %s - reconnecting", self.device)
                    cap.release()
                    cap = None
                    fails = 0
                    self._set(connected=False, status="reconnecting")
                    time.sleep(RETRY_DELAY)
                else:
                    time.sleep(0.05)
                continue
            fails = 0

            # --- sample the centre box ---
            h, w = frame.shape[:2]
            half = self.box_size // 2
            cx, cy = w // 2, h // 2
            x1, y1 = max(0, cx - half), max(0, cy - half)
            x2, y2 = min(w, cx + half), min(h, cy + half)

            b, g, r = frame[y1:y2, x1:x2].reshape(-1, 3).mean(axis=0)
            rgb = (int(r), int(g), int(b))
            name = closest_color_name(rgb)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

            enc_ok, buf = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if not enc_ok:
                continue

            frames += 1
            now = time.time()
            fps = None
            if now - last >= 1.0:
                fps = frames / (now - last)
                last, frames = now, 0

            with self.lock:
                self.jpeg = buf.tobytes()
                self.name = name
                self.rgb = rgb
                self.connected = True
                self.status = "streaming"
                if fps is not None:
                    self.fps = fps

        if cap is not None:
            cap.release()

    def release(self):
        self.running = False
        time.sleep(0.3)


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Colour reader</title>
<style>
  :root {
    --ink: #e8e6e1;
    --dim: #8d8a84;
    --panel: #1c1e21;
    --edge: #32363b;
    --warn: #e0705f;
    --sample: #666;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    padding: 24px 20px 40px;
    background: #131518;
    color: var(--ink);
    font-family: "DejaVu Sans Mono", ui-monospace, monospace;
    font-size: 14px;
  }
  .wrap { max-width: 700px; margin: 0 auto; }
  header {
    display: flex; align-items: baseline; gap: 12px;
    padding-bottom: 14px; border-bottom: 1px solid var(--edge);
  }
  h1 {
    margin: 0; font-size: 15px; font-weight: 700;
    letter-spacing: 0.14em; text-transform: uppercase;
  }
  .device { color: var(--dim); font-size: 12px; margin-left: auto; }

  /* The signature: the sample colour drives the frame around the feed,
     so the page itself takes on whatever the camera is looking at. */
  .stage {
    position: relative;
    margin: 20px 0;
    padding: 8px;
    background: var(--panel);
    border: 2px solid var(--sample);
    transition: border-color 200ms linear;
  }
  .stage img { display: block; width: 100%; height: auto; }
  .stage.offline { border-color: var(--warn); }
  .stage.offline img { opacity: 0.25; filter: grayscale(1); }
  .veil {
    display: none;
    position: absolute; inset: 0;
    align-items: center; justify-content: center;
    color: var(--warn); letter-spacing: 0.1em; text-transform: uppercase;
    font-size: 12px; text-align: center; padding: 20px;
  }
  .stage.offline .veil { display: flex; }

  .readout { display: flex; align-items: stretch; gap: 14px; }
  .swatch {
    width: 92px; flex: none;
    background: var(--sample);
    border: 1px solid var(--edge);
    transition: background-color 200ms linear;
  }
  .fields { flex: 1; display: grid; grid-template-columns: 1fr 1fr; gap: 10px 14px; }
  .field { border-top: 1px solid var(--edge); padding-top: 6px; }
  .field .k {
    display: block; color: var(--dim); font-size: 11px;
    letter-spacing: 0.1em; text-transform: uppercase;
  }
  .field .v { font-size: 18px; }
  #name { font-size: 22px; }
  .status { margin-top: 18px; color: var(--dim); font-size: 12px; min-height: 1.2em; }
  .status.bad { color: var(--warn); }
  @media (max-width: 480px) {
    .readout { flex-direction: column; }
    .swatch { width: 100%; height: 56px; }
  }
  @media (prefers-reduced-motion: reduce) {
    .stage, .swatch { transition: none; }
  }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>Colour reader</h1>
    <span class="device">{{ device }}</span>
  </header>

  <div class="stage" id="stage">
    <img src="/stream" alt="Camera feed" id="feed">
    <div class="veil" id="veil">Camera disconnected</div>
  </div>

  <div class="readout">
    <div class="swatch" id="swatch"></div>
    <div class="fields">
      <div class="field"><span class="k">Colour</span><span class="v" id="name">-</span></div>
      <div class="field"><span class="k">Hex</span><span class="v" id="hex">-</span></div>
      <div class="field"><span class="k">RGB</span><span class="v" id="rgb">-</span></div>
      <div class="field"><span class="k">Capture</span><span class="v" id="fps">-</span></div>
    </div>
  </div>

  <p class="status" id="status">Point the camera at an object.</p>
</div>

<script>
  const stage  = document.getElementById('stage');
  const swatch = document.getElementById('swatch');
  const status = document.getElementById('status');
  const veil   = document.getElementById('veil');
  const feed   = document.getElementById('feed');
  let wasOffline = false;

  async function poll() {
    try {
      const r = await fetch('/color', {cache: 'no-store'});
      const d = await r.json();
      document.getElementById('name').textContent = d.name;
      document.getElementById('hex').textContent  = d.hex;
      document.getElementById('rgb').textContent  = d.rgb.join(', ');
      document.getElementById('fps').textContent  = d.fps + ' fps';
      stage.style.setProperty('--sample', d.hex);
      swatch.style.setProperty('--sample', d.hex);

      if (d.connected) {
        stage.classList.remove('offline');
        status.textContent = 'Sampling a ' + d.box + ' px box at the centre.';
        status.className = 'status';
        // The MJPEG connection dies with the camera, so restart it once
        // the device comes back.
        if (wasOffline) {
          feed.src = '/stream?t=' + Date.now();
          wasOffline = false;
        }
      } else {
        stage.classList.add('offline');
        veil.textContent = d.status;
        status.textContent = d.status + ' - retrying automatically.';
        status.className = 'status bad';
        wasOffline = true;
      }
    } catch (e) {
      stage.classList.add('offline');
      veil.textContent = 'No server';
      status.textContent = 'Lost contact with the camera server.';
      status.className = 'status bad';
      wasOffline = true;
    }
  }
  poll();
  setInterval(poll, 500);
</script>
</body>
</html>"""

app = Flask(__name__)
camera = None


@app.route("/")
def index():
    return render_template_string(PAGE, device=DEVICE)


@app.route("/color")
def color():
    return jsonify(camera.state())


@app.route("/stream")
def stream():
    def frames():
        while True:
            if not camera.connected:
                time.sleep(0.25)
                continue
            jpeg = camera.snapshot()
            if jpeg is None:
                time.sleep(0.05)
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n"
                   b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                   + jpeg + b"\r\n")
            time.sleep(1 / 30)

    return Response(frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/snapshot")
def snapshot():
    jpeg = camera.snapshot()
    if jpeg is None:
        return "No frame yet", 503
    return Response(jpeg, mimetype="image/jpeg")


if __name__ == "__main__":
    camera = CameraWorker(DEVICE)
    try:
        log.info("serving %s on http://0.0.0.0:5000", DEVICE)
        app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)
    except KeyboardInterrupt:
        pass
    finally:
        camera.release()
        log.info("stopped")
