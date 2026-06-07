# minos_sigint_ufo_scanner_advanced_lock.py
# Android/Pydroid 3 friendly Kivy + OpenCV SIGINT / UFO micro-motion scanner.
#
# Advanced lock version:
# - Reticle capture selects the target
# - YOLO ONNX snap-tags ONCE after capture, then idles during tracking
# - Stabilized lock uses:
#     1) Micro-motion reacquire near last target
#     2) Optical-flow point tracking around the lock
#     3) Kalman prediction / smoothing
#     4) Target memory + temporary search hold
# - Stabilized inspector follows predicted target, not raw jitter
# - Android-safe digital zoom
# - Manual focus slider when supported
# - Exposure lock / white-balance lock attempts when supported
# - Compact controls to hide sliders
#
# Expected YOLO model path:
#   /storage/emulated/0/Python Projects/models/yolov8n.onnx
#
# Run:
#   python minos_sigint_ufo_scanner_advanced_lock.py

import os
import time
import math
import random
import threading
from collections import deque

import cv2
import numpy as np

from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.graphics.texture import Texture
from kivy.metrics import dp
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.slider import Slider
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout


def request_android_permissions():
    try:
        from android.permissions import request_permissions, Permission
        request_permissions([
            Permission.CAMERA,
            Permission.READ_EXTERNAL_STORAGE,
            Permission.WRITE_EXTERNAL_STORAGE,
        ])
    except Exception:
        pass


GREEN = (0, 255, 110)
AMBER = (255, 180, 40)
RED = (255, 50, 50)
CYAN = (40, 220, 255)
WHITE = (230, 255, 235)

# Resolve model path: bundled inside the APK takes priority,
# falling back to the old Pydroid 3 location for manual installs.
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_BUNDLED_MODEL = os.path.join(_APP_DIR, "models", "yolov8n.onnx")
_PYDROID_MODEL = "/storage/emulated/0/Python Projects/models/yolov8n.onnx"
YOLO_ONNX_PATH = _BUNDLED_MODEL if os.path.exists(_BUNDLED_MODEL) else _PYDROID_MODEL

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]


class MinosSigintUFO(FloatLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.capture = None
        self.running = True
        self.frame = None
        self.frame_lock = threading.Lock()
        self.camera_index = 0

        self.render_mode = "NORMAL"
        self.overlay_color = GREEN
        self.overlay_i = 0
        self.scanline_enabled = True
        self.mirror = False

        self.camera_zoom = 1.0
        self.camera_zoom_max = 6.0

        # Camera controls
        self.manual_focus_enabled = False
        self.focus_value = 0
        self.focus_status = "FOCUS: AUTO"
        self.exposure_locked = False
        self.white_balance_locked = False
        self.camera_status = "CAM CTRL: AUTO"

        # Micro motion
        self.ufo_scan_enabled = True
        self.motion_min_area = 1
        self.motion_threshold = 6
        self.motion_blur = 1
        self.motion_max_points = 220
        self.motion_decay_seconds = 1.20
        self.prev_motion_gray = None
        self.motion_history = deque(maxlen=900)
        self.last_motion_count = 0
        self.last_motion_area = 0
        self.last_points = []
        self.fps_clock = time.time()
        self.fps_frames = 0
        self.fps = 0.0

        # Advanced lock
        self.capture_requested = False
        self.target_locked = False
        self.lock_target = None
        self.lock_last_seen = 0.0
        self.lock_lost_seconds = 2.2
        self.lock_search_radius = 72
        self.capture_radius = 92
        self.lock_box_color = CYAN
        self.lock_status = "RETICLE READY"
        self.track_confidence = 0.0

        # Stabilization / prediction
        self.kalman = self.create_kalman()
        self.kalman_ready = False
        self.pred_x = 0.0
        self.pred_y = 0.0
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.prev_track_gray = None
        self.flow_points = None
        self.flow_enabled = True
        self.flow_radius = 28
        self.target_memory = {
            "color": None,
            "brightness": 0.0,
            "area": 1,
            "last_box": None,
        }

        # YOLO snap tag
        self.yolo_enabled = True
        self.yolo_net = None
        self.yolo_loaded = False
        self.yolo_status = "YOLO: LOADING"
        self.yolo_input_size = 640
        self.yolo_conf_threshold = 0.22
        self.yolo_nms_threshold = 0.45
        self.yolo_crop_pad = 128
        self.snap_tag_pending = False
        self.snap_tag_done = False
        self.snap_label = "UNSCANNED"
        self.snap_conf = 0.0
        self.snap_box = None
        self.snap_crop_rect = None
        self.load_yolo()

        # Inspector / radar
        self.inspector_enabled = True
        self.inspector_crop_half = 16
        self.radar_enabled = True
        self.radar_sweep_angle = 0.0
        self.radar_blips = deque(maxlen=450)
        self.radar_decay_seconds = 2.4
        self.radar_radius = 58
        self.radar_margin = 16

        self.controls_expanded = True

        self.image = Image(size_hint=(1, 1), allow_stretch=True, keep_ratio=True)
        self.add_widget(self.image)

        self.status = Label(
            text="MINOS SIGINT // ADVANCED TARGET LOCK",
            size_hint=(1, None),
            height=dp(26),
            pos_hint={"x": 0, "top": 1},
            color=(0, 1, 0.42, 1),
            font_size=dp(13),
            bold=True,
        )
        self.add_widget(self.status)

        self.controls = BoxLayout(
            orientation="vertical",
            size_hint=(1, None),
            height=dp(352),
            pos_hint={"x": 0, "y": 0},
            spacing=dp(2),
            padding=[dp(4), dp(2), dp(4), dp(2)],
        )
        self.add_widget(self.controls)

        row = GridLayout(cols=8, size_hint=(1, None), height=dp(42), spacing=dp(3))
        self.controls.add_widget(row)

        self.btn_capture = Button(text="CAPTURE\nRETICLE", font_size=dp(9), background_color=(0.0, 0.35, 0.16, 1))
        self.btn_capture.bind(on_press=self.request_reticle_capture)
        row.add_widget(self.btn_capture)

        self.btn_unlock = Button(text="UNLOCK\nTARGET", font_size=dp(9))
        self.btn_unlock.bind(on_press=self.unlock_target)
        row.add_widget(self.btn_unlock)

        self.btn_yolo = ToggleButton(text="SNAP\nYOLO", state="down", font_size=dp(9))
        self.btn_yolo.bind(on_press=self.toggle_yolo)
        row.add_widget(self.btn_yolo)

        self.btn_flow = ToggleButton(text="FLOW\nON", state="down", font_size=dp(9))
        self.btn_flow.bind(on_press=self.toggle_flow)
        row.add_widget(self.btn_flow)

        self.btn_render = Button(text="MODE\nNORMAL", font_size=dp(9))
        self.btn_render.bind(on_press=self.cycle_render)
        row.add_widget(self.btn_render)

        self.btn_radar = ToggleButton(text="RADAR\nON", state="down", font_size=dp(9))
        self.btn_radar.bind(on_press=self.toggle_radar)
        row.add_widget(self.btn_radar)

        self.btn_focus = ToggleButton(text="FOCUS\nAUTO", state="normal", font_size=dp(9))
        self.btn_focus.bind(on_press=self.toggle_manual_focus)
        row.add_widget(self.btn_focus)

        self.btn_compact = ToggleButton(text="SLIDERS\nHIDE", state="normal", font_size=dp(9))
        self.btn_compact.bind(on_press=self.toggle_controls)
        row.add_widget(self.btn_compact)

        self.compact_row = GridLayout(cols=5, size_hint=(1, None), height=dp(30), spacing=dp(3))
        self.controls.add_widget(self.compact_row)

        self.btn_ufo = ToggleButton(text="UFO SCAN", state="down", font_size=dp(10))
        self.btn_ufo.bind(on_press=self.toggle_ufo)
        self.compact_row.add_widget(self.btn_ufo)

        self.btn_scanline = ToggleButton(text="SCAN LINES", state="down", font_size=dp(10))
        self.btn_scanline.bind(on_press=self.toggle_scanlines)
        self.compact_row.add_widget(self.btn_scanline)

        self.btn_exp_lock = ToggleButton(text="EXP LOCK", state="normal", font_size=dp(10))
        self.btn_exp_lock.bind(on_press=self.toggle_exposure_lock)
        self.compact_row.add_widget(self.btn_exp_lock)

        self.btn_wb_lock = ToggleButton(text="WB LOCK", state="normal", font_size=dp(10))
        self.btn_wb_lock.bind(on_press=self.toggle_white_balance_lock)
        self.compact_row.add_widget(self.btn_wb_lock)

        self.btn_mirror = ToggleButton(text="MIRROR OFF", state="normal", font_size=dp(10))
        self.btn_mirror.bind(on_press=self.toggle_mirror)
        self.compact_row.add_widget(self.btn_mirror)

        self.slider_widgets = []
        self.add_slider_block("camera_zoom", f"CAMERA ZOOM: {self.camera_zoom:.1f}x", 1, self.camera_zoom_max, self.camera_zoom, 0.1, self.on_camera_zoom)
        self.add_slider_block("threshold", f"SENS / THRESHOLD: {self.motion_threshold}", 1, 60, self.motion_threshold, 1, self.on_threshold)
        self.add_slider_block("area", f"MIN MOTION AREA: {self.motion_min_area}px", 1, 80, self.motion_min_area, 1, self.on_area)
        self.add_slider_block("points", f"MAX POINTS: {self.motion_max_points}", 25, 500, self.motion_max_points, 5, self.on_points)
        self.add_slider_block("focus", "MANUAL FOCUS: AUTO  | enable FOCUS first", 0, 255, self.focus_value, 1, self.on_focus)
        self.add_slider_block("capture_radius", f"RETICLE CAPTURE RADIUS: {self.capture_radius}px", 20, 180, self.capture_radius, 5, self.on_capture_radius)
        self.add_slider_block("lock_radius", f"LOCK SEARCH RADIUS: {self.lock_search_radius}px", 15, 220, self.lock_search_radius, 5, self.on_lock_radius)
        self.add_slider_block("flow_radius", f"OPTICAL FLOW RADIUS: {self.flow_radius}px", 12, 80, self.flow_radius, 2, self.on_flow_radius)
        self.add_slider_block("yolo_crop_pad", f"YOLO SNAP CROP PAD: {self.yolo_crop_pad}px", 50, 260, self.yolo_crop_pad, 5, self.on_yolo_crop_pad)

        request_android_permissions()
        self.start_camera()
        Clock.schedule_interval(self.update, 1 / 30.0)

    def add_slider_block(self, attr_name, label_text, min_v, max_v, value, step, callback):
        label = Label(text=label_text, size_hint=(1, None), height=dp(20), color=(0, 1, 0.42, 1), font_size=dp(11))
        slider = Slider(min=min_v, max=max_v, value=value, step=step, size_hint=(1, None), height=dp(22))
        slider.bind(value=callback)
        setattr(self, attr_name + "_label", label)
        setattr(self, attr_name + "_slider", slider)
        self.controls.add_widget(label)
        self.controls.add_widget(slider)
        self.slider_widgets.extend([label, slider])

    def create_kalman(self):
        k = cv2.KalmanFilter(4, 2)
        k.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        k.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        k.processNoiseCov = np.eye(4, dtype=np.float32) * 0.018
        k.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.55
        k.errorCovPost = np.eye(4, dtype=np.float32)
        return k

    def reset_kalman(self, x, y):
        self.kalman = self.create_kalman()
        self.kalman.statePre = np.array([[x], [y], [0], [0]], np.float32)
        self.kalman.statePost = np.array([[x], [y], [0], [0]], np.float32)
        self.kalman_ready = True
        self.pred_x = float(x)
        self.pred_y = float(y)
        self.vel_x = 0.0
        self.vel_y = 0.0

    def kalman_predict(self):
        if not self.kalman_ready:
            return self.pred_x, self.pred_y
        pred = self.kalman.predict()
        self.pred_x = float(pred[0])
        self.pred_y = float(pred[1])
        self.vel_x = float(pred[2])
        self.vel_y = float(pred[3])
        return self.pred_x, self.pred_y

    def kalman_correct(self, x, y):
        if not self.kalman_ready:
            self.reset_kalman(x, y)
            return x, y
        meas = np.array([[np.float32(x)], [np.float32(y)]])
        corrected = self.kalman.correct(meas)
        self.pred_x = float(corrected[0])
        self.pred_y = float(corrected[1])
        self.vel_x = float(corrected[2])
        self.vel_y = float(corrected[3])
        return self.pred_x, self.pred_y

    def load_yolo(self):
        try:
            if not os.path.exists(YOLO_ONNX_PATH):
                self.yolo_loaded = False
                self.yolo_net = None
                self.yolo_status = "YOLO: MODEL MISSING"
                return
            self.yolo_net = cv2.dnn.readNetFromONNX(YOLO_ONNX_PATH)
            try:
                self.yolo_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.yolo_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            except Exception:
                pass
            self.yolo_loaded = True
            self.yolo_status = "YOLO: READY / SNAP ONLY"
        except Exception as exc:
            self.yolo_loaded = False
            self.yolo_net = None
            self.yolo_status = f"YOLO: LOAD FAIL {type(exc).__name__}"

    def start_camera(self):
        self.capture = cv2.VideoCapture(self.camera_index)
        try:
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.capture.set(cv2.CAP_PROP_FPS, 30)
        except Exception:
            pass

        try:
            cur_focus = self.capture.get(cv2.CAP_PROP_FOCUS)
            if cur_focus >= 0:
                self.focus_value = int(max(0, min(255, cur_focus)))
                self.focus_slider.value = self.focus_value
        except Exception:
            pass

        t = threading.Thread(target=self.camera_loop, daemon=True)
        t.start()

    def camera_loop(self):
        while self.running:
            if self.capture is None or not self.capture.isOpened():
                time.sleep(0.05)
                continue
            ok, frame = self.capture.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue
            if self.mirror:
                frame = cv2.flip(frame, 1)
            with self.frame_lock:
                self.frame = frame
            time.sleep(0.001)

    def apply_focus_value(self):
        if self.capture is None:
            return
        try:
            if self.manual_focus_enabled:
                self.capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                ok = self.capture.set(cv2.CAP_PROP_FOCUS, float(self.focus_value))
                readback = self.capture.get(cv2.CAP_PROP_FOCUS)
                self.focus_status = f"FOCUS: MANUAL {self.focus_value}"
                if ok and readback is not None and readback >= 0:
                    self.focus_status += f" / CAM {readback:.0f}"
            else:
                self.capture.set(cv2.CAP_PROP_AUTOFOCUS, 1)
                self.focus_status = "FOCUS: AUTO"
            self.focus_label.text = self.focus_status
        except Exception as exc:
            self.focus_status = f"FOCUS: DRIVER IGNORED ({type(exc).__name__})"
            self.focus_label.text = self.focus_status

    def apply_camera_locks(self):
        if self.capture is None:
            return
        notes = []
        try:
            if self.exposure_locked:
                self.capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                notes.append("EXP LOCK")
            else:
                self.capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
                notes.append("EXP AUTO")
        except Exception:
            notes.append("EXP ?")

        try:
            if self.white_balance_locked:
                self.capture.set(cv2.CAP_PROP_AUTO_WB, 0)
                notes.append("WB LOCK")
            else:
                self.capture.set(cv2.CAP_PROP_AUTO_WB, 1)
                notes.append("WB AUTO")
        except Exception:
            notes.append("WB ?")

        self.camera_status = "CAM CTRL: " + " / ".join(notes)

    def toggle_controls(self, *_):
        self.controls_expanded = self.btn_compact.state != "down"
        self.btn_compact.text = "SLIDERS\nHIDE" if self.controls_expanded else "SLIDERS\nSHOW"
        for w in self.slider_widgets:
            w.opacity = 1 if self.controls_expanded else 0
            w.disabled = not self.controls_expanded
            w.height = dp(20 if isinstance(w, Label) else 22) if self.controls_expanded else 0
        self.controls.height = dp(352) if self.controls_expanded else dp(76)

    def toggle_yolo(self, *_):
        self.yolo_enabled = self.btn_yolo.state == "down"
        self.btn_yolo.text = "SNAP\nYOLO" if self.yolo_enabled else "YOLO\nOFF"
        if self.yolo_enabled and not self.yolo_loaded:
            self.load_yolo()
        if not self.yolo_enabled:
            self.yolo_status = "YOLO: OFF"

    def toggle_flow(self, *_):
        self.flow_enabled = self.btn_flow.state == "down"
        self.btn_flow.text = "FLOW\nON" if self.flow_enabled else "FLOW\nOFF"
        self.flow_points = None
        self.prev_track_gray = None

    def toggle_exposure_lock(self, *_):
        self.exposure_locked = self.btn_exp_lock.state == "down"
        self.btn_exp_lock.text = "EXP LOCK" if self.exposure_locked else "EXP AUTO"
        self.apply_camera_locks()
        self.prev_motion_gray = None

    def toggle_white_balance_lock(self, *_):
        self.white_balance_locked = self.btn_wb_lock.state == "down"
        self.btn_wb_lock.text = "WB LOCK" if self.white_balance_locked else "WB AUTO"
        self.apply_camera_locks()
        self.prev_motion_gray = None

    def request_reticle_capture(self, *_):
        self.capture_requested = True
        self.lock_status = "CAPTURE ARMED: CENTER RETICLE"
        self.btn_capture.text = "CAPTURE\nARMED"

    def unlock_target(self, *_):
        self.target_locked = False
        self.lock_target = None
        self.capture_requested = False
        self.lock_status = "RETICLE READY"
        self.btn_capture.text = "CAPTURE\nRETICLE"
        self.kalman_ready = False
        self.track_confidence = 0.0
        self.flow_points = None
        self.prev_track_gray = None
        self.snap_tag_pending = False
        self.snap_tag_done = False
        self.snap_label = "UNSCANNED"
        self.snap_conf = 0.0
        self.snap_box = None
        self.snap_crop_rect = None
        if self.yolo_enabled:
            self.yolo_status = "YOLO: READY / SNAP ONLY" if self.yolo_loaded else "YOLO: MODEL MISSING"

    def toggle_manual_focus(self, *_):
        self.manual_focus_enabled = self.btn_focus.state == "down"
        self.btn_focus.text = "FOCUS\nMAN" if self.manual_focus_enabled else "FOCUS\nAUTO"
        self.apply_focus_value()

    def toggle_ufo(self, *_):
        self.ufo_scan_enabled = self.btn_ufo.state == "down"
        self.btn_ufo.text = "UFO SCAN" if self.ufo_scan_enabled else "UFO OFF"
        self.prev_motion_gray = None
        self.motion_history.clear()
        self.unlock_target()

    def toggle_scanlines(self, *_):
        self.scanline_enabled = self.btn_scanline.state == "down"

    def toggle_radar(self, *_):
        self.radar_enabled = self.btn_radar.state == "down"
        self.btn_radar.text = "RADAR\nON" if self.radar_enabled else "RADAR\nOFF"
        if not self.radar_enabled:
            self.radar_blips.clear()

    def toggle_mirror(self, *_):
        self.mirror = self.btn_mirror.state == "down"
        self.btn_mirror.text = "MIRROR ON" if self.mirror else "MIRROR OFF"
        self.prev_motion_gray = None
        self.unlock_target()

    def cycle_render(self, *_):
        modes = ["NORMAL", "EDGE", "THERMAL", "DITHER"]
        self.render_mode = modes[(modes.index(self.render_mode) + 1) % len(modes)]
        self.btn_render.text = "MODE\n" + self.render_mode

    def cycle_color(self, *_):
        colors = [("GREEN", GREEN), ("CYAN", CYAN), ("AMBER", AMBER), ("RED", RED), ("WHITE", WHITE)]
        self.overlay_i = (self.overlay_i + 1) % len(colors)
        name, col = colors[self.overlay_i]
        self.overlay_color = col

    def on_threshold(self, _, value):
        self.motion_threshold = int(value)
        self.threshold_label.text = f"SENS / THRESHOLD: {self.motion_threshold}"

    def on_area(self, _, value):
        self.motion_min_area = int(value)
        self.area_label.text = f"MIN MOTION AREA: {self.motion_min_area}px"

    def on_points(self, _, value):
        self.motion_max_points = int(value)
        self.points_label.text = f"MAX POINTS: {self.motion_max_points}"

    def on_focus(self, _, value):
        self.focus_value = int(value)
        if self.manual_focus_enabled:
            self.apply_focus_value()
        else:
            self.focus_label.text = f"MANUAL FOCUS: {self.focus_value}  | enable FOCUS first"

    def on_camera_zoom(self, _, value):
        self.camera_zoom = max(1.0, float(value))
        self.camera_zoom_label.text = f"CAMERA ZOOM: {self.camera_zoom:.1f}x"
        self.prev_motion_gray = None
        self.unlock_target()

    def on_capture_radius(self, _, value):
        self.capture_radius = int(value)
        self.capture_radius_label.text = f"RETICLE CAPTURE RADIUS: {self.capture_radius}px"

    def on_lock_radius(self, _, value):
        self.lock_search_radius = int(value)
        self.lock_radius_label.text = f"LOCK SEARCH RADIUS: {self.lock_search_radius}px"

    def on_flow_radius(self, _, value):
        self.flow_radius = int(value)
        self.flow_radius_label.text = f"OPTICAL FLOW RADIUS: {self.flow_radius}px"

    def on_yolo_crop_pad(self, _, value):
        self.yolo_crop_pad = int(value)
        self.yolo_crop_pad_label.text = f"YOLO SNAP CROP PAD: {self.yolo_crop_pad}px"

    def apply_camera_zoom(self, frame):
        z = max(1.0, float(self.camera_zoom))
        if z <= 1.01:
            return frame
        h, w = frame.shape[:2]
        crop_w = max(8, int(w / z))
        crop_h = max(8, int(h / z))
        x1 = max(0, (w - crop_w) // 2)
        y1 = max(0, (h - crop_h) // 2)
        x2 = min(w, x1 + crop_w)
        y2 = min(h, y1 + crop_h)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return frame
        return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)

    def render_base(self, frame):
        if self.render_mode == "NORMAL":
            return frame.copy()
        if self.render_mode == "EDGE":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 45, 120)
            out = np.zeros_like(frame)
            out[:, :, 1] = edges
            out[:, :, 0] = edges // 5
            return cv2.addWeighted(frame, 0.28, out, 1.0, 0)
        if self.render_mode == "THERMAL":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            eq = cv2.equalizeHist(gray)
            return cv2.applyColorMap(eq, cv2.COLORMAP_TURBO)
        if self.render_mode == "DITHER":
            small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            bayer = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], dtype=np.uint8) * 16
            tile = np.tile(bayer, (gray.shape[0] // 4 + 1, gray.shape[1] // 4 + 1))[:gray.shape[0], :gray.shape[1]]
            d = (gray > tile).astype(np.uint8) * 255
            out = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
            out[:, :, 1] = d
            out[:, :, 0] = d // 8
            return cv2.resize(out, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        return frame.copy()

    def detect_micro_motion(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.motion_blur > 1:
            k = self.motion_blur if self.motion_blur % 2 == 1 else self.motion_blur + 1
            gray = cv2.GaussianBlur(gray, (k, k), 0)

        if self.prev_motion_gray is None:
            self.prev_motion_gray = gray
            return []

        diff = cv2.absdiff(self.prev_motion_gray, gray)
        self.prev_motion_gray = gray
        _, mask = cv2.threshold(diff, self.motion_threshold, 255, cv2.THRESH_BINARY)
        kernel = np.ones((1, 1), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

        points = []
        total_area = 0
        for i in range(1, num_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area >= self.motion_min_area:
                cx, cy = centroids[i]
                x = int(cx)
                y = int(cy)
                bw = int(stats[i, cv2.CC_STAT_WIDTH])
                bh = int(stats[i, cv2.CC_STAT_HEIGHT])
                left = int(stats[i, cv2.CC_STAT_LEFT])
                top = int(stats[i, cv2.CC_STAT_TOP])
                points.append({"x": x, "y": y, "area": area, "box": (left, top, bw, bh)})
                total_area += area

        if len(points) > self.motion_max_points:
            points = random.sample(points, self.motion_max_points)

        self.last_motion_count = len(points)
        self.last_motion_area = total_area
        self.last_points = points
        now = time.time()
        for p in points:
            self.motion_history.append((p["x"], p["y"], p["area"], now))

        self.process_advanced_lock(points, frame)
        return points

    def init_target_memory(self, frame, target):
        x, y = int(target["x"]), int(target["y"])
        h, w = frame.shape[:2]
        half = 8
        x1, y1 = max(0, x - half), max(0, y - half)
        x2, y2 = min(w, x + half + 1), min(h, y + half + 1)
        crop = frame[y1:y2, x1:x2]
        if crop.size:
            self.target_memory["color"] = np.mean(crop.reshape(-1, 3), axis=0)
            self.target_memory["brightness"] = float(np.mean(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)))
        self.target_memory["area"] = max(1, int(target.get("area", 1)))
        self.target_memory["last_box"] = target.get("box", (x, y, 1, 1))

    def init_flow_points(self, frame, target):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x, y = int(target["x"]), int(target["y"])
        h, w = gray.shape[:2]
        r = self.flow_radius
        x1, y1 = max(0, x - r), max(0, y - r)
        x2, y2 = min(w, x + r), min(h, y + r)
        roi = gray[y1:y2, x1:x2]
        pts = None
        if roi.size:
            pts = cv2.goodFeaturesToTrack(roi, maxCorners=24, qualityLevel=0.01, minDistance=3, blockSize=3)
            if pts is not None:
                pts[:, 0, 0] += x1
                pts[:, 0, 1] += y1
        if pts is None or len(pts) == 0:
            pts = np.array([[[np.float32(x), np.float32(y)]]], dtype=np.float32)
        self.flow_points = pts.astype(np.float32)
        self.prev_track_gray = gray

    def update_optical_flow(self, frame):
        if not self.flow_enabled or self.flow_points is None or self.prev_track_gray is None:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        try:
            next_pts, status, err = cv2.calcOpticalFlowPyrLK(
                self.prev_track_gray,
                gray,
                self.flow_points,
                None,
                winSize=(21, 21),
                maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 12, 0.03),
            )
            self.prev_track_gray = gray
            if next_pts is None or status is None:
                return None
            good_new = next_pts[status.flatten() == 1]
            if len(good_new) == 0:
                return None
            self.flow_points = good_new.reshape(-1, 1, 2).astype(np.float32)
            mx = float(np.median(good_new[:, 0]))
            my = float(np.median(good_new[:, 1]))
            return mx, my, min(1.0, len(good_new) / 8.0)
        except Exception:
            return None

    def process_advanced_lock(self, points, frame):
        now = time.time()
        h, w = frame.shape[:2]
        cx = w // 2
        cy = h // 2

        if self.capture_requested:
            candidates = []
            for p in points:
                d = math.hypot(p["x"] - cx, p["y"] - cy)
                if d <= self.capture_radius:
                    score = (self.capture_radius - d) * 2.0 + min(80, p.get("area", 1))
                    candidates.append((score, p))
            if candidates:
                candidates.sort(key=lambda item: item[0], reverse=True)
                best = candidates[0][1].copy()
                self.lock_target = best
                self.target_locked = True
                self.capture_requested = False
                self.lock_last_seen = now
                self.lock_status = f"LOCKED FROM RETICLE X{best['x']} Y{best['y']} A{best['area']}"
                self.btn_capture.text = "CAPTURE\nRETICLE"
                self.reset_kalman(best["x"], best["y"])
                self.init_target_memory(frame, best)
                self.init_flow_points(frame, best)
                self.track_confidence = 1.0
                self.snap_tag_pending = bool(self.yolo_enabled)
                self.snap_tag_done = False
                self.snap_label = "SNAP PENDING"
                self.snap_conf = 0.0
                self.snap_box = None
                self.snap_crop_rect = None
                self.yolo_status = "YOLO: SNAP QUEUED"
            else:
                self.lock_status = "CAPTURE ARMED: NO MOTION IN RETICLE"

        if not self.target_locked or self.lock_target is None:
            return

        px, py = self.kalman_predict()
        measurement = None
        confidence = 0.0
        source = "PREDICT"

        # 1) Optical flow measurement
        flow = self.update_optical_flow(frame)
        if flow is not None:
            fx, fy, fc = flow
            if math.hypot(fx - px, fy - py) <= self.lock_search_radius * 1.7:
                measurement = (fx, fy)
                confidence = max(confidence, 0.55 + 0.35 * fc)
                source = "FLOW"

        # 2) Motion reacquire near predicted target
        candidates = []
        for p in points:
            d_pred = math.hypot(p["x"] - px, p["y"] - py)
            d_last = math.hypot(p["x"] - self.lock_target["x"], p["y"] - self.lock_target["y"])
            if d_pred <= self.lock_search_radius or d_last <= self.lock_search_radius:
                area_bonus = min(60, p.get("area", 1))
                score = (self.lock_search_radius - min(d_pred, d_last)) * 2.3 + area_bonus
                candidates.append((score, p))
        if candidates:
            candidates.sort(key=lambda item: item[0], reverse=True)
            best = candidates[0][1]
            # Blend flow and motion if both exist
            if measurement is not None:
                mx = measurement[0] * 0.55 + best["x"] * 0.45
                my = measurement[1] * 0.55 + best["y"] * 0.45
                confidence = max(confidence, 0.88)
                source = "FLOW+MOTION"
            else:
                mx, my = best["x"], best["y"]
                confidence = max(confidence, 0.74)
                source = "MOTION"
            measurement = (mx, my)
            updated_box = best.get("box", self.lock_target.get("box", (int(mx), int(my), 1, 1)))
            updated_area = best.get("area", self.lock_target.get("area", 1))
        else:
            updated_box = self.lock_target.get("box", (int(px), int(py), 1, 1))
            updated_area = self.lock_target.get("area", 1)

        if measurement is not None:
            sx, sy = self.kalman_correct(measurement[0], measurement[1])
            self.lock_last_seen = now
            self.track_confidence = min(1.0, self.track_confidence * 0.82 + confidence * 0.28)
            self.lock_target = {
                "x": int(sx),
                "y": int(sy),
                "area": int(updated_area),
                "box": updated_box,
            }
            # Refresh sparse flow when it degrades
            if self.flow_points is None or len(self.flow_points) < 5:
                self.init_flow_points(frame, self.lock_target)
            self.lock_status = f"ADV LOCK {source} X{int(sx)} Y{int(sy)} CONF {self.track_confidence:.2f}"
        else:
            age = now - self.lock_last_seen
            self.track_confidence = max(0.0, self.track_confidence * 0.92 - 0.025)
            self.lock_target = {
                "x": int(px),
                "y": int(py),
                "area": int(updated_area),
                "box": updated_box,
            }
            self.lock_status = f"TARGET MEMORY / SEARCHING {age:.1f}s CONF {self.track_confidence:.2f}"
            if age > self.lock_lost_seconds and self.track_confidence < 0.20:
                self.target_locked = False
                self.lock_target = None
                self.kalman_ready = False
                self.lock_status = "TARGET LOST - RETICLE READY"
                self.flow_points = None
                self.snap_tag_pending = False

    def get_target_crop_rect(self, frame, target, pad=None):
        h, w = frame.shape[:2]
        x = int(target.get("x", w // 2))
        y = int(target.get("y", h // 2))
        left, top, bw, bh = target.get("box", (x, y, 1, 1))
        pad = self.yolo_crop_pad if pad is None else int(pad)
        x1 = max(0, min(left, x) - pad)
        y1 = max(0, min(top, y) - pad)
        x2 = min(w, max(left + max(1, bw), x) + pad)
        y2 = min(h, max(top + max(1, bh), y) + pad)
        min_size = 96
        if x2 - x1 < min_size:
            extra = (min_size - (x2 - x1)) // 2
            x1 = max(0, x1 - extra)
            x2 = min(w, x2 + extra)
        if y2 - y1 < min_size:
            extra = (min_size - (y2 - y1)) // 2
            y1 = max(0, y1 - extra)
            y2 = min(h, y2 + extra)
        return int(x1), int(y1), int(x2), int(y2)

    def parse_yolo_output(self, output, crop_w, crop_h):
        out = output[0] if isinstance(output, (tuple, list)) else output
        out = np.squeeze(out)
        if out.ndim != 2:
            return []
        if out.shape[0] < out.shape[1] and out.shape[0] in (84, 85, 116):
            out = out.T

        detections = []
        x_factor = crop_w / float(self.yolo_input_size)
        y_factor = crop_h / float(self.yolo_input_size)
        for row in out:
            if row.shape[0] < 6:
                continue
            scores = row[4:]
            class_id = int(np.argmax(scores))
            conf = float(scores[class_id])
            if conf < self.yolo_conf_threshold:
                continue
            cx, cy, bw, bh = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            x1 = int((cx - bw / 2) * x_factor)
            y1 = int((cy - bh / 2) * y_factor)
            x2 = int((cx + bw / 2) * x_factor)
            y2 = int((cy + bh / 2) * y_factor)
            x1 = max(0, min(crop_w - 1, x1))
            y1 = max(0, min(crop_h - 1, y1))
            x2 = max(0, min(crop_w - 1, x2))
            y2 = max(0, min(crop_h - 1, y2))
            detections.append((x1, y1, x2, y2, conf, class_id))

        if not detections:
            return []
        boxes = [[x1, y1, max(1, x2 - x1), max(1, y2 - y1)] for x1, y1, x2, y2, conf, class_id in detections]
        scores = [conf for x1, y1, x2, y2, conf, class_id in detections]
        idxs = cv2.dnn.NMSBoxes(boxes, scores, self.yolo_conf_threshold, self.yolo_nms_threshold)
        if idxs is None or len(idxs) == 0:
            return []
        return [detections[int(i)] for i in np.array(idxs).flatten()]

    def run_yolo_snap_once(self, frame):
        if not self.snap_tag_pending or self.snap_tag_done:
            return
        if not self.yolo_enabled:
            self.snap_label = "YOLO OFF"
            self.snap_tag_done = True
            self.snap_tag_pending = False
            self.yolo_status = "YOLO: OFF / SNAP SKIPPED"
            return
        if not self.target_locked or self.lock_target is None:
            self.snap_tag_pending = False
            return
        if not self.yolo_loaded or self.yolo_net is None:
            self.load_yolo()
            if not self.yolo_loaded:
                self.snap_label = "NO MODEL"
                self.snap_conf = 0.0
                self.snap_tag_done = True
                self.snap_tag_pending = False
                self.yolo_status = "YOLO: MODEL MISSING"
                return

        try:
            target_snapshot = self.lock_target.copy()
            x1, y1, x2, y2 = self.get_target_crop_rect(frame, target_snapshot)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                self.snap_label = "EMPTY CROP"
                self.snap_tag_done = True
                self.snap_tag_pending = False
                self.yolo_status = "YOLO: EMPTY CROP"
                return
            blob = cv2.dnn.blobFromImage(crop, 1.0 / 255.0, (self.yolo_input_size, self.yolo_input_size), swapRB=True, crop=False)
            self.yolo_net.setInput(blob)
            output = self.yolo_net.forward()
            crop_h, crop_w = crop.shape[:2]
            detections = self.parse_yolo_output(output, crop_w, crop_h)
            if not detections:
                self.snap_label = "UNKNOWN"
                self.snap_conf = 0.0
                self.snap_box = None
                self.snap_crop_rect = (x1, y1, x2, y2)
                self.snap_tag_done = True
                self.snap_tag_pending = False
                self.yolo_status = "YOLO SNAP: UNKNOWN / IDLE"
                return
            tx = target_snapshot["x"] - x1
            ty = target_snapshot["y"] - y1
            scored = []
            for dx1, dy1, dx2, dy2, conf, class_id in detections:
                dcx = (dx1 + dx2) / 2.0
                dcy = (dy1 + dy2) / 2.0
                dist = math.hypot(dcx - tx, dcy - ty)
                inside = 1.0 if (dx1 <= tx <= dx2 and dy1 <= ty <= dy2) else 0.0
                score = conf + inside * 0.35 - dist * 0.001
                scored.append((score, dx1, dy1, dx2, dy2, conf, class_id))
            scored.sort(key=lambda q: q[0], reverse=True)
            _, dx1, dy1, dx2, dy2, conf, class_id = scored[0]
            label = COCO_CLASSES[class_id] if 0 <= class_id < len(COCO_CLASSES) else f"class_{class_id}"
            self.snap_label = label.upper()
            self.snap_conf = float(conf)
            self.snap_box = (x1 + dx1, y1 + dy1, x1 + dx2, y1 + dy2)
            self.snap_crop_rect = (x1, y1, x2, y2)
            self.snap_tag_done = True
            self.snap_tag_pending = False
            self.yolo_status = f"YOLO SNAP TAGGED: {self.snap_label} {self.snap_conf:.2f} / IDLE"
        except Exception as exc:
            self.snap_label = "YOLO ERROR"
            self.snap_conf = 0.0
            self.snap_box = None
            self.snap_tag_done = True
            self.snap_tag_pending = False
            self.yolo_status = f"YOLO SNAP FAIL: {type(exc).__name__}"

    def draw_sigint_frame(self, img):
        h, w = img.shape[:2]
        col = self.overlay_color
        margin = 10
        for a, b in [
            ((margin, margin), (margin + 55, margin)), ((margin, margin), (margin, margin + 55)),
            ((w - margin, margin), (w - margin - 55, margin)), ((w - margin, margin), (w - margin, margin + 55)),
            ((margin, h - margin), (margin + 55, h - margin)), ((margin, h - margin), (margin, h - margin - 55)),
            ((w - margin, h - margin), (w - margin - 55, h - margin)), ((w - margin, h - margin), (w - margin, h - margin - 55)),
        ]:
            cv2.line(img, a, b, col, 1)

        cx, cy = w // 2, h // 2
        ret_col = CYAN if self.capture_requested else (self.lock_box_color if self.target_locked else col)
        cv2.line(img, (cx - 18, cy), (cx - 5, cy), ret_col, 1)
        cv2.line(img, (cx + 5, cy), (cx + 18, cy), ret_col, 1)
        cv2.line(img, (cx, cy - 18), (cx, cy - 5), ret_col, 1)
        cv2.line(img, (cx, cy + 5), (cx, cy + 18), ret_col, 1)
        cv2.circle(img, (cx, cy), 22, ret_col, 1)
        cv2.circle(img, (cx, cy), self.capture_radius, tuple(int(c * 0.45) for c in ret_col), 1)

        if self.scanline_enabled:
            for y in range(0, h, 8):
                cv2.line(img, (0, y), (w, y), (0, 45, 20), 1)
        return img

    def draw_micro_motion_and_lock(self, img, points):
        h, w = img.shape[:2]
        col = self.overlay_color
        now = time.time()

        fresh = deque(maxlen=900)
        for x, y, area, ts in self.motion_history:
            age = now - ts
            if age <= self.motion_decay_seconds:
                fresh.append((x, y, area, ts))
                fade = max(0.15, 1.0 - age / self.motion_decay_seconds)
                cv2.circle(img, (x, y), 1 if area <= 3 else 2, tuple(int(c * fade) for c in col), 1)
        self.motion_history = fresh

        for p in points:
            cv2.circle(img, (p["x"], p["y"]), 2 if p["area"] <= 3 else 3, col, 1)

        if self.target_locked and self.lock_target is not None:
            p = self.lock_target
            x, y, area = p["x"], p["y"], p["area"]
            left, top, bw, bh = p.get("box", (x, y, 1, 1))
            pad = 14 if area <= 3 else 9
            vx1 = max(0, int(left - pad))
            vy1 = max(0, int(top - pad))
            vx2 = min(w - 1, int(left + max(1, bw) + pad))
            vy2 = min(h - 1, int(top + max(1, bh) + pad))
            lock_col = self.lock_box_color if self.track_confidence >= 0.35 else AMBER
            cv2.rectangle(img, (vx1, vy1), (vx2, vy2), lock_col, 2)
            cv2.circle(img, (x, y), 12, lock_col, 1)
            cv2.line(img, (x - 18, y), (x - 5, y), lock_col, 1)
            cv2.line(img, (x + 5, y), (x + 18, y), lock_col, 1)
            cv2.line(img, (x, y - 18), (x, y - 5), lock_col, 1)
            cv2.line(img, (x, y + 5), (x, y + 18), lock_col, 1)

            # Predicted next point
            px = int(x + self.vel_x * 5)
            py = int(y + self.vel_y * 5)
            cv2.line(img, (x, y), (px, py), AMBER, 1)
            cv2.circle(img, (px, py), 4, AMBER, 1)

            tag = self.snap_label if self.snap_conf <= 0 else f"{self.snap_label} {self.snap_conf:.2f}"
            label = f"ADV LOCK {self.track_confidence:.2f} // {tag}"
            cv2.putText(img, label, (min(w - 270, x + 8), max(16, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, label, (min(w - 270, x + 8), max(16, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, lock_col, 1, cv2.LINE_AA)

        if self.snap_box is not None and self.target_locked:
            ax1, ay1, ax2, ay2 = [int(v) for v in self.snap_box]
            cv2.rectangle(img, (ax1, ay1), (ax2, ay2), AMBER, 1)
            txt = f"SNAP TAG: {self.snap_label} {self.snap_conf:.2f}"
            cv2.putText(img, txt, (ax1, max(14, ay1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, txt, (ax1, max(14, ay1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.36, AMBER, 1, cv2.LINE_AA)

        return img

    def draw_target_inspector(self, img, raw_frame):
        h, w = img.shape[:2]
        col = AMBER if self.snap_tag_done else (self.lock_box_color if self.target_locked else self.overlay_color)
        target = self.lock_target if self.target_locked else None
        panel_w, panel_h = 200, 174
        px2, py1 = w - 14, 82
        px1 = max(8, px2 - panel_w)
        py2 = min(h - 90, py1 + panel_h)
        if py2 <= py1 + 40:
            return img

        overlay = img.copy()
        cv2.rectangle(overlay, (px1, py1), (px2, py2), (0, 18, 8), -1)
        cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)
        cv2.rectangle(img, (px1, py1), (px2, py2), col, 1)

        title = "ADV LOCK VIEWER"
        cv2.putText(img, title, (px1 + 8, py1 + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, title, (px1 + 8, py1 + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.36, col, 1, cv2.LINE_AA)

        if target is None:
            msg = "AIM + CAPTURE"
            cv2.putText(img, msg, (px1 + 28, py1 + 76), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, msg, (px1 + 28, py1 + 76), cv2.FONT_HERSHEY_SIMPLEX, 0.46, col, 1, cv2.LINE_AA)
            return img

        x, y, area = int(target["x"]), int(target["y"]), int(target["area"])
        raw_h, raw_w = raw_frame.shape[:2]
        half = self.inspector_crop_half
        x1, y1 = max(0, x - half), max(0, y - half)
        x2, y2 = min(raw_w, x + half + 1), min(raw_h, y + half + 1)
        crop = raw_frame[y1:y2, x1:x2]
        if crop.size:
            view_w, view_h = panel_w - 18, panel_h - 76
            zoomed = cv2.resize(crop, (view_w, view_h), interpolation=cv2.INTER_NEAREST)
            zx1, zy1 = px1 + 9, py1 + 28
            zx2, zy2 = zx1 + view_w, zy1 + view_h
            img[zy1:zy2, zx1:zx2] = zoomed
            rel_x = (x - x1) / max(1, (x2 - x1 - 1))
            rel_y = (y - y1) / max(1, (y2 - y1 - 1))
            tx = int(zx1 + rel_x * view_w)
            ty = int(zy1 + rel_y * view_h)
            cv2.rectangle(img, (zx1, zy1), (zx2, zy2), col, 1)
            cv2.circle(img, (tx, ty), 9, col, 1)
            cv2.line(img, (tx - 15, ty), (tx - 5, ty), col, 1)
            cv2.line(img, (tx + 5, ty), (tx + 15, ty), col, 1)
            cv2.line(img, (tx, ty - 15), (tx, ty - 5), col, 1)
            cv2.line(img, (tx, ty + 5), (tx, ty + 15), col, 1)

        tag = self.snap_label if self.snap_conf <= 0 else f"{self.snap_label} {self.snap_conf:.2f}"
        footers = [
            f"X{x} Y{y} A{area}px CONF {self.track_confidence:.2f}",
            f"TAG: {tag}",
            f"{self.yolo_status}",
        ]
        for i, txt in enumerate(footers):
            yy = py2 - 42 + i * 15
            cv2.putText(img, txt, (px1 + 8, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.285, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, txt, (px1 + 8, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.285, AMBER if i >= 1 else col, 1, cv2.LINE_AA)
        return img

    def draw_radar(self, img, points):
        if not self.radar_enabled:
            return img
        h, w = img.shape[:2]
        col = self.overlay_color
        now = time.time()
        r = self.radar_radius
        cx = w - r - self.radar_margin
        cy = h - r - self.radar_margin - 8
        for p in points:
            nx = (p["x"] / max(1, w)) * 2.0 - 1.0
            ny = (p["y"] / max(1, h)) * 2.0 - 1.0
            self.radar_blips.append((nx, ny, p.get("area", 1), now))
        overlay = img.copy()
        cv2.circle(overlay, (cx, cy), r + 7, (0, 22, 10), -1)
        cv2.addWeighted(overlay, 0.28, img, 0.72, 0, img)
        cv2.circle(img, (cx, cy), r, col, 1)
        cv2.circle(img, (cx, cy), int(r * 0.66), tuple(int(c * 0.65) for c in col), 1)
        cv2.circle(img, (cx, cy), int(r * 0.33), tuple(int(c * 0.45) for c in col), 1)
        cv2.line(img, (cx - r, cy), (cx + r, cy), tuple(int(c * 0.45) for c in col), 1)
        cv2.line(img, (cx, cy - r), (cx, cy + r), tuple(int(c * 0.45) for c in col), 1)

        self.radar_sweep_angle = (self.radar_sweep_angle + 4.0) % 360.0
        ang = math.radians(self.radar_sweep_angle)
        sx, sy = int(cx + math.cos(ang) * r), int(cy + math.sin(ang) * r)
        cv2.line(img, (cx, cy), (sx, sy), col, 1)

        fresh = deque(maxlen=450)
        strongest = 0
        for nx, ny, area, ts in self.radar_blips:
            age = now - ts
            if age <= self.radar_decay_seconds:
                fresh.append((nx, ny, area, ts))
                fade = max(0.18, 1.0 - age / self.radar_decay_seconds)
                bx = int(cx + nx * r * 0.92)
                by = int(cy + ny * r * 0.92)
                if (bx - cx) ** 2 + (by - cy) ** 2 <= r * r:
                    bcol = tuple(int(c * fade) for c in col)
                    cv2.circle(img, (bx, by), 2 if area <= 8 else 3, bcol, -1)
                    strongest = max(strongest, int(area))
        self.radar_blips = fresh
        cv2.putText(img, "RADAR", (cx - r, cy - r - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, "RADAR", (cx - r, cy - r - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1, cv2.LINE_AA)
        cv2.putText(img, f"BLIPS {len(self.radar_blips)}", (cx - r, cy + r + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, f"BLIPS {len(self.radar_blips)}", (cx - r, cy + r + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1, cv2.LINE_AA)
        if strongest > 0:
            cv2.putText(img, f"SIG {strongest}", (cx + 12, cy + r + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, f"SIG {strongest}", (cx + 12, cy + r + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1, cv2.LINE_AA)
        return img

    def draw_hud_text(self, img):
        h, w = img.shape[:2]
        col = self.overlay_color
        self.fps_frames += 1
        now = time.time()
        if now - self.fps_clock >= 1.0:
            self.fps = self.fps_frames / (now - self.fps_clock)
            self.fps_frames = 0
            self.fps_clock = now
        target_txt = "NONE"
        if self.lock_target is not None:
            target_txt = f"X{self.lock_target['x']} Y{self.lock_target['y']} A{self.lock_target['area']}"
        snap_txt = self.snap_label if self.snap_conf <= 0 else f"{self.snap_label} {self.snap_conf:.2f}"
        lines = [
            "MINOS SIGINT // ADVANCED LOCK",
            f"THRESH: {self.motion_threshold} MINAREA: {self.motion_min_area}px POINTS: {self.last_motion_count} FPS: {self.fps:.1f}",
            f"ZOOM: {self.camera_zoom:.1f}x LOCK: {target_txt} CONF: {self.track_confidence:.2f}",
            f"{self.lock_status}",
            f"SNAP TAG: {snap_txt} | {self.yolo_status}",
            f"{self.focus_status} | {self.camera_status}",
        ]
        y = 28
        for line in lines:
            cv2.putText(img, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, col, 1, cv2.LINE_AA)
            y += 18
        warning = "RETICLE CAPTURE -> YOLO SNAP ONCE -> KALMAN + FLOW + MOTION MEMORY TRACK"
        cv2.putText(img, warning, (14, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, warning, (14, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)
        return img

    def update(self, _dt):
        with self.frame_lock:
            frame = None if self.frame is None else self.frame.copy()
        if frame is None:
            return
        frame = self.apply_camera_zoom(frame)
        display = self.render_base(frame)

        points = []
        if self.ufo_scan_enabled:
            points = self.detect_micro_motion(frame)
            self.run_yolo_snap_once(frame)
            display = self.draw_micro_motion_and_lock(display, points)
        else:
            self.prev_motion_gray = None
            self.last_motion_count = 0
            self.last_motion_area = 0

        display = self.draw_target_inspector(display, frame)
        display = self.draw_radar(display, points)
        display = self.draw_sigint_frame(display)
        display = self.draw_hud_text(display)

        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        rgb = cv2.flip(rgb, 0)
        texture = Texture.create(size=(rgb.shape[1], rgb.shape[0]), colorfmt="rgb")
        texture.blit_buffer(rgb.tobytes(), colorfmt="rgb", bufferfmt="ubyte")
        self.image.texture = texture

        self.status.text = (
            f"MINOS SIGINT // ADV LOCK | points:{self.last_motion_count} "
            f"| lock:{'ON' if self.target_locked else 'OFF'} | conf:{self.track_confidence:.2f} | tag:{self.snap_label}"
        )

    def stop(self):
        self.running = False
        try:
            if self.capture is not None:
                self.capture.release()
        except Exception:
            pass


class MinosSigintUFOApp(App):
    def build(self):
        Window.clearcolor = (0, 0, 0, 1)
        self.root_widget = MinosSigintUFO()
        return self.root_widget

    def on_stop(self):
        try:
            self.root_widget.stop()
        except Exception:
            pass


if __name__ == "__main__":
    MinosSigintUFOApp().run()
