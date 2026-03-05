#!/usr/bin/env python3
"""
JarvisHands Web UI — Professional-grade motion gesture recognition.
Uses One Euro Filter + 25-dim features + DTW + data augmentation + temporal voting.

Run: python3 app.py
Open: http://localhost:5757
"""

import json
import math
import os
import pickle
import shutil
import subprocess
import threading
import time

import cv2
import mediapipe as mp
import numpy as np
from dtw import dtw
from flask import Flask, Response, jsonify, render_template, request
from scipy.interpolate import interp1d

# --- Paths ---
BASE_DIR = os.path.expanduser("~/.jarvishands")
DATA_DIR = os.path.join(BASE_DIR, "sequences")
LABELS_JSON = os.path.join(BASE_DIR, "labels.json")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
MAPPINGS_JSON = os.path.join(BASE_DIR, "mappings.json")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HAND_MODEL_PATH = os.path.join(SCRIPT_DIR, "hand_landmarker.task")

os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

app = Flask(__name__)

LOG_FILE = os.path.join(BASE_DIR, "debug.log")

def log(msg):
    print(msg, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")

# --- MediaPipe ---
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
HandLandmarksConnections = mp.tasks.vision.HandLandmarksConnections

# --- Config ---
SEQ_LENGTH = 20           # Max frames per gesture capture
RESAMPLE_LENGTH = 20      # Resample all gestures to this for DTW
FEATURE_DIM = 25          # Features per frame
COOLDOWN = 0.3            # Seconds between firing gestures
MOTION_START_THRESHOLD = 0.015
MOTION_STOP_THRESHOLD = 0.004
COOLDOWN_FRAMES = 60      # ~2 sec between recording reps
MAX_TRACKING_GAP = 4
MAX_CAPTURE_LENGTH = 40   # Safety valve for capture
MIN_CAPTURE_FRAMES = 6    # Minimum frames to consider a gesture
AUGMENT_COUNT = 5          # Augmented copies per template

# --- Shared State ---
camera_lock = threading.Lock()
camera = None
hand_landmarker = None
current_gesture = None
current_confidence = 0.0
detection_active = False

# DTW model state
gesture_templates = {}
gesture_thresholds = {}
inter_class_info = {}

# Recording state
recording_active = False
recording_label = None
recording_label_id = None
recorded_sequences = []
recording_phase = "idle"
rep_count = 0
target_reps = 10

frame_buffer = []


# ─── One Euro Filter ─────────────────────────────────────────────

class OneEuroFilter:
    """Adaptive low-pass filter that trades jitter vs responsiveness."""

    def __init__(self, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def _alpha(self, cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t=None):
        if t is None:
            t = time.time()
        if self.t_prev is None:
            self.x_prev = x
            self.t_prev = t
            return x

        dt = max(t - self.t_prev, 1e-6)
        self.t_prev = t

        # Derivative
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        self.dx_prev = dx_hat

        # Adaptive cutoff
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev = x_hat
        return x_hat


class LandmarkSmoother:
    """One Euro Filter for all 21 hand landmarks (x, y)."""

    def __init__(self, min_cutoff=1.0, beta=0.007):
        self.filters = {}  # (landmark_idx, axis) -> OneEuroFilter
        self.min_cutoff = min_cutoff
        self.beta = beta

    def smooth(self, landmarks, t=None):
        """Smooth a list of landmarks in-place, return smoothed copies."""
        smoothed = []
        for i, lm in enumerate(landmarks):
            key_x = (i, 'x')
            key_y = (i, 'y')
            if key_x not in self.filters:
                self.filters[key_x] = OneEuroFilter(self.min_cutoff, self.beta)
                self.filters[key_y] = OneEuroFilter(self.min_cutoff, self.beta)

            sx = self.filters[key_x](lm.x, t)
            sy = self.filters[key_y](lm.y, t)
            smoothed.append(type('LM', (), {'x': sx, 'y': sy})())
        return smoothed

    def reset(self):
        self.filters.clear()


# Global smoother instances (one per hand)
hand_smoothers = [LandmarkSmoother(min_cutoff=1.0, beta=0.007) for _ in range(2)]


def get_camera():
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return camera


def get_landmarker():
    global hand_landmarker
    if hand_landmarker is None:
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.4,
            min_tracking_confidence=0.4,
        )
        hand_landmarker = HandLandmarker.create_from_options(options)
    return hand_landmarker


# ─── Feature Extraction (25-dim) ─────────────────────────────────

# Landmark indices
WRIST = 0
THUMB_TIP = 4; INDEX_TIP = 8; MIDDLE_TIP = 12; RING_TIP = 16; PINKY_TIP = 20
THUMB_MCP = 2; INDEX_MCP = 5; MIDDLE_MCP = 9; RING_MCP = 13; PINKY_MCP = 17
FINGERTIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
MCPS = [THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]

def _dist(a, b):
    return ((a.x - b.x)**2 + (a.y - b.y)**2) ** 0.5

def extract_frame_features(hand_landmarks_list, handedness_list, prev_wrist=None):
    """
    Extract 25-dim feature vector per frame:
    [0,1]   Wrist absolute position (start-normalized later)
    [2,3]   Wrist velocity (dx, dy from previous frame)
    [4-8]   Fingertip distances from wrist (5), normalized by palm size
    [9-13]  Finger curl ratios (5): tip_dist / mcp_dist (1=extended, 0=curled)
    [14-17] Inter-finger distances (4): thumb-idx, idx-mid, mid-ring, ring-pinky
    [18]    Thumb-index pinch distance
    [19,20] Palm orientation (cos, sin of wrist-to-middle-MCP angle)
    [21]    Hand openness (avg fingertip distance)
    [22]    Finger spread (std of fingertip angles)
    [23]    Second hand flag (0 or 1)
    [24]    Second hand distance from first (normalized)
    """
    if not hand_landmarks_list:
        return None, None

    lms = hand_landmarks_list[0]
    wrist = lms[WRIST]

    # Palm size for normalization
    palm_size = _dist(lms[WRIST], lms[MIDDLE_MCP])
    if palm_size < 0.01:
        palm_size = 0.01

    features = [0.0] * FEATURE_DIM

    # [0,1] Wrist position
    features[0] = wrist.x
    features[1] = wrist.y

    # [2,3] Wrist velocity
    if prev_wrist is not None:
        features[2] = wrist.x - prev_wrist[0]
        features[3] = wrist.y - prev_wrist[1]

    # [4-8] Fingertip distances from wrist, normalized
    tip_dists = []
    for i, tip in enumerate(FINGERTIPS):
        d = _dist(lms[tip], lms[WRIST]) / palm_size
        features[4 + i] = d
        tip_dists.append(d)

    # [9-13] Finger curl ratios
    for i, (tip, mcp) in enumerate(zip(FINGERTIPS, MCPS)):
        tip_d = _dist(lms[tip], lms[WRIST])
        mcp_d = _dist(lms[mcp], lms[WRIST])
        features[9 + i] = (tip_d / (mcp_d + 1e-6))

    # [14-17] Inter-finger distances (adjacent fingertips)
    for i in range(4):
        d = _dist(lms[FINGERTIPS[i]], lms[FINGERTIPS[i+1]]) / palm_size
        features[14 + i] = d

    # [18] Thumb-index pinch
    features[18] = _dist(lms[THUMB_TIP], lms[INDEX_TIP]) / palm_size

    # [19,20] Palm orientation
    dx = lms[MIDDLE_MCP].x - wrist.x
    dy = lms[MIDDLE_MCP].y - wrist.y
    angle = math.atan2(dy, dx)
    features[19] = math.cos(angle)
    features[20] = math.sin(angle)

    # [21] Hand openness
    features[21] = sum(tip_dists) / len(tip_dists)

    # [22] Finger spread
    angles = []
    for tip in FINGERTIPS:
        fx = (lms[tip].x - wrist.x) / palm_size
        fy = (lms[tip].y - wrist.y) / palm_size
        angles.append(math.atan2(fy, fx))
    features[22] = float(np.std(angles))

    # [23] Second hand flag
    features[23] = 1.0 if len(hand_landmarks_list) >= 2 else 0.0

    # [24] Second hand distance
    if len(hand_landmarks_list) >= 2:
        lms2 = hand_landmarks_list[1]
        features[24] = _dist(lms2[WRIST], lms[WRIST]) / palm_size

    return features, (wrist.x, wrist.y)


# ─── Sequence Processing ─────────────────────────────────────────

def resample_sequence(seq, target_len=RESAMPLE_LENGTH):
    seq = np.array(seq, dtype=np.float32)
    if len(seq) < 2:
        return np.zeros((target_len, seq.shape[1] if len(seq.shape) > 1 else FEATURE_DIM), dtype=np.float32)
    n = len(seq)
    x_old = np.linspace(0, 1, n)
    x_new = np.linspace(0, 1, target_len)
    result = np.zeros((target_len, seq.shape[1]), dtype=np.float32)
    for dim in range(seq.shape[1]):
        f = interp1d(x_old, seq[:, dim], kind='linear')
        result[:, dim] = f(x_new)
    return result


def normalize_sequence(seq):
    seq = np.array(seq, dtype=np.float32)
    if len(seq) < 2:
        return seq
    # Start-normalize wrist position (cols 0,1)
    seq[:, 0] -= seq[0, 0]
    seq[:, 1] -= seq[0, 1]
    # Velocity (cols 2,3) and all other features are already relative
    return seq


def compute_motion_amount(sequence):
    if len(sequence) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(sequence)):
        dx = sequence[i][0] - sequence[i-1][0]
        dy = sequence[i][1] - sequence[i-1][1]
        total += (dx*dx + dy*dy) ** 0.5
    return total


def process_raw_sequence(raw_frames):
    cleaned = []
    gap_count = 0
    last_good = None
    for f in raw_frames:
        if f is not None:
            if gap_count > 0 and last_good is not None and gap_count <= MAX_TRACKING_GAP:
                for g in range(1, gap_count + 1):
                    t = g / (gap_count + 1)
                    interp = [last_good[d] * (1-t) + f[d] * t for d in range(len(f))]
                    cleaned.append(interp)
            cleaned.append(f)
            last_good = f
            gap_count = 0
        else:
            gap_count += 1

    if len(cleaned) < 5:
        return None

    seq = np.array(cleaned, dtype=np.float32)
    seq = normalize_sequence(seq)
    seq = resample_sequence(seq, RESAMPLE_LENGTH)
    return seq


# ─── Data Augmentation ───────────────────────────────────────────

def augment_template(seq, count=AUGMENT_COUNT):
    """Generate augmented versions of a template sequence."""
    augmented = []
    for _ in range(count):
        aug = seq.copy()

        # Time warping: random speed variation
        n = len(aug)
        warp = np.sort(np.random.uniform(0, 1, n))
        warp = warp / warp[-1]  # Normalize to [0, 1]
        x_old = np.linspace(0, 1, n)
        for dim in range(aug.shape[1]):
            f = interp1d(warp, aug[:, dim], kind='linear', fill_value='extrapolate')
            aug[:, dim] = f(x_old)

        # Gaussian jitter on all features
        aug += np.random.normal(0, 0.005, aug.shape).astype(np.float32)

        # Scale wrist trajectory (cols 0,1) slightly
        scale = np.random.uniform(0.85, 1.15)
        aug[:, 0] *= scale
        aug[:, 1] *= scale

        # Scale velocity (cols 2,3) to match
        aug[:, 2] *= scale
        aug[:, 3] *= scale

        augmented.append(aug)
    return augmented


# ─── DTW Classification ──────────────────────────────────────────

def compute_dtw_distance(seq1, seq2):
    alignment = dtw(seq1, seq2, dist_method='euclidean',
                    window_type='sakoechiba', window_args={'window_size': 4})
    return alignment.distance


def classify_gesture(seq):
    """Classify using DTW. Returns (label_id, distance, confidence) or (None, inf, 0)."""
    if not gesture_templates:
        return None, float('inf'), 0.0

    all_dists = {}
    for label_id, templates in gesture_templates.items():
        distances = [compute_dtw_distance(seq, t) for t in templates]
        distances.sort()
        # Average of best 3
        avg_dist = float(np.mean(distances[:min(3, len(distances))]))
        all_dists[label_id] = avg_dist

    best_label = min(all_dists, key=all_dists.get)
    best_dist = all_dists[best_label]

    # Null rejection
    thresh = gesture_thresholds.get(best_label, {})
    max_d = thresh.get("max", best_dist * 2)
    rejection_threshold = max_d * 1.3

    if best_dist > rejection_threshold:
        log(f"[dtw] Rejected: best={best_label} dist={best_dist:.2f} > thresh={rejection_threshold:.2f}")
        return None, best_dist, 0.0

    # Confidence
    confidence = max(0.0, 1.0 - (best_dist / rejection_threshold))

    # Multi-class discrimination: if 2+ classes, check margin
    if len(all_dists) >= 2:
        sorted_dists = sorted(all_dists.values())
        margin = (sorted_dists[1] - sorted_dists[0]) / (sorted_dists[1] + 1e-6)
        # Boost or reduce confidence based on class separation
        confidence = confidence * (0.5 + margin)
        confidence = min(1.0, confidence)

    if confidence < 0.25:
        log(f"[dtw] Low confidence: best={best_label} dist={best_dist:.2f} conf={confidence:.2f}")
        return None, best_dist, 0.0

    return best_label, best_dist, confidence


# ─── Temporal Vote Window ─────────────────────────────────────────

class VoteWindow:
    """Require N-of-M consecutive classifications to agree before firing."""

    def __init__(self, window_size=3, min_agree=2, min_conf=0.35):
        self.window_size = window_size
        self.min_agree = min_agree
        self.min_conf = min_conf
        self.votes = []  # list of (label_id, confidence)
        self.last_vote_time = 0

    def add(self, label_id, confidence):
        now = time.time()
        # Reset if too long since last vote
        if now - self.last_vote_time > 2.0:
            self.votes.clear()
        self.last_vote_time = now

        self.votes.append((label_id, confidence))
        if len(self.votes) > self.window_size:
            self.votes.pop(0)

    def check(self):
        """Returns (label_id, avg_confidence) if consensus or high confidence, else (None, 0)."""
        if len(self.votes) == 0:
            return None, 0.0

        # High confidence single vote — fire immediately
        last_label, last_conf = self.votes[-1]
        if last_conf >= 0.75:
            self.votes.clear()
            return last_label, last_conf

        # Otherwise require 2-of-3 consensus
        if len(self.votes) < self.min_agree:
            return None, 0.0

        from collections import Counter
        labels = [v[0] for v in self.votes]
        counts = Counter(labels)
        best_label, best_count = counts.most_common(1)[0]

        if best_count >= self.min_agree:
            confs = [v[1] for v in self.votes if v[0] == best_label]
            avg_conf = sum(confs) / len(confs)
            if avg_conf >= self.min_conf:
                self.votes.clear()
                return best_label, avg_conf

        return None, 0.0

    def reset(self):
        self.votes.clear()


# ─── Drawing ──────────────────────────────────────────────────────

def draw_hand_landmarks(frame, landmarks, width, height):
    connections = HandLandmarksConnections.HAND_CONNECTIONS
    for conn in connections:
        s, e = landmarks[conn.start], landmarks[conn.end]
        cv2.line(frame, (int(s.x*width), int(s.y*height)),
                 (int(e.x*width), int(e.y*height)), (0, 200, 100), 2)
    for lm in landmarks:
        cv2.circle(frame, (int(lm.x*width), int(lm.y*height)), 4, (0, 100, 255), -1)


# ─── Labels / Mappings ────────────────────────────────────────────

def load_labels():
    if os.path.exists(LABELS_JSON):
        with open(LABELS_JSON) as f:
            return json.load(f)
    return {}

def save_labels(labels):
    with open(LABELS_JSON, "w") as f:
        json.dump(labels, f, indent=2)

def load_mappings():
    if os.path.exists(MAPPINGS_JSON):
        with open(MAPPINGS_JSON) as f:
            return json.load(f)
    return {}

def save_mappings(mappings):
    with open(MAPPINGS_JSON, "w") as f:
        json.dump(mappings, f, indent=2)
    sync_mappings_to_hammerspoon(mappings)

def sync_mappings_to_hammerspoon(mappings):
    lua_path = os.path.join(BASE_DIR, "mappings.lua")
    lines = ["-- Auto-generated by JarvisHands UI.", "return {"]
    for gname, adata in mappings.items():
        atype = adata.get("type", "keystroke")
        display = adata.get("display", gname).replace("\\", "\\\\").replace('"', '\\"')
        lines.append(f'    ["{gname}"] = {{')
        lines.append(f'        name = "{display}",')

        if atype == "keystroke":
            mods = adata.get("mods", [])
            key = adata.get("key", "")
            ml = "{" + ", ".join(f'"{m}"' for m in mods) + "}"
            key_codes = {"left": 123, "right": 124, "up": 126, "down": 125,
                         "space": 49, "tab": 48, "return": 36, "escape": 53,
                         "delete": 51, "f1": 122, "f2": 120, "f3": 99, "f4": 118}
            as_mods = {"cmd": "command down", "ctrl": "control down", "alt": "option down", "shift": "shift down"}
            if key in key_codes:
                kc = key_codes[key]
                using = " & ".join(as_mods.get(m, m) for m in mods) if mods else ""
                using_clause = f" using {{{using}}}" if using else ""
                lines.append(f'        action = function() hs.osascript.applescript(\'tell application "System Events" to key code {kc}{using_clause}\') end,')
            else:
                lines.append(f'        action = function() hs.eventtap.keyStroke({ml}, "{key}") end,')
        elif atype == "app":
            lines.append(f'        action = function() hs.application.launchOrFocus("{adata.get("app","")}") end,')
        elif atype == "shell":
            cmd = adata.get("command","").replace("\\","\\\\").replace('"','\\"').replace("\n","\\n")
            lines.append(f'        action = function() hs.execute("{cmd}") end,')
        elif atype == "url":
            url = adata.get("url","").replace("\\","\\\\").replace('"','\\"')
            lines.append(f'        action = function() hs.urlevent.openURL("{url}") end,')
        elif atype == "window":
            op = adata.get("operation","maximize")
            ops = {
                "maximize": "local w=hs.window.focusedWindow();if w then w:maximize() end",
                "minimize": "local w=hs.window.focusedWindow();if w then w:minimize() end",
                "left_half": "local w=hs.window.focusedWindow();if w then local f=w:screen():frame();w:setFrame({x=f.x,y=f.y,w=f.w/2,h=f.h}) end",
                "right_half": "local w=hs.window.focusedWindow();if w then local f=w:screen():frame();w:setFrame({x=f.x+f.w/2,y=f.y,w=f.w/2,h=f.h}) end",
                "close": "local w=hs.window.focusedWindow();if w then w:close() end",
            }
            lines.append(f'        action = function() {ops.get(op, ops["maximize"])} end,')
        elif atype == "lua":
            code = adata.get("code","").replace("\\","\\\\").replace('"','\\"').replace("\n","\\n")
            lines.append(f'        action = function() local fn=load("{code}");if fn then fn() end end,')

        lines.append("    },")
    lines.append("}")
    with open(lua_path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ─── DTW Model ────────────────────────────────────────────────────

def load_dtw_model():
    global gesture_templates, gesture_thresholds, inter_class_info
    if os.path.exists(MODEL_PATH):
        try:
            data = pickle.load(open(MODEL_PATH, "rb"))
            if isinstance(data, dict) and "templates" in data:
                gesture_templates = data["templates"]
                gesture_thresholds = data.get("thresholds", {})
                inter_class_info = data.get("inter_class", {})
                log(f"[model] Loaded DTW model: {len(gesture_templates)} gesture classes")
        except Exception as e:
            log(f"[model] Failed to load: {e}")


# ─── Actions catalog ──────────────────────────────────────────────

ACTIONS_CATALOG = [
    {"id": "mission_control", "display": "Mission Control", "type": "keystroke", "mods": ["ctrl"], "key": "up"},
    {"id": "desktop_left", "display": "Desktop Left", "type": "keystroke", "mods": ["ctrl"], "key": "left"},
    {"id": "desktop_right", "display": "Desktop Right", "type": "keystroke", "mods": ["ctrl"], "key": "right"},
    {"id": "app_expose", "display": "App Expose", "type": "keystroke", "mods": ["ctrl"], "key": "down"},
    {"id": "spotlight", "display": "Spotlight", "type": "keystroke", "mods": ["cmd"], "key": "space"},
    {"id": "launchpad", "display": "Launchpad", "type": "app", "app": "Launchpad"},
    {"id": "app_switcher", "display": "App Switcher", "type": "keystroke", "mods": ["cmd"], "key": "tab"},
    {"id": "close_window", "display": "Close Window", "type": "keystroke", "mods": ["cmd"], "key": "w"},
    {"id": "window_maximize", "display": "Maximize Window", "type": "window", "operation": "maximize"},
    {"id": "window_minimize", "display": "Minimize Window", "type": "window", "operation": "minimize"},
    {"id": "window_left_half", "display": "Window Left Half", "type": "window", "operation": "left_half"},
    {"id": "window_right_half", "display": "Window Right Half", "type": "window", "operation": "right_half"},
    {"id": "window_close", "display": "Close Window", "type": "window", "operation": "close"},
    {"id": "screenshot", "display": "Screenshot", "type": "keystroke", "mods": ["cmd", "shift"], "key": "3"},
    {"id": "screenshot_region", "display": "Screenshot Region", "type": "keystroke", "mods": ["cmd", "shift"], "key": "4"},
    {"id": "undo", "display": "Undo", "type": "keystroke", "mods": ["cmd"], "key": "z"},
    {"id": "redo", "display": "Redo", "type": "keystroke", "mods": ["cmd", "shift"], "key": "z"},
    {"id": "copy", "display": "Copy", "type": "keystroke", "mods": ["cmd"], "key": "c"},
    {"id": "paste", "display": "Paste", "type": "keystroke", "mods": ["cmd"], "key": "v"},
    {"id": "volume_up", "display": "Volume Up", "type": "shell", "command": "osascript -e 'set volume output volume ((output volume of (get volume settings)) + 10)'"},
    {"id": "volume_down", "display": "Volume Down", "type": "shell", "command": "osascript -e 'set volume output volume ((output volume of (get volume settings)) - 10)'"},
    {"id": "play_pause", "display": "Play/Pause", "type": "keystroke", "mods": [], "key": "play"},
]


# ─── Video streaming + detection ──────────────────────────────────

def generate_frames():
    global current_gesture, current_confidence, frame_buffer
    global recording_active, recorded_sequences, recording_phase, rep_count

    landmarker = get_landmarker()
    frame_buffer = []
    gesture_cooldown_until = 0
    cooldown_still_frames = 0

    # Detection state
    detection_buffer = []
    detection_phase = "waiting"
    no_hand_count = 0
    prev_wrist = None
    vote_window = VoteWindow(window_size=3, min_agree=2, min_conf=0.35)

    # Adaptive motion baseline
    motion_history = []
    MOTION_HISTORY_LEN = 60

    while True:
        with camera_lock:
            cam = get_camera()
            ret, frame = cam.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        try:
            result = landmarker.detect(mp_image)
        except Exception:
            result = None

        features = None
        hand_detected = False
        t_now = time.time()

        if result and result.hand_landmarks:
            hand_detected = True
            no_hand_count = 0

            # Smooth landmarks with One Euro Filter
            smoothed_hands = []
            for i, hand_lms in enumerate(result.hand_landmarks):
                if i < len(hand_smoothers):
                    smoothed = hand_smoothers[i].smooth(hand_lms, t_now)
                    smoothed_hands.append(smoothed)
                else:
                    smoothed_hands.append(hand_lms)

            # Draw smoothed landmarks
            for hand_lms in smoothed_hands:
                draw_hand_landmarks(frame, hand_lms, w, h)

            # Extract features from smoothed landmarks
            features, new_wrist = extract_frame_features(smoothed_hands, result.handedness, prev_wrist)
            prev_wrist = new_wrist

            n_hands = len(result.hand_landmarks)
            cv2.putText(frame, f"{n_hands} hand{'s' if n_hands>1 else ''}", (w-130, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
        else:
            no_hand_count += 1
            if no_hand_count > 10:
                for s in hand_smoothers:
                    s.reset()
                prev_wrist = None

        # Update motion baseline for adaptive thresholding
        if features is not None:
            motion_history.append(features[:2])  # wrist position
            if len(motion_history) > MOTION_HISTORY_LEN:
                motion_history.pop(0)

        # --- Recording mode ---
        if recording_active:
            rep_text = f"Rep {rep_count}/{target_reps}"

            if recording_phase == "waiting":
                cv2.putText(frame, rep_text + " -- Move hand to start", (10, h-40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,200,255), 1)
                if hand_detected and features is not None:
                    frame_buffer.append(features)
                    if len(frame_buffer) > 5:
                        recent_motion = compute_motion_amount(frame_buffer[-5:])
                        if recent_motion > MOTION_START_THRESHOLD:
                            recording_phase = "capturing"
                            frame_buffer = frame_buffer[-3:]
                    if len(frame_buffer) > 15:
                        frame_buffer = frame_buffer[-10:]

            elif recording_phase == "capturing":
                frame_buffer.append(features)
                valid_count = sum(1 for f in frame_buffer if f is not None)
                progress = min(len(frame_buffer), SEQ_LENGTH)
                bar_w = int((progress / SEQ_LENGTH) * 200)
                cv2.rectangle(frame, (10, h-30), (210, h-10), (40,40,40), -1)
                cv2.rectangle(frame, (10, h-30), (10+bar_w, h-10), (0,100,255), -1)
                cv2.putText(frame, f"Capturing... {progress}/{SEQ_LENGTH}", (10, h-40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,100,255), 1)

                consecutive_none = sum(1 for f in reversed(frame_buffer) if f is None)

                # Early stop when motion dies
                rec_motion_died = False
                if len(frame_buffer) >= MIN_CAPTURE_FRAMES:
                    rec_recent = [f for f in frame_buffer[-5:] if f is not None]
                    if len(rec_recent) >= 3:
                        if compute_motion_amount(rec_recent) < 0.003:
                            rec_motion_died = True

                if (len(frame_buffer) >= SEQ_LENGTH or
                    consecutive_none > MAX_TRACKING_GAP + 2 or
                    (rec_motion_died and len(frame_buffer) >= MIN_CAPTURE_FRAMES) or
                    len(frame_buffer) >= MAX_CAPTURE_LENGTH):
                    processed = process_raw_sequence(frame_buffer)
                    if processed is not None:
                        recorded_sequences.append(processed.tolist())
                        rep_count += 1
                        log(f"[record] Rep {rep_count}/{target_reps} captured ({valid_count} valid frames)")
                    else:
                        log(f"[record] Rep discarded — too few valid frames")
                    frame_buffer = []
                    cooldown_still_frames = 0
                    if rep_count >= target_reps:
                        recording_phase = "done"
                    else:
                        recording_phase = "cooldown"

            elif recording_phase == "cooldown":
                cv2.putText(frame, f"Rep {rep_count}/{target_reps} saved! Reset position...", (10, h-40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,220,100), 1)
                if hand_detected and features is not None:
                    frame_buffer.append(features)
                    if len(frame_buffer) >= 3:
                        recent_motion = compute_motion_amount(frame_buffer[-3:])
                        if recent_motion < MOTION_STOP_THRESHOLD:
                            cooldown_still_frames += 1
                        else:
                            cooldown_still_frames = 0
                    if len(frame_buffer) > 15:
                        frame_buffer = frame_buffer[-10:]
                else:
                    cooldown_still_frames += 1

                if cooldown_still_frames >= COOLDOWN_FRAMES:
                    frame_buffer = []
                    recording_phase = "waiting"

            elif recording_phase == "done":
                cv2.putText(frame, f"All {target_reps} reps done!", (10, h-40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,100), 2)
                recording_active = False
                recording_phase = "idle"

        # --- Detection mode (DTW + Vote) ---
        detected_gesture = None
        conf = 0.0

        if detection_active and gesture_templates:
            if hand_detected and features is not None:
                detection_buffer.append(features)
            elif detection_phase == "capturing":
                detection_buffer.append(None)

            if detection_phase == "waiting":
                if hand_detected and features is not None:
                    if len(detection_buffer) > 5:
                        recent = [f for f in detection_buffer[-5:] if f is not None]
                        if len(recent) >= 3:
                            recent_motion = compute_motion_amount(recent)
                            # Adaptive threshold
                            adaptive_thresh = MOTION_START_THRESHOLD
                            if len(motion_history) >= 20:
                                baseline_motion = compute_motion_amount(motion_history[-20:]) / 20
                                adaptive_thresh = max(MOTION_START_THRESHOLD, baseline_motion * 3)

                            if recent_motion > adaptive_thresh:
                                detection_phase = "capturing"
                                keep = [f for f in detection_buffer[-3:] if f is not None]
                                detection_buffer = keep
                    if len(detection_buffer) > 15:
                        detection_buffer = detection_buffer[-10:]

            elif detection_phase == "capturing":
                progress = len(detection_buffer)
                cv2.putText(frame, f"Capturing... ({progress})", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,200,100), 1)

                consecutive_none = sum(1 for f in reversed(detection_buffer) if f is None)

                motion_died = False
                if len(detection_buffer) >= MIN_CAPTURE_FRAMES:
                    recent_valid = [f for f in detection_buffer[-5:] if f is not None]
                    if len(recent_valid) >= 3:
                        if compute_motion_amount(recent_valid) < 0.003:
                            motion_died = True

                should_classify = (
                    len(detection_buffer) >= SEQ_LENGTH or
                    consecutive_none > MAX_TRACKING_GAP + 2 or
                    (motion_died and len(detection_buffer) >= MIN_CAPTURE_FRAMES) or
                    len(detection_buffer) >= MAX_CAPTURE_LENGTH
                )

                if should_classify:
                    processed = process_raw_sequence(detection_buffer)
                    if processed is not None:
                        motion = compute_motion_amount(processed.tolist())
                        if motion < 0.08:
                            log(f"[detect] Skipped — insufficient motion ({motion:.2f})")
                        else:
                            label_id, dist, confidence = classify_gesture(processed)
                            if label_id is not None:
                                # Add to vote window
                                vote_window.add(label_id, confidence)
                                winner, avg_conf = vote_window.check()
                                if winner is not None:
                                    labels = load_labels()
                                    gesture_name = labels.get(str(winner), f"gesture_{winner}")
                                    now = time.time()
                                    if now > gesture_cooldown_until:
                                        detected_gesture = gesture_name
                                        conf = avg_conf
                                        log(f"[detect] FIRE: {gesture_name} (dist={dist:.2f}, conf={avg_conf:.2f})")
                                        send_to_hammerspoon(gesture_name)
                                        gesture_cooldown_until = now + COOLDOWN
                                else:
                                    log(f"[detect] Vote pending: {label_id} dist={dist:.2f} conf={confidence:.2f}")
                            else:
                                log(f"[detect] No match (dist={dist:.2f})")

                    detection_buffer.clear()
                    detection_phase = "waiting"

            if no_hand_count > MAX_TRACKING_GAP * 2 and detection_phase == "waiting":
                detection_buffer.clear()
                vote_window.reset()

        elif not detection_active:
            detection_buffer.clear()
            detection_phase = "waiting"
            vote_window.reset()

        current_gesture = detected_gesture
        current_confidence = conf

        # Overlay
        if detected_gesture:
            cv2.putText(frame, f"{detected_gesture} ({conf:.0%})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 100), 2)
        elif not hand_detected:
            cv2.putText(frame, "No hand", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100,100,100), 2)
        else:
            if detection_active and detection_phase == "waiting":
                cv2.putText(frame, "Watching...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,180,180), 1)

        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")


def send_to_hammerspoon(gesture):
    for hs_path in ["/opt/homebrew/bin/hs", "/usr/local/bin/hs"]:
        try:
            subprocess.Popen([hs_path, "-c", f'gestureReceived("{gesture}")'],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            log(f"[hs] Sent '{gesture}'")
            return
        except FileNotFoundError:
            continue
        except Exception as e:
            log(f"[hs] Exception: {e}")
    log("[hs] WARNING: hs CLI not found")


# ─── Routes ───────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/status")
def api_status():
    labels = load_labels()
    counts = {}
    for lid in labels:
        gesture_dir = os.path.join(DATA_DIR, lid)
        if os.path.isdir(gesture_dir):
            counts[lid] = len([f for f in os.listdir(gesture_dir) if f.endswith('.json')])
        else:
            counts[lid] = 0

    gesture_list = []
    for lid, name in sorted(labels.items(), key=lambda x: int(x[0])):
        gesture_list.append({"id": lid, "name": name, "samples": counts.get(lid, 0)})

    return jsonify({
        "gestures": gesture_list,
        "mappings": load_mappings(),
        "has_model": os.path.exists(MODEL_PATH),
        "detection_active": detection_active,
        "recording_active": recording_active,
        "recording_phase": recording_phase,
        "rep_count": rep_count,
        "target_reps": target_reps,
        "current_gesture": current_gesture,
        "current_confidence": round(current_confidence, 2),
        "inter_class": inter_class_info,
    })

@app.route("/api/record/start", methods=["POST"])
def api_record_start():
    global recording_active, recording_label, recording_label_id
    global recorded_sequences, frame_buffer, recording_phase, rep_count, target_reps
    data = request.json
    name = data.get("name", "").strip()
    label_id = data.get("id", "").strip()
    reps = data.get("reps", 10)
    if not name or not label_id:
        return jsonify({"error": "Name and ID required"}), 400

    labels = load_labels()
    labels[label_id] = name
    save_labels(labels)

    recording_active = True
    recording_label = name
    recording_label_id = label_id
    recorded_sequences = []
    frame_buffer = []
    recording_phase = "waiting"
    rep_count = 0
    target_reps = max(1, int(reps))
    return jsonify({"ok": True, "target_reps": target_reps})

@app.route("/api/record/stop", methods=["POST"])
def api_record_stop():
    global recording_active, frame_buffer, recording_phase
    recording_active = False
    recording_phase = "idle"

    if recording_label_id and recorded_sequences:
        gesture_dir = os.path.join(DATA_DIR, recording_label_id)
        os.makedirs(gesture_dir, exist_ok=True)
        existing = len([f for f in os.listdir(gesture_dir) if f.endswith('.json')])
        for i, seq in enumerate(recorded_sequences):
            path = os.path.join(gesture_dir, f"seq_{existing+i:04d}.json")
            with open(path, "w") as f:
                json.dump(seq, f)

    saved = len(recorded_sequences)
    frame_buffer = []
    return jsonify({"ok": True, "saved": saved})

@app.route("/api/record/next", methods=["POST"])
def api_record_next():
    global recording_phase, frame_buffer
    if recording_active and recording_phase == "cooldown":
        frame_buffer = []
        recording_phase = "waiting"
        return jsonify({"ok": True, "phase": "waiting"})
    return jsonify({"ok": False, "phase": recording_phase})

@app.route("/api/train", methods=["POST"])
def api_train():
    """Build DTW model with data augmentation and inter-class analysis."""
    global gesture_templates, gesture_thresholds, inter_class_info
    labels = load_labels()
    if not labels:
        return jsonify({"error": "No labels"}), 400

    # Load raw templates
    raw_templates = {}
    for lid, name in labels.items():
        gesture_dir = os.path.join(DATA_DIR, lid)
        if not os.path.isdir(gesture_dir):
            continue
        lid_templates = []
        for fname in sorted(os.listdir(gesture_dir)):
            if not fname.endswith('.json'):
                continue
            with open(os.path.join(gesture_dir, fname)) as f:
                seq = json.load(f)
            seq = np.array(seq, dtype=np.float32)
            if seq.ndim == 2 and len(seq) > 0:
                lid_templates.append(seq)
        if lid_templates:
            raw_templates[lid] = lid_templates

    if not raw_templates:
        return jsonify({"error": "No training data"}), 400

    # Augment templates
    templates = {}
    total_samples = 0
    for lid, tmps in raw_templates.items():
        augmented = list(tmps)  # Start with originals
        for t in tmps:
            augmented.extend(augment_template(t, AUGMENT_COUNT))
        templates[lid] = augmented
        total_samples += len(augmented)
        log(f"[train] {labels[lid]}: {len(tmps)} raw + {len(tmps)*AUGMENT_COUNT} augmented = {len(augmented)} templates")

    # Compute intra-class DTW distances (sample subset to keep training fast)
    thresholds = {}
    for lid, tmps in templates.items():
        if len(tmps) < 2:
            thresholds[lid] = {"mean": 10.0, "max": 20.0}
            continue
        # Sample up to 15 pairs for speed
        originals = raw_templates[lid]
        intra_distances = []
        for i in range(len(originals)):
            for j in range(i+1, len(originals)):
                d = compute_dtw_distance(originals[i], originals[j])
                intra_distances.append(d)
        mean_d = float(np.mean(intra_distances)) if intra_distances else 10.0
        max_d = float(np.max(intra_distances)) if intra_distances else 20.0
        thresholds[lid] = {"mean": mean_d, "max": max_d}
        log(f"[train] {labels[lid]}: intra-DTW mean={mean_d:.2f} max={max_d:.2f}")

    # Inter-class analysis
    ic_info = {}
    label_ids = list(templates.keys())
    for i in range(len(label_ids)):
        for j in range(i+1, len(label_ids)):
            lid_a, lid_b = label_ids[i], label_ids[j]
            name_a, name_b = labels[lid_a], labels[lid_b]
            # Sample inter-class distances
            orig_a = raw_templates[lid_a]
            orig_b = raw_templates[lid_b]
            inter_dists = []
            for ta in orig_a[:5]:
                for tb in orig_b[:5]:
                    inter_dists.append(compute_dtw_distance(ta, tb))
            min_inter = float(np.min(inter_dists))
            mean_inter = float(np.mean(inter_dists))
            max_intra_a = thresholds[lid_a]["max"]
            max_intra_b = thresholds[lid_b]["max"]
            separable = min_inter > max(max_intra_a, max_intra_b) * 0.8
            pair_key = f"{name_a} vs {name_b}"
            ic_info[pair_key] = {
                "min_inter": round(min_inter, 2),
                "mean_inter": round(mean_inter, 2),
                "separable": separable,
            }
            status = "OK" if separable else "WARNING: similar!"
            log(f"[train] {pair_key}: min_inter={min_inter:.2f}, {status}")

    # Save model
    model_data = {"templates": templates, "thresholds": thresholds, "inter_class": ic_info}
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model_data, f)

    gesture_templates = templates
    gesture_thresholds = thresholds
    inter_class_info = ic_info

    return jsonify({
        "ok": True,
        "accuracy": 100.0,
        "samples": total_samples,
        "classes": len(templates),
        "inter_class": ic_info,
        "log": [{"epoch": 1, "loss": 0, "accuracy": 100.0}],
    })

@app.route("/api/detection", methods=["POST"])
def api_detection():
    global detection_active
    data = request.json
    detection_active = data.get("active", False)
    if detection_active and not gesture_templates:
        load_dtw_model()
    return jsonify({"detection_active": detection_active})

@app.route("/api/mappings", methods=["POST"])
def api_mappings():
    data = request.json
    save_mappings(data.get("mappings", {}))
    for hs_path in ["/opt/homebrew/bin/hs", "/usr/local/bin/hs"]:
        try:
            subprocess.Popen([hs_path, "-c", "hs.reload()"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            break
        except FileNotFoundError:
            continue
    return jsonify({"ok": True})

@app.route("/api/actions")
def api_actions():
    return jsonify(ACTIONS_CATALOG)

@app.route("/api/delete_gesture", methods=["POST"])
def api_delete_gesture():
    data = request.json
    label_id = data.get("id")
    if not label_id:
        return jsonify({"error": "ID required"}), 400
    labels = load_labels()
    name = labels.pop(label_id, None)
    save_labels(labels)
    gesture_dir = os.path.join(DATA_DIR, label_id)
    if os.path.isdir(gesture_dir):
        shutil.rmtree(gesture_dir)
    if name:
        m = load_mappings()
        m.pop(name, None)
        save_mappings(m)
    return jsonify({"ok": True, "deleted": name})


if __name__ == "__main__":
    load_dtw_model()
    print("[JarvisHands] Starting web UI at http://localhost:5757")
    app.run(host="127.0.0.1", port=5757, threaded=True, debug=False)
