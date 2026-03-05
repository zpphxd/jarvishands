#!/usr/bin/env python3
"""
JarvisHands — Custom Hand Gesture Control for macOS via Hammerspoon.

Uses the MediaPipe Tasks API (HandLandmarker) + a simple MLP classifier.

Modes:
  --record              Interactive recording mode — press 0-9 to label gestures live
  --train               Train the MLP classifier on recorded data
  --run                 Live detection, sends gestures to Hammerspoon
  --run --preview       Live detection with camera preview
  --list                List recorded gestures
  --delete <name>       Delete a gesture's training data

Gesture data: ~/.jarvishands/
"""

import argparse
import csv
import json
import math
import os
import pickle
import subprocess
import sys
import time

import cv2
import mediapipe as mp
import numpy as np

# --- Paths ---
BASE_DIR = os.path.expanduser("~/.jarvishands")
DATA_CSV = os.path.join(BASE_DIR, "keypoints.csv")
LABELS_JSON = os.path.join(BASE_DIR, "labels.json")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HAND_MODEL_PATH = os.path.join(SCRIPT_DIR, "hand_landmarker.task")

# --- MediaPipe Tasks API ---
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
HandLandmarksConnections = mp.tasks.vision.HandLandmarksConnections


def ensure_dirs():
    os.makedirs(BASE_DIR, exist_ok=True)


def create_landmarker():
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    return HandLandmarker.create_from_options(options)


def preprocess_landmarks(landmarks):
    """Normalize hand landmarks to wrist-relative, scale-invariant 42-dim vector."""
    points = [(lm.x, lm.y) for lm in landmarks]
    wrist = points[0]
    relative = [(p[0] - wrist[0], p[1] - wrist[1]) for p in points]
    flat = [coord for point in relative for coord in point]
    max_val = max(abs(v) for v in flat)
    if max_val > 0:
        flat = [v / max_val for v in flat]
    return flat


def draw_hand_landmarks(frame, landmarks, width, height):
    connections = HandLandmarksConnections.HAND_CONNECTIONS
    for conn in connections:
        start = landmarks[conn.start]
        end = landmarks[conn.end]
        sx, sy = int(start.x * width), int(start.y * height)
        ex, ey = int(end.x * width), int(end.y * height)
        cv2.line(frame, (sx, sy), (ex, ey), (0, 200, 100), 2)
    for lm in landmarks:
        cx, cy = int(lm.x * width), int(lm.y * height)
        cv2.circle(frame, (cx, cy), 4, (0, 100, 255), -1)


# --- Labels ---
def load_labels():
    if os.path.exists(LABELS_JSON):
        with open(LABELS_JSON) as f:
            return json.load(f)
    return {}


def save_labels(labels):
    with open(LABELS_JSON, "w") as f:
        json.dump(labels, f, indent=2)


# --- MLP Classifier ---
class SimpleMLP:
    def __init__(self, input_size=42, hidden_size=64, output_size=10):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros(hidden_size)
        self.W3 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b3 = np.zeros(output_size)

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        e = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e / np.sum(e, axis=-1, keepdims=True)

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.relu(self.z2)
        self.z3 = self.a2 @ self.W3 + self.b3
        return self.softmax(self.z3)

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=-1), np.max(probs, axis=-1)

    def train(self, X, y, epochs=500, lr=0.005, batch_size=32):
        n_samples = len(X)
        n_classes = self.output_size
        Y = np.zeros((n_samples, n_classes))
        for i, label in enumerate(y):
            Y[i, label] = 1.0
        vW1, vb1 = np.zeros_like(self.W1), np.zeros_like(self.b1)
        vW2, vb2 = np.zeros_like(self.W2), np.zeros_like(self.b2)
        vW3, vb3 = np.zeros_like(self.W3), np.zeros_like(self.b3)
        mom = 0.9

        for epoch in range(epochs):
            idx = np.random.permutation(n_samples)
            X_s, Y_s = X[idx], Y[idx]
            total_loss = 0.0
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                Xb, Yb = X_s[start:end], Y_s[start:end]
                bs = len(Xb)
                probs = self.forward(Xb)
                loss = -np.sum(Yb * np.log(probs + 1e-10)) / bs
                total_loss += loss * bs
                dz3 = (probs - Yb) / bs
                dW3 = self.a2.T @ dz3; db3 = np.sum(dz3, axis=0)
                da2 = dz3 @ self.W3.T; dz2 = da2 * (self.z2 > 0)
                dW2 = self.a1.T @ dz2; db2 = np.sum(dz2, axis=0)
                da1 = dz2 @ self.W2.T; dz1 = da1 * (self.z1 > 0)
                dW1 = Xb.T @ dz1; db1 = np.sum(dz1, axis=0)
                vW3 = mom*vW3 - lr*dW3; vb3 = mom*vb3 - lr*db3
                vW2 = mom*vW2 - lr*dW2; vb2 = mom*vb2 - lr*db2
                vW1 = mom*vW1 - lr*dW1; vb1 = mom*vb1 - lr*db1
                self.W3 += vW3; self.b3 += vb3
                self.W2 += vW2; self.b2 += vb2
                self.W1 += vW1; self.b1 += vb1
            avg_loss = total_loss / n_samples
            if (epoch + 1) % 50 == 0 or epoch == 0:
                preds, _ = self.predict(X)
                acc = np.mean(preds == y) * 100
                print(f"  Epoch {epoch+1}/{epochs} — loss: {avg_loss:.4f}, acc: {acc:.1f}%")

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)


# --- Hammerspoon IPC ---
def send_to_hammerspoon(gesture):
    for hs_path in ["/opt/homebrew/bin/hs", "/usr/local/bin/hs"]:
        try:
            subprocess.Popen(
                [hs_path, "-c", f'gestureReceived("{gesture}")'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            return
        except FileNotFoundError:
            continue
    print(f"[jarvishands] hs CLI not found, gesture: {gesture}", flush=True)


# ─── Record Mode ──────────────────────────────────────────────────
def record_mode():
    ensure_dirs()
    labels = load_labels()

    print("[jarvishands] RECORD MODE")
    print("  Show a gesture, press 0-9 to label it. Press 'q' to quit.")
    if labels:
        print("  Existing labels:")
        for k, v in sorted(labels.items(), key=lambda x: int(x[0])):
            print(f"    {k} = {v}")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    landmarker = create_landmarker()

    sample_counts = {}
    if os.path.exists(DATA_CSV):
        with open(DATA_CSV) as f:
            for row in csv.reader(f):
                if row:
                    sample_counts[row[0]] = sample_counts.get(row[0], 0) + 1

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            current_vector = None
            try:
                result = landmarker.detect(mp_image)
                if result.hand_landmarks:
                    landmarks = result.hand_landmarks[0]
                    draw_hand_landmarks(frame, landmarks, w, h)
                    current_vector = preprocess_landmarks(landmarks)
            except Exception:
                pass

            status = "Hand detected" if current_vector else "No hand"
            color = (0, 255, 0) if current_vector else (0, 0, 255)
            cv2.putText(frame, f"RECORD | {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            y_off = 60
            for k, v in sorted(labels.items(), key=lambda x: int(x[0])):
                c = sample_counts.get(k, 0)
                cv2.putText(frame, f"[{k}] {v}: {c}", (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                y_off += 22

            cv2.imshow("JarvisHands - Record", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if ord("0") <= key <= ord("9") and current_vector is not None:
                lid = str(key - ord("0"))
                if lid not in labels:
                    cv2.destroyAllWindows()
                    name = input(f"  Name for gesture [{lid}]: ").strip()
                    if not name:
                        name = f"gesture_{lid}"
                    labels[lid] = name
                    save_labels(labels)
                with open(DATA_CSV, "a", newline="") as f:
                    csv.writer(f).writerow([lid] + current_vector)
                sample_counts[lid] = sample_counts.get(lid, 0) + 1
                print(f"  [{lid}] {labels[lid]}: #{sample_counts[lid]}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()
        print("[jarvishands] Done. Run --train to build model.")


# ─── Train Mode ───────────────────────────────────────────────────
def train_mode():
    if not os.path.exists(DATA_CSV):
        print("[jarvishands] No data. Run --record first.")
        sys.exit(1)
    labels = load_labels()
    X, y = [], []
    with open(DATA_CSV) as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                X.append([float(v) for v in row[1:]])
                y.append(int(row[0]))
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    n_classes = max(y) + 1

    print(f"[jarvishands] Training: {len(X)} samples, {n_classes} classes")
    model = SimpleMLP(input_size=42, hidden_size=64, output_size=n_classes)
    model.train(X, y)
    preds, _ = model.predict(X)
    print(f"[jarvishands] Final acc: {np.mean(preds == y) * 100:.1f}%")
    model.save(MODEL_PATH)
    print(f"[jarvishands] Model saved.")


# ─── Run Mode ─────────────────────────────────────────────────────
def run_mode(show_preview=False):
    if not os.path.exists(MODEL_PATH):
        print("[jarvishands] No model. Run --train first.", flush=True)
        sys.exit(1)

    model = SimpleMLP.load(MODEL_PATH)
    labels = load_labels()
    print(f"[jarvishands] Gestures: {', '.join(labels.values())}", flush=True)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    landmarker = create_landmarker()

    COOLDOWN = 1.0
    STABLE_FRAMES = 6
    CONF_THRESHOLD = 0.85
    last_time = {}
    stable_gesture = None
    stable_count = 0

    print("[jarvishands] Detecting...", flush=True)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            gesture = None
            conf = 0.0

            try:
                result = landmarker.detect(mp_image)
                if result.hand_landmarks:
                    landmarks = result.hand_landmarks[0]
                    vector = preprocess_landmarks(landmarks)
                    X_in = np.array([vector], dtype=np.float32)
                    pred, confidence = model.predict(X_in)
                    pid = str(pred[0])
                    conf = float(confidence[0])
                    if conf >= CONF_THRESHOLD and pid in labels:
                        gesture = labels[pid]
                    if show_preview:
                        draw_hand_landmarks(frame, landmarks, w, h)
            except Exception:
                pass

            if gesture and gesture == stable_gesture:
                stable_count += 1
            else:
                stable_gesture = gesture
                stable_count = 1 if gesture else 0

            if stable_count == STABLE_FRAMES and gesture:
                now = time.time()
                if now - last_time.get(gesture, 0) >= COOLDOWN:
                    print(f"[jarvishands] {gesture} ({conf:.0%})", flush=True)
                    send_to_hammerspoon(gesture)
                    last_time[gesture] = now

            if show_preview:
                label = gesture or "none"
                cv2.putText(frame, f"{label} ({conf:.0%}) [{stable_count}/{STABLE_FRAMES}]",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("JarvisHands", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if show_preview:
            cv2.destroyAllWindows()
        landmarker.close()


# ─── List / Delete ─────────────────────────────────────────────────
def list_gestures():
    labels = load_labels()
    if not labels:
        print("No gestures recorded.")
        return
    counts = {}
    if os.path.exists(DATA_CSV):
        with open(DATA_CSV) as f:
            for row in csv.reader(f):
                if row:
                    counts[row[0]] = counts.get(row[0], 0) + 1
    for lid, name in sorted(labels.items(), key=lambda x: int(x[0])):
        print(f"  [{lid}] {name}: {counts.get(lid, 0)} samples")
    print(f"Model: {'yes' if os.path.exists(MODEL_PATH) else 'no'}")


def delete_gesture(name):
    labels = load_labels()
    tid = None
    for lid, lname in labels.items():
        if lname == name:
            tid = lid
            break
    if not tid:
        print(f"'{name}' not found.")
        return
    del labels[tid]
    save_labels(labels)
    if os.path.exists(DATA_CSV):
        rows = [r for r in csv.reader(open(DATA_CSV)) if r and r[0] != tid]
        with open(DATA_CSV, "w", newline="") as f:
            csv.writer(f).writerows(rows)
    print(f"Deleted '{name}'. Run --train to rebuild.")


def main():
    parser = argparse.ArgumentParser(description="JarvisHands")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--delete", metavar="NAME")
    args = parser.parse_args()

    if args.record:
        record_mode()
    elif args.train:
        train_mode()
    elif args.run:
        run_mode(show_preview=args.preview)
    elif args.list:
        list_gestures()
    elif args.delete:
        delete_gesture(args.delete)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
