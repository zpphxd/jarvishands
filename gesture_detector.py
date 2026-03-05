#!/usr/bin/env python3
"""
JarvisHands — Custom Hand Gesture Control for macOS via Hammerspoon.

Inspired by Kazuhito00/hand-gesture-recognition-using-mediapipe.
Uses MediaPipe for hand tracking + a simple MLP classifier for gesture recognition.

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

# --- MediaPipe ---
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def ensure_dirs():
    os.makedirs(BASE_DIR, exist_ok=True)


# ─── Landmark Processing ──────────────────────────────────────────

def preprocess_landmarks(landmarks):
    """
    Convert 21 hand landmarks to a normalized feature vector.
    1. Extract (x, y) for each landmark (drop z — unreliable)
    2. Translate to wrist-relative coordinates
    3. Normalize by max absolute value (scale invariant)
    Returns a flat list of 42 floats.
    """
    points = [(lm.x, lm.y) for lm in landmarks]

    # Relative to wrist
    wrist = points[0]
    relative = [(p[0] - wrist[0], p[1] - wrist[1]) for p in points]

    # Flatten
    flat = [coord for point in relative for coord in point]

    # Normalize by max absolute value
    max_val = max(abs(v) for v in flat)
    if max_val > 0:
        flat = [v / max_val for v in flat]

    return flat


# ─── Labels Management ────────────────────────────────────────────

def load_labels():
    """Load gesture labels. Returns dict: {id: name}"""
    if os.path.exists(LABELS_JSON):
        with open(LABELS_JSON) as f:
            return json.load(f)
    return {}


def save_labels(labels):
    with open(LABELS_JSON, "w") as f:
        json.dump(labels, f, indent=2)


# ─── MLP Classifier ───────────────────────────────────────────────
# Simple numpy-only MLP to avoid tensorflow/pytorch dependency.

class SimpleMLP:
    """A tiny 2-layer MLP using only numpy. Good enough for gesture classification."""

    def __init__(self, input_size=42, hidden_size=64, output_size=10):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        # Xavier initialization
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

    def train(self, X, y, epochs=300, lr=0.01, batch_size=32):
        """Train with mini-batch SGD + momentum."""
        n_samples = len(X)
        n_classes = self.output_size

        # One-hot encode
        Y = np.zeros((n_samples, n_classes))
        for i, label in enumerate(y):
            Y[i, label] = 1.0

        # Momentum
        vW1 = np.zeros_like(self.W1)
        vb1 = np.zeros_like(self.b1)
        vW2 = np.zeros_like(self.W2)
        vb2 = np.zeros_like(self.b2)
        vW3 = np.zeros_like(self.W3)
        vb3 = np.zeros_like(self.b3)
        momentum = 0.9

        for epoch in range(epochs):
            # Shuffle
            idx = np.random.permutation(n_samples)
            X_shuf = X[idx]
            Y_shuf = Y[idx]

            total_loss = 0.0

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                Xb = X_shuf[start:end]
                Yb = Y_shuf[start:end]
                bs = len(Xb)

                # Forward
                probs = self.forward(Xb)

                # Cross-entropy loss
                loss = -np.sum(Yb * np.log(probs + 1e-10)) / bs
                total_loss += loss * bs

                # Backward
                dz3 = (probs - Yb) / bs
                dW3 = self.a2.T @ dz3
                db3 = np.sum(dz3, axis=0)

                da2 = dz3 @ self.W3.T
                dz2 = da2 * (self.z2 > 0)
                dW2 = self.a1.T @ dz2
                db2 = np.sum(dz2, axis=0)

                da1 = dz2 @ self.W2.T
                dz1 = da1 * (self.z1 > 0)
                dW1 = Xb.T @ dz1
                db1 = np.sum(dz1, axis=0)

                # Update with momentum
                vW3 = momentum * vW3 - lr * dW3
                vb3 = momentum * vb3 - lr * db3
                vW2 = momentum * vW2 - lr * dW2
                vb2 = momentum * vb2 - lr * db2
                vW1 = momentum * vW1 - lr * dW1
                vb1 = momentum * vb1 - lr * db1

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


# ─── Hammerspoon IPC ──────────────────────────────────────────────

def send_to_hammerspoon(gesture):
    for hs_path in ["/opt/homebrew/bin/hs", "/usr/local/bin/hs"]:
        try:
            subprocess.Popen(
                [hs_path, "-c", f'gestureReceived("{gesture}")'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return
        except FileNotFoundError:
            continue
    print(f"[jarvishands] hs CLI not found, gesture: {gesture}", flush=True)


# ─── Record Mode ──────────────────────────────────────────────────

def record_mode():
    """
    Interactive recording: camera shows live hand tracking.
    Press a number key (0-9) to assign a label to the current hand pose.
    First time a number is pressed, you'll be prompted to name it.
    """
    ensure_dirs()
    labels = load_labels()

    print("[jarvishands] RECORD MODE")
    print("  Show a gesture to the camera, then press a number key (0-9) to label it.")
    print("  Each key press saves one sample. Record 20-50 samples per gesture.")
    print("  Press 'q' to quit.")
    print()

    # Show existing label assignments
    if labels:
        print("  Existing labels:")
        for k, v in sorted(labels.items(), key=lambda x: int(x[0])):
            print(f"    {k} = {v}")
    print()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    )

    # Count existing samples per label
    sample_counts = {}
    if os.path.exists(DATA_CSV):
        with open(DATA_CSV) as f:
            reader = csv.reader(f)
            for row in reader:
                if row:
                    lid = row[0]
                    sample_counts[lid] = sample_counts.get(lid, 0) + 1

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            current_vector = None

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                current_vector = preprocess_landmarks(hand_landmarks.landmark)

            # Draw UI
            hand_status = "Hand detected" if current_vector else "No hand"
            color = (0, 255, 0) if current_vector else (0, 0, 255)
            cv2.putText(frame, f"RECORD MODE | {hand_status}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Show label assignments
            y_offset = 60
            for k, v in sorted(labels.items(), key=lambda x: int(x[0])):
                count = sample_counts.get(k, 0)
                cv2.putText(frame, f"[{k}] {v}: {count} samples", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                y_offset += 22

            cv2.putText(frame, "Press 0-9 to save, 'q' to quit", (10, 460),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("JarvisHands - Record", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            # Number keys 0-9
            if ord("0") <= key <= ord("9") and current_vector is not None:
                label_id = str(key - ord("0"))

                # If new label, prompt for name
                if label_id not in labels:
                    cv2.destroyAllWindows()
                    name = input(f"  Name for gesture [{label_id}]: ").strip()
                    if not name:
                        name = f"gesture_{label_id}"
                    labels[label_id] = name
                    save_labels(labels)
                    print(f"  Assigned [{label_id}] = '{name}'")

                # Save sample to CSV
                with open(DATA_CSV, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([label_id] + current_vector)

                sample_counts[label_id] = sample_counts.get(label_id, 0) + 1
                count = sample_counts[label_id]
                print(f"  Saved [{label_id}] {labels[label_id]}: sample #{count}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        print(f"\n[jarvishands] Recording complete. Run --train to build the model.")


# ─── Train Mode ───────────────────────────────────────────────────

def train_mode():
    """Train the MLP classifier on recorded data."""
    if not os.path.exists(DATA_CSV):
        print("[jarvishands] No training data found. Run --record first.")
        sys.exit(1)

    labels = load_labels()
    if not labels:
        print("[jarvishands] No labels found. Run --record first.")
        sys.exit(1)

    # Load data
    X = []
    y = []
    with open(DATA_CSV) as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            label_id = int(row[0])
            features = [float(v) for v in row[1:]]
            X.append(features)
            y.append(label_id)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    n_classes = max(y) + 1
    print(f"[jarvishands] Training on {len(X)} samples, {n_classes} classes")
    for lid, name in sorted(labels.items(), key=lambda x: int(x[0])):
        count = np.sum(y == int(lid))
        print(f"  [{lid}] {name}: {count} samples")

    # Train
    model = SimpleMLP(input_size=42, hidden_size=64, output_size=n_classes)
    model.train(X, y, epochs=500, lr=0.005, batch_size=32)

    # Final accuracy
    preds, _ = model.predict(X)
    acc = np.mean(preds == y) * 100
    print(f"\n[jarvishands] Final training accuracy: {acc:.1f}%")

    # Save
    model.save(MODEL_PATH)
    print(f"[jarvishands] Model saved to {MODEL_PATH}")


# ─── Run Mode ─────────────────────────────────────────────────────

def run_mode(show_preview=False):
    """Live gesture detection."""
    if not os.path.exists(MODEL_PATH):
        print("[jarvishands] No trained model. Run --train first.", flush=True)
        sys.exit(1)

    model = SimpleMLP.load(MODEL_PATH)
    labels = load_labels()

    if not labels:
        print("[jarvishands] No labels found.", flush=True)
        sys.exit(1)

    print(f"[jarvishands] Loaded model with gestures: {', '.join(labels.values())}", flush=True)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    )

    COOLDOWN = 1.0
    STABLE_FRAMES = 6
    CONFIDENCE_THRESHOLD = 0.85
    last_gesture_time = {}
    stable_gesture = None
    stable_count = 0

    print("[jarvishands] Detecting gestures...", flush=True)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            current_gesture = None
            confidence = 0.0

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                vector = preprocess_landmarks(hand_landmarks.landmark)
                X_input = np.array([vector], dtype=np.float32)
                pred, conf = model.predict(X_input)
                pred_id = str(pred[0])
                confidence = conf[0]

                if confidence >= CONFIDENCE_THRESHOLD and pred_id in labels:
                    current_gesture = labels[pred_id]

                if show_preview:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Stability
            if current_gesture and current_gesture == stable_gesture:
                stable_count += 1
            else:
                stable_gesture = current_gesture
                stable_count = 1 if current_gesture else 0

            # Fire
            if stable_count == STABLE_FRAMES and current_gesture:
                now = time.time()
                last = last_gesture_time.get(current_gesture, 0)
                if now - last >= COOLDOWN:
                    print(f"[jarvishands] {current_gesture} (conf={confidence:.2f})", flush=True)
                    send_to_hammerspoon(current_gesture)
                    last_gesture_time[current_gesture] = now

            if show_preview:
                label = current_gesture or "none"
                cv2.putText(frame, f"{label} ({confidence:.0%}) [{stable_count}/{STABLE_FRAMES}]",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("JarvisHands - Live", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        pass
    finally:
        print("[jarvishands] Shutting down...", flush=True)
        cap.release()
        if show_preview:
            cv2.destroyAllWindows()
        hands.close()


# ─── List / Delete ─────────────────────────────────────────────────

def list_gestures():
    labels = load_labels()
    if not labels:
        print("No gestures recorded yet.")
        return

    # Count samples per label
    counts = {}
    if os.path.exists(DATA_CSV):
        with open(DATA_CSV) as f:
            for row in csv.reader(f):
                if row:
                    counts[row[0]] = counts.get(row[0], 0) + 1

    print("Recorded gestures:")
    for lid, name in sorted(labels.items(), key=lambda x: int(x[0])):
        c = counts.get(lid, 0)
        print(f"  [{lid}] {name}: {c} samples")

    has_model = os.path.exists(MODEL_PATH)
    print(f"\nModel trained: {'yes' if has_model else 'no — run --train'}")


def delete_gesture(name):
    labels = load_labels()
    target_id = None
    for lid, lname in labels.items():
        if lname == name:
            target_id = lid
            break

    if target_id is None:
        print(f"Gesture '{name}' not found.")
        return

    # Remove from labels
    del labels[target_id]
    save_labels(labels)

    # Remove from CSV
    if os.path.exists(DATA_CSV):
        rows = []
        with open(DATA_CSV) as f:
            for row in csv.reader(f):
                if row and row[0] != target_id:
                    rows.append(row)
        with open(DATA_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

    print(f"Deleted gesture '{name}' (id={target_id}). Run --train to rebuild model.")


# ─── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="JarvisHands - Custom Hand Gesture Control")
    parser.add_argument("--record", action="store_true", help="Record gestures interactively")
    parser.add_argument("--train", action="store_true", help="Train the classifier")
    parser.add_argument("--run", action="store_true", help="Run live gesture detection")
    parser.add_argument("--preview", action="store_true", help="Show camera preview (with --run)")
    parser.add_argument("--list", action="store_true", help="List recorded gestures")
    parser.add_argument("--delete", metavar="NAME", help="Delete a gesture by name")
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
