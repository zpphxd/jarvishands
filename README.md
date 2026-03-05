# JarvisHands

Custom hand gesture recognition for macOS control via Hammerspoon.

Record your own gestures, train a classifier, map them to any macOS action.

Uses MediaPipe for hand tracking and a lightweight MLP neural network (pure numpy, no TensorFlow needed) for gesture classification. Inspired by [Kazuhito00/hand-gesture-recognition-using-mediapipe](https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe).

## Setup

```bash
pip3 install -r requirements.txt
```

Copy `gestures.lua` to `~/.hammerspoon/` and add to your `init.lua`:
```lua
local gestures = require("gestures")
gestures.bindHotkeys()
```

## Workflow

### 1. Record gestures

```bash
python3 gesture_detector.py --record
```

A camera window opens with your hand tracked. Press number keys (0-9) to label the current hand pose. First time you press a number, you'll name the gesture. Record **20-50 samples** per gesture with slight variations (angle, distance, hand rotation).

### 2. Train the model

```bash
python3 gesture_detector.py --train
```

Trains a 3-layer MLP on your recorded data. Takes a few seconds.

### 3. Map gestures to actions

Edit `~/.hammerspoon/gestures.lua` — add entries to `M.mappings`:

```lua
M.mappings = {
    open_palm = {
        name = "Mission Control",
        action = function() hs.eventtap.keyStroke({"ctrl"}, "up") end,
    },
    fist = {
        name = "Minimize",
        action = function()
            local win = hs.window.focusedWindow()
            if win then win:minimize() end
        end,
    },
    point_right = {
        name = "Desktop Right",
        action = function() hs.eventtap.keyStroke({"ctrl"}, "right") end,
    },
}
```

Keys must match the gesture names you chose during recording.

### 4. Activate

Press **Ctrl+Cmd+G** to toggle gesture mode on/off.

### Other commands

```bash
python3 gesture_detector.py --list              # List recorded gestures
python3 gesture_detector.py --run --preview      # Test live with camera window
python3 gesture_detector.py --delete <name>      # Remove a gesture
```

## How it works

1. MediaPipe detects 21 hand landmarks from the webcam
2. Landmarks are normalized (wrist-relative, scale-invariant) into a 42-dim vector
3. A simple MLP classifier (numpy-only, no heavy ML frameworks) predicts the gesture
4. Gestures must be held stable for ~6 frames and exceed 85% confidence to trigger
5. Recognized gestures are sent to Hammerspoon via the `hs` CLI (IPC)
6. Hammerspoon executes the mapped action

## Data storage

All data lives in `~/.jarvishands/`:
- `keypoints.csv` — training samples (label_id + 42 normalized landmark values)
- `labels.json` — mapping of label IDs to gesture names
- `model.pkl` — trained MLP model
