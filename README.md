# Hand Tracking with OpenCV & MediaPipe

A simple hand tracking project built while learning OpenCV and MediaPipe.

## What it does

- Opens your webcam feed
- Detects hands in real-time using MediaPipe's HandLandmarker
- Draws 21 landmark points and skeleton connections on detected hands
- Supports tracking up to 2 hands simultaneously

## Requirements

```bash
pip install opencv-python mediapipe
```

## Usage

```bash
python3 open.py
```

Press `q` to quit.

## Files

- `open.py` - Main script
- `hand_landmarker.task` - MediaPipe hand detection model

