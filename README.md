# Real-time Iranian Sign Language Recognition (ISLR101)

A real-time Iranian Sign Language recognition system using a webcam.

The main model is trained with a **BiGRU** architecture on the ISLR101 dataset (101 classes).  
In real-time mode, **MediaPipe** is used to extract body and hand keypoints, which are then mapped to the trained model's input space through calibration.

---

## Features

- Real-time Iranian Sign Language recognition
- Support for 101 classes
- Automatic calibration to align MediaPipe features with the training data space
- Display results in Persian on the video frame
- Manual control of sign start/end (using `m` and `SPACE` keys)

---

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME

# Create a virtual environment (recommended)
python -m venv venv

# Windows:
venv\Scripts\activate

# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

---

## Project Structure

├── model.py                  # BiGRU model architecture
├── preprocessing.py          # Preprocessing and DataLoader
├── keypoint_extractor.py     # Keypoint extraction using MediaPipe
├── calibration.py            # Feature space calibration
├── input_spec.py             # Model input specifications
├── train.py                  # Model training
├── evaluate.py               # Evaluation
├── test.py                   # Simple testing
├── realtime_inference.py     # Real-time inference engine
├── realtime_demo.py          # Live demo with webcam
├── audit_phase0.py           # Data inspection
└── fix_t5_phase1.py          # MediaPipe Tasks API test

---

## How to Use (Real-time)

### 1. Calibration (only once)

python calibration.py

This script generates the checkpoints/calibration.json file.

### 2. Offline Test (optional)

python realtime_inference.py

### 3. Run Live Demo

python realtime_demo.py

Important notes while using:

- Lower your hands → perform the sign → lower your hands again.
- If automatic start/end detection is not working well, press the `m` key to switch to manual mode, then use `SPACE` to mark the start and end of each sign.
- If your webcam feed is mirrored, change this line inside realtime_demo.py:

FRAME_IS_MIRRORED = True

---

## Requirements for Real-time

- Webcam
- MediaPipe models (downloaded automatically on first run)
- Trained model checkpoint files (must be placed in the checkpoints/ folder)

---

## License

This project is released for educational and research purposes.
```
