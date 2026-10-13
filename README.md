# Driver Drowsiness Monitoring System

### AI-Powered Real-Time Driver Monitoring Using Computer Vision & Deep Learning

## Overview

The **Driver Drowsiness Monitoring System** is an AI-powered application that monitors driver alertness in real time using a webcam. It combines a custom-trained Convolutional Neural Network (CNN), MediaPipe facial landmarks, and geometric feature extraction to detect prolonged eye closure, yawning, and hand movements near the mouth.

The application also includes adaptive lighting detection, low-light image enhancement, browser-based voice alerts, and an interactive Streamlit dashboard with historical analytics.

## Features

- Real-time webcam monitoring
- CNN-based open/closed eye classification
- Yawning detection using CNN and Mouth Aspect Ratio (MAR)
- Hybrid CNN + EAR/MAR detection pipeline
- Hand-on-mouth distraction detection
- Automatic Bright, Normal, Dim, and Night classification
- CLAHE-based low-light image enhancement
- Browser-based voice alerts
- Live EAR/MAR metrics and trend charts
- Historical analytics and lighting-wise evaluation
- CSV-based alert logging and download
- Batched CNN inference and cached predictions
- Interactive dashboard with pause/resume controls

## Tech Stack

| Technology | Usage |
|---|---|
| Python | Core application |
| TensorFlow / Keras | CNN inference |
| OpenCV | Webcam and image processing |
| MediaPipe | Face and hand landmark detection |
| NumPy | Numerical calculations |
| Pandas | Data processing and analytics |
| Streamlit | Interactive dashboard |
| Matplotlib | Evaluation charts |
| HTML, CSS, JavaScript | Styling and voice alerts |

## Detection Methodology

### 1. Eye Aspect Ratio (EAR)

EAR estimates eye openness using facial landmarks. A lower value generally indicates a more closed eye.

**Configured threshold:** `0.21`

Prolonged eye closure triggers a drowsiness warning.

### 2. Mouth Aspect Ratio (MAR)

MAR measures mouth openness using vertical and horizontal facial landmark distances.

**Configured threshold:** `0.60`

Sustained mouth-opening predictions can trigger a yawning warning.

### 3. CNN-Based Classification

The trained CNN analyzes separate square eye crops and a face crop.

- `Open` — eyes appear open
- `Closed` — eyes appear closed
- `yawn` — yawning detected
- `no_yawn` — yawning not detected

### 4. Hybrid Detection

CNN predictions are combined with EAR/MAR measurements. Geometric checks provide fallback decisions when CNN confidence is low and can override certain contradictory predictions.

### 5. Hand Detection

MediaPipe Hands checks whether hand landmarks overlap the detected mouth region. Sustained hand presence can trigger a distraction warning.

### 6. Adaptive Lighting

| Zone | Grayscale Brightness |
|---|---:|
| ☀️ Bright | ≥ 170 |
| 🌤️ Normal | 90–169 |
| 🌥️ Dim | 40–89 |
| 🌙 Night | < 40 |

CLAHE enhancement is applied automatically in Dim and Night conditions.

## Dashboard

### Live Monitoring

- Live webcam feed and detection overlays
- Driver status and lighting alerts
- EAR and MAR metric cards
- Session duration and alert counters
- Voice-alert controls
- Recent event history

### Analytics & History

- Saved model-comparison results
- Accuracy and latency charts
- Detailed metrics when available
- Batch evaluation under simulated lighting
- Full alert log
- CSV download

## 📸 Screenshots

Add your actual application screenshots to the `screenshots/` directory, then replace the image paths below with the filenames you use.

### Live Dashboard

<img width="1920" height="2339" alt="LiveDashboard" src="https://github.com/user-attachments/assets/00bab1e1-8275-42a8-8673-36759dfc2979" />

### Drowsiness Detection

<img width="1920" height="2391" alt="Drowsy" src="https://github.com/user-attachments/assets/e3009756-0a6d-4821-aeec-e69aaff0ae82" />

### Analytics Dashboard

![Analytics Dashboard](screenshots/analytics_dashboard.png)

## Installation

### Prerequisites

- Python 3.10 or a compatible Python version
- Webcam
- Git
- Trained CNN weights and matching model architecture for hybrid mode

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal.

**Note:** CNN mode requires the appropriate model files. Without them, the application can use rule-based EAR/MAR detection.

## Evaluation

The application supports offline rule-based evaluation under four simulated lighting conditions.

Available outputs include:

- Accuracy
- Average processing latency
- Skipped face detections
- Model-comparison charts, when comparison files are available
- Precision, recall, and F1-score tables, when detailed metrics are available

Use a separate labeled test dataset to measure generalization. Report actual performance metrics only after running and validating the evaluation.

## Future Enhancements

- Improve robustness to glasses, head movement, and occlusions
- Evaluate on diverse drivers and real-world lighting
- Optimize inference for edge devices
- Improve temporal smoothing and alert-state handling
- Add automated tests and more detailed performance evaluation

## Privacy & Safety

- Webcam processing depends on the environment where the application runs.
- Alert events are stored locally in `detection_log.csv` by default.
- Do not upload private recordings or sensitive data.
- This project is a prototype and must not be used as the sole safety mechanism in a vehicle.

## Author

**Shubham Modi**

*Data Science | Machine Learning | Computer Vision | Deep Learning*

- LinkedIn: www.linkedin.com/in/shubhammodi17
---

If you find this project interesting, consider giving the repository a star!
