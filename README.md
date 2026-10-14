# Face & Eye Monitoring System

A beginner-friendly computer vision mini project that detects faces and eyes simultaneously from a live webcam stream using pre-trained Haar Cascade classifiers in OpenCV.

## Features
- Real-time webcam face detection
- Eye detection inside each detected face
- Green face bounding boxes and blue eye bounding boxes
- Live face/eye counts
- Still-image testing mode that saves an annotated image
- Basic error handling for missing model files, unreadable images, and unavailable cameras

## 1. Problem statement
Detect visible human faces and eyes from webcam frames and mark their locations. This demonstrates how a pre-trained computer-vision model can be used for inference in a practical application without training a model from scratch.

## 2. Objective
Use OpenCV's pre-trained Haar Cascade XML classifiers to detect faces and eyes in a webcam stream, process the detections, and display annotated frames.

## 3. Model information
- **Model:** Haar Cascade Classifier (frontal face and eye cascades)
- **Type:** Cascade of boosted classifiers using Haar-like image features
- **Input:** Image/video frame; this project converts frames to grayscale and equalizes the histogram
- **Output:** Bounding-box coordinates for candidate face and eye regions
- **Inference API:** `CascadeClassifier.detectMultiScale()`
- **Why selected:** Lightweight, included with OpenCV's Python package, works on CPU, and is suitable for a small real-time demonstration
- **Model source:** OpenCV repository, `data/haarcascades`: https://github.com/opencv/opencv/tree/4.x/data/haarcascades
- **Library source:** https://github.com/opencv/opencv-python
- **License note:** OpenCV 4.x is distributed under Apache-2.0. Check the license and notice files in the exact OpenCV release and any individual cascade XML file before redistribution; keep upstream notices with redistributed model files. This project uses the XML files supplied by the installed `opencv-python` package and does not redistribute them separately.

Haar Cascades are classical machine-learning detectors, not modern deep neural networks. The classifiers were trained before use; this project only performs inference.

## 4. Workflow
1. Capture a webcam frame or read a test image.
2. Convert the BGR frame to grayscale.
3. Apply histogram equalization to improve local contrast.
4. Run the frontal-face cascade using `detectMultiScale()`.
5. For each detected face, run the eye cascade within that face region.
6. Draw bounding boxes, display detection counts, and show/save the output.

## 5. Requirements
- Python 3.9 or newer recommended
- A webcam for live mode
- Windows, macOS, or Linux desktop session for OpenCV display windows

## 6. Installation

Open a terminal in this project folder.

### Windows
```bash
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 7. Run the project

### Live webcam
```bash
python main.py
```
Press **Q** or **Esc** while the video window is active to quit.

### Select a different webcam
```bash
python main.py --camera 1
```

### Test an image
```bash
python main.py --image path/to/your_photo.jpg
```
The annotated image is saved next to the input image with `_detected.jpg` appended to its filename. Use images you have permission to use, preferably with the subjects' knowledge and consent.

## 8. Important parameters
- `scaleFactor=1.1`: how much the image size is reduced at each scale step. Smaller steps can improve detection but take longer.
- `minNeighbors=5` for faces: higher values generally reduce false positives but may miss faces.
- `minSize=(40, 40)` for faces: ignores very small candidate regions.
- Eye detection runs inside each face crop, reducing irrelevant detections elsewhere in the frame.

These are starting values, not guarantees of accuracy. Adjust them based on your environment and test results.

## 9. Test cases
Run at least three cases and record what actually happens on your machine. Do not claim results that you have not tested.

| Test case | Input | Expected observation |
|---|---|---|
| 1 | One person facing the camera in good lighting | Usually one face box; eye boxes may appear if visible |
| 2 | Two or more people in the frame | Attempt to mark each clearly visible frontal face and eyes |
| 3 | Face partly turned, partly covered, or in dim light | Detection may be missed or unstable; record the limitation |
| 4 (optional) | A still image passed with `--image` | Annotated result saved as a new JPG |

Record your actual face/eye counts and any false detections in a separate results table for submission.

## 10. Results / observations
Complete this table after running the program.

| Test case | Actual faces detected | Actual eyes detected | Observation |
|---|---:|---:|---|
| Test 1 | Fill after test | Fill after test | Fill after test |
| Test 2 | Fill after test | Fill after test | Fill after test |
| Test 3 | Fill after test | Fill after test | Fill after test |

## 11. Limitations
- Haar Cascades can miss faces under poor lighting, unusual angles, occlusion, or large variations in scale.
- Eye detection may identify glasses, eyebrows, or other regions incorrectly.
- It does not identify a person's identity, emotion, attention, or health.
- Detection counts are estimates, not a reliable security or attendance record.
- Performance depends on webcam quality, lighting, face orientation, and parameter settings.

## 12. Future scope
- Add a simple graphical interface and a start/stop control.
- Add optional video-file input.
- Compare Haar Cascade performance with a modern pre-trained face detector.
- Add FPS display and configurable detection parameters.
- Improve testing across lighting conditions and camera angles.

## 13. Conclusion
This project demonstrates a complete input-to-output inference pipeline using pre-trained Haar Cascade classifiers. It captures video frames, preprocesses them, detects face and eye regions, and visualizes the results. It is lightweight and easy to understand, but its detections can be less reliable than modern deep-learning detectors in challenging conditions.

## 14. GitHub upload
1. Create a new empty GitHub repository named `face-eye-monitoring-system`.
2. Open a terminal in this folder.
3. Run:

```bash
git init
git add .
git commit -m "Add face and eye monitoring system"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/face-eye-monitoring-system.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your GitHub username. Do not commit private webcam recordings or photos of people without permission.
