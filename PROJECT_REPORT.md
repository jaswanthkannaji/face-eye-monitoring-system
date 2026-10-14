# Face & Eye Monitoring System — Project Report

## Student details
- Student name: [Enter your name]
- Student ID: [Enter your ID]
- Course/subject: Pre-Trained Deep Learning Models — Individual Mini Project
- Project title: Face & Eye Monitoring System
- Implementation language: Python

## 1. Problem statement
The project detects visible frontal faces and eyes in webcam frames and draws bounding boxes around them. It demonstrates basic visual monitoring using a pre-trained detector.

## 2. Objective
To implement a real-time face and eye detection pipeline using OpenCV Haar Cascade classifiers, without training a model from scratch.

## 3. Introduction to the pre-trained model
Haar Cascade is a classical object-detection approach based on Haar-like features and a cascade of classifiers. OpenCV provides XML classifier files for frontal-face and eye detection. `CascadeClassifier.detectMultiScale()` scans an image at different scales and returns candidate bounding boxes.

## 4. Why this model was selected
- It is easy to load using OpenCV.
- It runs on a CPU and is suitable for a small webcam demonstration.
- It avoids training from scratch.
- It is straightforward to explain in a project review.

## 5. Model architecture / working principle
Webcam frame → grayscale conversion → histogram equalization → face cascade inference → eye cascade inference inside each face → bounding boxes and counts → display.

## 6. Libraries and installation
- Python
- OpenCV Python package (`opencv-python`)

Install using `pip install -r requirements.txt`.

## 7. Input, preprocessing, inference and output
- **Input:** Webcam frame or image file.
- **Preprocessing:** BGR-to-grayscale conversion and histogram equalization.
- **Inference:** `detectMultiScale()` with the face cascade, followed by the eye cascade for each face crop.
- **Post-processing:** Draw face and eye rectangles and count detections.
- **Output:** Live annotated video or a saved annotated image.

## 8. Test cases
Perform and document at least three actual tests:
1. One person facing the camera in good lighting.
2. Multiple people facing the camera.
3. Face turned partly away or a dimly lit scene.

## 9. Results
Complete after testing. Do not fill in guessed results.

| Test | Faces detected | Eyes detected | Notes |
|---|---:|---:|---|
| 1 | [Actual result] | [Actual result] | [Observation] |
| 2 | [Actual result] | [Actual result] | [Observation] |
| 3 | [Actual result] | [Actual result] | [Observation] |

## 10. Limitations
Detection can be unreliable for non-frontal faces, poor lighting, occlusion, and some eyewear. False positives and missed detections are possible. The system does not recognize identity or infer attention, emotion, or health.

## 11. Future scope
Possible improvements include video-file input, FPS display, configurable thresholds, a user interface, and comparison with a modern pre-trained detector.

## 12. Conclusion
The implementation demonstrates the main stages of a pre-trained model workflow: input, preprocessing, inference, post-processing, and output visualization. The final report should include screenshots and actual observations collected while running the program.

## 13. Model source and license
The cascade files are provided by OpenCV: https://github.com/opencv/opencv/tree/4.x/data/haarcascades. The Python package is at https://github.com/opencv/opencv-python. OpenCV 4.x is distributed under Apache-2.0; verify the exact release and any individual cascade-file notices before redistributing model files. This project loads the cascades from the installed package and does not separately redistribute XML files.
