# Viva Preparation: Face & Eye Monitoring System

## 1. What is a pre-trained model?
A model whose parameters or classifier structure were learned from training performed earlier. We use it on new input rather than training it from scratch.

## 2. Why use a pre-trained model?
It saves training time and computing resources and lets us build an application from an existing model.

## 3. What is inference?
Applying a trained model to new input to obtain a prediction or detection.

## 4. Training vs inference
Training learns model parameters from training data. Inference uses the already-trained model to process new images.

## 5. What is Haar Cascade?
A classical object-detection method that uses Haar-like features and a cascade of classifiers to find candidate objects efficiently.

## 6. What does `detectMultiScale()` do?
It searches for objects at different image scales and returns bounding boxes represented by `(x, y, width, height)`.

## 7. Why convert to grayscale?
The supplied Haar cascade detectors commonly operate on grayscale images; grayscale also reduces the amount of image data to process.

## 8. Why histogram equalization?
It can improve image contrast, which may help the detector under some lighting conditions. It does not guarantee better detection.

## 9. Why detect eyes inside each face?
It restricts eye detection to a face region, reducing irrelevant detections in the background.

## 10. What do the rectangles mean?
The green rectangle indicates a detected face region; blue rectangles indicate regions classified as eyes.

## 11. What are the model inputs and outputs?
Input is a grayscale image (or face crop). Output is a list of candidate bounding boxes.

## 12. What are limitations?
Poor lighting, side views, occlusion, unusual poses, glasses, and similar visual patterns may cause missed or false detections.

## 13. Is it a deep-learning model?
Haar Cascade is a classical machine-learning detector, not a modern deep neural network. It is still a pre-trained model approach suitable for the assignment's suggested model.

## 14. What license does it use?
OpenCV 4.x is distributed under Apache-2.0, but verify the exact release and any individual cascade XML notices before redistribution.

## 15. What improvements can be added?
Video file support, FPS display, a user interface, adjustable parameters, and evaluation against a modern detector.
