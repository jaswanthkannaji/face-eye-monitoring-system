"""
Face & Eye Monitoring System
Real-time face and eye detection using OpenCV Haar Cascade classifiers.

Run:
    python main.py                 # webcam mode
    python main.py --image photo.jpg  # test a still image
Press Q or ESC to exit webcam mode.
"""

import argparse
import sys
from pathlib import Path

import cv2


def load_cascade(path: str, description: str) -> cv2.CascadeClassifier:
    """Load a Haar Cascade XML file and fail clearly if it cannot be loaded."""
    cascade = cv2.CascadeClassifier(path)
    if cascade.empty():
        raise FileNotFoundError(
            f"Could not load {description} cascade from:\n{path}\n"
            "Check that OpenCV is installed and the XML file exists."
        )
    return cascade


def get_cascade_paths():
    """Use the Haar Cascade XML files bundled with the OpenCV Python package."""
    cascade_dir = Path(cv2.data.haarcascades)
    face_path = cascade_dir / "haarcascade_frontalface_default.xml"
    eye_path = cascade_dir / "haarcascade_eye.xml"
    return str(face_path), str(eye_path)


def detect_and_draw(frame, face_cascade, eye_cascade):
    """
    Detect faces on a grayscale frame, then detect eyes inside each face region.
    Returns the annotated frame, number of faces, and total number of detected eyes.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40),
    )

    total_eyes = 0

    for (x, y, width, height) in faces:
        # Green rectangle around each detected face.
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
        cv2.putText(
            frame, "Face", (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )

        # Search for eyes only inside the detected face region.
        face_gray = gray[y:y + height, x:x + width]
        eyes = eye_cascade.detectMultiScale(
            face_gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(15, 15),
        )

        for (eye_x, eye_y, eye_width, eye_height) in eyes:
            # Blue rectangle around each detected eye.
            cv2.rectangle(
                frame,
                (x + eye_x, y + eye_y),
                (x + eye_x + eye_width, y + eye_y + eye_height),
                (255, 0, 0),
                2,
            )
            total_eyes += 1

    cv2.putText(
        frame,
        f"Faces: {len(faces)} | Eyes: {total_eyes}",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2,
    )
    return frame, len(faces), total_eyes


def run_webcam(face_cascade, eye_cascade, camera_index=0):
    """Run live webcam detection until Q or ESC is pressed."""
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        print(
            "Error: webcam could not be opened. Check camera permissions or "
            "try another camera index, for example --camera 1."
        )
        return 1

    print("Webcam started. Press Q or ESC in the video window to quit.")
    while True:
        success, frame = camera.read()
        if not success:
            print("Warning: unable to read a frame from the webcam.")
            break

        # Mirror the preview to make it feel like a mirror.
        frame = cv2.flip(frame, 1)
        output, face_count, eye_count = detect_and_draw(
            frame, face_cascade, eye_cascade
        )
        cv2.imshow("Face & Eye Monitoring System", output)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    camera.release()
    cv2.destroyAllWindows()
    return 0


def run_image(image_path, face_cascade, eye_cascade):
    """Run the same detection pipeline on one image and display/save the result."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: could not read image: {image_path}")
        return 1

    output, face_count, eye_count = detect_and_draw(
        image, face_cascade, eye_cascade
    )
    output_path = str(Path(image_path).with_name(
        Path(image_path).stem + "_detected.jpg"
    ))
    cv2.imwrite(output_path, output)
    print(f"Detected faces: {face_count}")
    print(f"Detected eyes: {eye_count}")
    print(f"Annotated result saved to: {output_path}")
    cv2.imshow("Face & Eye Detection - Image Test", output)
    print("Press any key in the image window to close it.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Detect faces and eyes using pre-trained Haar Cascades."
    )
    parser.add_argument(
        "--image", type=str, help="Optional path to an image instead of webcam mode."
    )
    parser.add_argument(
        "--camera", type=int, default=0, help="Webcam index (default: 0)."
    )
    args = parser.parse_args()

    try:
        face_path, eye_path = get_cascade_paths()
        face_cascade = load_cascade(face_path, "face")
        eye_cascade = load_cascade(eye_path, "eye")
    except (FileNotFoundError, OSError) as error:
        print(f"Model loading error: {error}")
        return 1

    print("Face Cascade:", face_path)
    print("Eye Cascade:", eye_path)

    if args.image:
        return run_image(args.image, face_cascade, eye_cascade)
    return run_webcam(face_cascade, eye_cascade, args.camera)


if __name__ == "__main__":
    sys.exit(main())
