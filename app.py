import streamlit as st
import cv2
import numpy as np

st.set_page_config(
    page_title="Face and Eye Monitoring System",
    page_icon="👁️",
    layout="centered"
)

st.title("Face and Eye Monitoring System")
st.write("Detect faces and eyes using OpenCV Haar Cascade classifiers.")

@st.cache_resource
def load_cascades():
    face_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    eye_path = cv2.data.haarcascades + "haarcascade_eye.xml"

    face_cascade = cv2.CascadeClassifier(face_path)
    eye_cascade = cv2.CascadeClassifier(eye_path)

    if face_cascade.empty() or eye_cascade.empty():
        raise RuntimeError("Unable to load Haar Cascade models.")

    return face_cascade, eye_cascade


def detect_faces_and_eyes(image, face_cascade, eye_cascade):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=8,
    minSize=(60, 60)
    )

    eye_count = 0

    for (x, y, w, h) in faces:
        cv2.rectangle(
            image, (x, y), (x + w, y + h),
            (0, 255, 0), 2
        )

        face_gray = gray[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(
            face_gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(15, 15)
        )

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(
                image,
                (x + ex, y + ey),
                (x + ex + ew, y + ey + eh),
                (255, 0, 0), 2
            )
            eye_count += 1

    return image, len(faces), eye_count


try:
    face_cascade, eye_cascade = load_cascades()
except Exception as error:
    st.error(f"Model loading failed: {error}")
    st.stop()

st.info("Upload an image or take a photo using your camera.")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

camera_photo = st.camera_input("Or take a photo")

selected_image = uploaded_file if uploaded_file is not None else camera_photo

if selected_image is not None:
    file_bytes = np.frombuffer(
        selected_image.getvalue(),
        np.uint8
    )

    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error("Unable to read the selected image.")
    else:
        result, face_count, eye_count = detect_faces_and_eyes(
            image.copy(),
            face_cascade,
            eye_cascade
        )

        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

        st.subheader("Detection Results")

        col1, col2 = st.columns(2)

        col1.metric("Faces Detected", face_count)
        col2.metric("Eyes Detected", eye_count)

        st.image(
            result_rgb,
            caption="Face and Eye Detection",
            use_container_width=True
        )

        success, encoded_image = cv2.imencode(".jpg", result)

        if success:
            st.download_button(
                "Download Result",
                data=encoded_image.tobytes(),
                file_name="detection_result.jpg",
                mime="image/jpeg"
            )

st.caption("Detection results may vary depending on lighting and image quality.")
