
import streamlit as st
import cv2
import numpy as np

# -----------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------

st.set_page_config(
    page_title="Face and Eye Monitoring System",
    page_icon="👁️",
    layout="centered"
)

st.title("Face and Eye Monitoring System")
st.write("Detect faces and eyes using OpenCV Haar Cascade classifiers.")

st.info(
    "Upload an image or capture a camera snapshot "
    "to detect faces and eyes."
)


# -----------------------------------------
# LOAD HAAR CASCADE CLASSIFIERS
# -----------------------------------------

@st.cache_resource
def load_cascades():

    face_path = (
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )

    eye_path = (
        cv2.data.haarcascades
        + "haarcascade_eye.xml"
    )

    face_cascade = cv2.CascadeClassifier(face_path)
    eye_cascade = cv2.CascadeClassifier(eye_path)

    if face_cascade.empty() or eye_cascade.empty():
        raise RuntimeError("Unable to load Haar Cascade models.")

    return face_cascade, eye_cascade


# -----------------------------------------
# FACE AND EYE DETECTION
# -----------------------------------------

def detect_faces_and_eyes(image, face_cascade, eye_cascade):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.2,
    minNeighbors=12,
    minSize=(120, 120),
    flags=cv2.CASCADE_SCALE_IMAGE
)

# Filter out unlikely face-shaped regions
valid_faces = []

for (x, y, w, h) in faces:

    aspect_ratio = w / float(h)

    if 0.75 <= aspect_ratio <= 1.30:
        valid_faces.append((x, y, w, h))

faces = valid_faces

    eye_count = 0

    for (x, y, w, h) in faces:

        # Draw face rectangle
        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            image,
            "Face",
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # Extract only the upper 60% of the face
        # This prevents the detector from scanning the mouth.

        upper_face_height = int(h * 0.60)

        face_gray = gray[
            y:y + upper_face_height,
            x:x + w
        ]

        # Improve contrast in the eye region
        face_gray = cv2.equalizeHist(face_gray)

        # Detect eyes only in the upper face region
        eyes = eye_cascade.detectMultiScale(
            face_gray,
            scaleFactor=1.1,
            minNeighbors=8,
            minSize=(20, 20),
            maxSize=(int(w * 0.45), int(h * 0.30))
        )

        for (ex, ey, ew, eh) in eyes:

            # Draw eye rectangle
            cv2.rectangle(
                image,
                (x + ex, y + ey),
                (x + ex + ew, y + ey + eh),
                (255, 0, 0),
                2
            )

            cv2.putText(
                image,
                "Eye",
                (x + ex, max(y + ey - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1
            )

            eye_count += 1

    return image, len(faces), eye_count


# -----------------------------------------
# PROCESS IMAGE
# -----------------------------------------

def process_image(uploaded_file):

    file_bytes = np.frombuffer(
        uploaded_file.getvalue(),
        np.uint8
    )

    image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise ValueError("Unable to read the uploaded image.")

    face_cascade, eye_cascade = load_cascades()

    result_image, face_count, eye_count = detect_faces_and_eyes(
        image,
        face_cascade,
        eye_cascade
    )

    result_image = cv2.cvtColor(
        result_image,
        cv2.COLOR_BGR2RGB
    )

    return result_image, face_count, eye_count


# -----------------------------------------
# INPUT OPTIONS
# -----------------------------------------

tab1, tab2 = st.tabs(["Upload Image", "Camera"])


# -----------------------------------------
# UPLOAD IMAGE
# -----------------------------------------

with tab1:

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        st.image(
            uploaded_file,
            caption="Original Image",
            use_container_width=True
        )

        if st.button("Detect Faces and Eyes"):

            try:

                with st.spinner("Detecting..."):

                    result, faces, eyes = process_image(
                        uploaded_file
                    )

                st.success("Detection completed!")

                st.image(
                    result,
                    caption="Detection Result",
                    use_container_width=True
                )

                col1, col2 = st.columns(2)

                col1.metric("Faces Detected", faces)
                col2.metric("Eye Regions Detected", eyes)

                if faces == 0:
                    st.warning("No faces detected.")

                elif eyes == 0:
                    st.warning(
                        "Face detected, but eyes were not detected. "
                        "Try a clear, front-facing image."
                    )

            except Exception as error:
                st.error(f"Error: {error}")


# -----------------------------------------
# CAMERA SNAPSHOT
# -----------------------------------------

with tab2:

    camera_image = st.camera_input("Take a picture")

    if camera_image is not None:

        if st.button("Detect Faces and Eyes", key="camera_detect"):

            try:

                with st.spinner("Detecting..."):

                    result, faces, eyes = process_image(
                        camera_image
                    )

                st.success("Detection completed!")

                st.image(
                    result,
                    caption="Camera Detection Result",
                    use_container_width=True
                )

                col1, col2 = st.columns(2)

                col1.metric("Faces Detected", faces)
                col2.metric("Eye Regions Detected", eyes)

                if faces == 0:
                    st.warning("No faces detected.")

            except Exception as error:
                st.error(f"Error: {error}")


# -----------------------------------------
# LEGEND
# -----------------------------------------

st.divider()

st.write("**Detection Legend**")
st.write("🟩 Green rectangle – Face")
st.write("🟦 Blue rectangle – Eye")
