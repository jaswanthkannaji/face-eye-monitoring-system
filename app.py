
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Face and Eye Monitoring System",
    page_icon="👁️",
    layout="wide"
)

# --------------------------------------------------
# TITLE AND DESCRIPTION
# --------------------------------------------------

st.title("👁️ Face and Eye Monitoring System")

st.write(
    "This application detects faces and eyes in an uploaded image "
    "or a camera snapshot using MediaPipe Face Mesh."
)

st.info(
    "Note: The camera option captures a snapshot. "
    "It does not provide continuous live video monitoring."
)

# --------------------------------------------------
# MEDIAPIPE FACE MESH INITIALIZATION
# --------------------------------------------------

mp_face_mesh = mp.solutions.face_mesh

# Landmark indexes used to identify the eye regions
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]


# --------------------------------------------------
# FACE AND EYE DETECTION FUNCTION
# --------------------------------------------------

def detect_faces_and_eyes(image):

    height, width, _ = image.shape

    # Convert BGR image to RGB for MediaPipe
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    face_count = 0
    eye_count = 0

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=10,
        refine_landmarks=True,
        min_detection_confidence=0.65
    ) as face_mesh:

        results = face_mesh.process(rgb_image)

        if results.multi_face_landmarks:

            face_count = len(results.multi_face_landmarks)

            for face_landmarks in results.multi_face_landmarks:

                # Get all face landmark coordinates
                points = []

                for landmark in face_landmarks.landmark:
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)

                    x = max(0, min(x, width - 1))
                    y = max(0, min(y, height - 1))

                    points.append((x, y))

                # Calculate face bounding box
                face_x = [point[0] for point in points]
                face_y = [point[1] for point in points]

                x_min = min(face_x)
                x_max = max(face_x)
                y_min = min(face_y)
                y_max = max(face_y)

                # Draw face rectangle
                cv2.rectangle(
                    image,
                    (x_min, y_min),
                    (x_max, y_max),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    image,
                    "Face",
                    (x_min, max(y_min - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                # Detect and draw both eye regions
                for eye_landmarks in [LEFT_EYE, RIGHT_EYE]:

                    eye_points = [
                        points[index] for index in eye_landmarks
                    ]

                    eye_x = [point[0] for point in eye_points]
                    eye_y = [point[1] for point in eye_points]

                    eye_x_min = min(eye_x)
                    eye_x_max = max(eye_x)
                    eye_y_min = min(eye_y)
                    eye_y_max = max(eye_y)

                    # Add small padding around eye region
                    padding = 8

                    eye_x_min = max(0, eye_x_min - padding)
                    eye_y_min = max(0, eye_y_min - padding)
                    eye_x_max = min(width - 1, eye_x_max + padding)
                    eye_y_max = min(height - 1, eye_y_max + padding)

                    # Draw eye rectangle
                    cv2.rectangle(
                        image,
                        (eye_x_min, eye_y_min),
                        (eye_x_max, eye_y_max),
                        (255, 0, 0),
                        2
                    )

                    cv2.putText(
                        image,
                        "Eye",
                        (eye_x_min, max(eye_y_min - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0),
                        1
                    )

                    eye_count += 1

    return image, face_count, eye_count


# --------------------------------------------------
# IMAGE PROCESSING FUNCTION
# --------------------------------------------------

def process_image(uploaded_file):

    pil_image = Image.open(uploaded_file).convert("RGB")

    image = np.array(pil_image)

    # Convert RGB to BGR for OpenCV
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    result_image, face_count, eye_count = detect_faces_and_eyes(image)

    # Convert BGR back to RGB for Streamlit
    result_image = cv2.cvtColor(
        result_image,
        cv2.COLOR_BGR2RGB
    )

    return result_image, face_count, eye_count


# --------------------------------------------------
# INPUT OPTIONS
# --------------------------------------------------

st.subheader("Choose an input method")

tab1, tab2 = st.tabs(["📁 Upload Image", "📷 Camera"])

# --------------------------------------------------
# OPTION 1: UPLOAD IMAGE
# --------------------------------------------------

with tab1:

    st.write("Upload an image containing one or more faces.")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        if st.button("Detect Faces and Eyes", key="upload_button"):

            try:
                with st.spinner("Detecting faces and eyes..."):

                    result_image, face_count, eye_count = process_image(
                        uploaded_file
                    )

                st.success("Detection completed!")

                st.image(
                    result_image,
                    caption="Face and Eye Detection Result",
                    use_container_width=True
                )

                col1, col2 = st.columns(2)

                col1.metric("Faces Detected", face_count)
                col2.metric("Eyes Detected", eye_count)

                if face_count == 0:
                    st.warning(
                        "No face detected. Try a clear, front-facing image."
                    )

            except Exception as error:
                st.error(f"An error occurred: {error}")


# --------------------------------------------------
# OPTION 2: CAMERA SNAPSHOT
# --------------------------------------------------

with tab2:

    st.write("Capture a photo using your device camera.")

    camera_image = st.camera_input("Take a picture")

    if camera_image is not None:

        if st.button("Detect Faces and Eyes", key="camera_button"):

            try:
                with st.spinner("Detecting faces and eyes..."):

                    result_image, face_count, eye_count = process_image(
                        camera_image
                    )

                st.success("Detection completed!")

                st.image(
                    result_image,
                    caption="Camera Detection Result",
                    use_container_width=True
                )

                col1, col2 = st.columns(2)

                col1.metric("Faces Detected", face_count)
                col2.metric("Eyes Detected", eye_count)

                if face_count == 0:
                    st.warning(
                        "No face detected. Please capture a clear image."
                    )

            except Exception as error:
                st.error(f"An error occurred: {error}")


# --------------------------------------------------
# COLOR LEGEND
# --------------------------------------------------

st.divider()

st.subheader("Detection Legend")

col1, col2 = st.columns(2)

with col1:
    st.markdown("🟩 **Green rectangle** – Detected face")

with col2:
    st.markdown("🟦 **Blue rectangle** – Detected eye region")


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Face and Eye Monitoring System | Built using Python, "
    "OpenCV, MediaPipe and Streamlit"
)
