
# -----------------------------------------
# CREATE TABS
# -----------------------------------------

tab1, tab2 = st.tabs(["Upload Image", "Camera"])


# -----------------------------------------
# TAB 1: UPLOAD IMAGE
# -----------------------------------------

with tab1:

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        if st.button("Detect Faces and Eyes", key="upload_detect"):

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
# TAB 2: CAMERA SNAPSHOT
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

                elif eyes == 0:
                    st.warning(
                        "Face detected, but eyes were not detected. "
                        "Try a clear, front-facing image."
                    )

            except Exception as error:
                st.error(f"Error: {error}")


# -----------------------------------------
# LEGEND
# -----------------------------------------

st.divider()

st.write("**Detection Legend**")
st.write("🟩 Green rectangle – Face")
st.write("🟦 Blue rectangle – Eye")
