import cv2
import multiprocessing
import streamlit as st
import time


def capture_frames(queue):
    capture = cv2.VideoCapture(
        r"Source/ML/Crimes/appleTheft.mp4"
    )

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        queue.put(frame)

    capture.release()


if __name__ == "__main__":
    frame_queue = multiprocessing.Queue()

    capture_process = multiprocessing.Process(
        target=capture_frames, args=(frame_queue,)
    )
    capture_process.start()

    st.title("Camera Stream")
    st.set_option("deprecation.showPyplotGlobalUse", False)
    videa_placeholder = st.empty()

    try:
        while True:
            frame = frame_queue.get()
            if frame is None:
                break

            videa_placeholder.image(frame, channels="BGR", use_column_width=True)
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass

    capture_process.terminate()  # Terminate the capture process
    capture_process.join()  # Wait for the capture process to finish