import cv2

def generate_frames():
    # Create a VideoCapture object for your camera (0 for the default camera).
    # If you want to stream a video file, you can specify the file path instead of 0.
    cap = cv2.VideoCapture(r"Source/Frontend/accident_footage.mp4")
    # cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # Check if the camera/video file is opened successfully
    if not cap.isOpened():
        print("Error: Unable to access the camera or video file.")
        return

    # Loop to continuously read frames and yield each frame
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Warning: Unable to read frame from the camera or video file.")
            break

        # Convert the frame from BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Yield the RGB frame
        yield rgb_frame

    # Release the VideoCapture when the loop ends
    cap.release()


# Example usage: Generate and display frames one by one
if __name__ == "__main__":
    for frame in generate_frames():
        cv2.imshow("OpenCV Video Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()