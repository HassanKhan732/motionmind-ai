"""
MotionMind AI - Step 1: Live Webcam Feed
Opens the default webcam, displays the live feed, and exits on 'q'.
"""

import cv2


def main():
    # 0 represents the default built-in or connected webcam
    camera_index = 0
    cap = cv2.VideoCapture(camera_index)

    # Check if the webcam opened successfully
    if not cap.isOpened():
        print(f"Error: Could not access the webcam at index {camera_index}.")
        print("Please verify that your camera is connected and not in use by another application.")
        return

    print("Webcam successfully opened.")
    print("Press 'q' in the video window to quit.")

    while True:
        # Capture frame-by-frame from webcam
        ret, frame = cap.read()

        # If frame was not retrieved properly, exit loop
        if not ret:
            print("Error: Failed to grab frame from webcam. Exiting...")
            break

        # Display the live frame in an OpenCV window
        cv2.imshow("MotionMind AI - Webcam Feed", frame)

        # Wait 1 ms for a key event; exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Exiting webcam feed...")
            break

    # Release resources cleanly
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
