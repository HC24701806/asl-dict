import cv2
import mediapipe as mp
mp_holistic = mp.solutions.holistic

cam = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cam.isOpened():
        ret, frame = cam.read()
        if ret:
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            frame.flags.writeable = False
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)

        if cv2.waitKey(1) & 0xFF == 27:
            break
cam.release()