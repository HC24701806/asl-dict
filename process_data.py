import cv2
import mediapipe as mp
mp_holistic = mp.solutions.holistic

video = cv2.VideoCapture('./ASL_Citizen/videos/66221823911-CARRY.mp4')
fps = video.get(cv2.CAP_PROP_FPS)
i = 0
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        frame.flags.writeable = False
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(img)
        i += int(fps * 0.25)
        video.set(cv2.CAP_PROP_POS_FRAMES, i)
        if cv2.waitKey(1) == ord('q'):
            break
video.release()