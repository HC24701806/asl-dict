import cv2
import mediapipe as mp
import matplotlib as plt
import pandas as pd

mp_holistic = mp.solutions.holistic

video = cv2.VideoCapture('./ASL_Citizen/videos/66221823911-CARRY.mp4')
fps = video.get(cv2.CAP_PROP_FPS)
frame_num = 0
data = pd.DataFrame()
row = 0
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while video.isOpened():
        if frame_num >= 3 * fps:
            break

        ret, frame = video.read()
        if not ret:
            break

        frame.flags.writeable = False
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(img)
        
        if results.face_landmarks:
            for i, point in enumerate(results.face_landmarks.landmark):
                data.loc[row, 'x'] = point.x
                data.loc[row, 'y'] = point.y
                data.loc[row, 'type'] = 'face'
                data.loc[row, 'frame'] = frame_num
                row += 1

        if results.left_hand_landmarks:
            for i, point in enumerate(results.left_hand_landmarks.landmark):
                data.loc[row, 'x'] = point.x
                data.loc[row, 'y'] = point.y
                data.loc[row, 'type'] = 'left hand'
                data.loc[row, 'frame'] = frame_num
                row += 1
        
        if results.right_hand_landmarks:
            for i, point in enumerate(results.right_hand_landmarks.landmark):
                data.loc[row, 'x'] = point.x
                data.loc[row, 'y'] = point.y
                data.loc[row, 'type'] = 'right hand'
                data.loc[row, 'frame'] = frame_num
                row += 1
        
        frame_num += int(fps * 0.25)
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        if cv2.waitKey(1) == ord('q'):
            break
video.release()
print(data)
data.to_csv('./dataset/66221823911-CARRY.csv')