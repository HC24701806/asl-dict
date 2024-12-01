import cv2
import os
import csv
import mediapipe as mp
import pandas as pd

mp_holistic = mp.solutions.holistic

for filename in os.listdir('./ASL_Citizen/splits'):
    split = filename.split('.')[0]
    with open(f'./ASL_Citizen/splits/{filename}') as csv_file:
        reader = csv.reader(csv_file)
        next(reader)

        for row in reader:
            print(row)
            video_id = row[1].split('.')[0]
            word = row[2]

            video = cv2.VideoCapture(f'./ASL_Citizen/videos/{video_id}.mp4')
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
                            data.loc[row, 'frame'] = frame_num
                            data.loc[row, 'x'] = point.x
                            data.loc[row, 'y'] = point.y
                            data.loc[row, 'type'] = 'face'
                            row += 1

                    if results.pose_landmarks:
                        for i, point in enumerate(results.pose_landmarks.landmark):
                            data.loc[row, 'frame'] = frame_num
                            data.loc[row, 'x'] = point.x
                            data.loc[row, 'y'] = point.y
                            data.loc[row, 'type'] = 'pose'
                            row += 1

                    if results.left_hand_landmarks:
                        for i, point in enumerate(results.left_hand_landmarks.landmark):
                            data.loc[row, 'frame'] = frame_num
                            data.loc[row, 'x'] = point.x
                            data.loc[row, 'y'] = point.y
                            data.loc[row, 'type'] = 'left hand'
                            row += 1
                    
                    if results.right_hand_landmarks:
                        for i, point in enumerate(results.right_hand_landmarks.landmark):
                            data.loc[row, 'frame'] = frame_num
                            data.loc[row, 'x'] = point.x
                            data.loc[row, 'y'] = point.y
                            data.loc[row, 'type'] = 'right hand'
                            row += 1
                    
                    frame_num += int(fps * 0.25)
                    video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                    if cv2.waitKey(1) == ord('q'):
                        break
            video.release()
            data.to_csv(f'./dataset/{video_id}.csv')