import numpy as np
import cv2
import csv
import mediapipe as mp
import pandas as pd

mp_holistic = mp.solutions.holistic

features_face = [70, 105, 107, 33, 160, 158, 133, 153, 144,
                 336, 334, 300, 362, 385, 387, 263, 373, 380,
                 78, 73, 11, 303, 308, 320, 315, 85, 90]

with open(f'random_sample.csv') as csv_file:
    reader = csv.reader(csv_file)
    next(reader)

    for row in reader:
        print(row)
        video_id = row[1].split('.')[0]
        word = row[2]

        video = cv2.VideoCapture(f'../ASL_Citizen/videos/{video_id}.mp4')
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_num = 0
        begin_frame_num = 0
        in_clip = False
        strikes = 0

        data = pd.DataFrame()
        row = 0
        with mp_holistic.Holistic(min_detection_confidence=0.25, min_tracking_confidence=0.25, refine_face_landmarks=True) as holistic:
            while video.isOpened():
                if frame_num - begin_frame_num >= 3 * fps:
                    break

                ret, frame = video.read()
                if not ret:
                    break

                frame.flags.writeable = False
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(img)
                num_landmarks = [0, 0]

                index = 0
                data_arr = np.zeros(138, dtype=float)
                if results.face_landmarks:
                    for i in features_face:
                        face_pt = results.face_landmarks.landmark[i]
                        if face_pt:
                            data_arr[index] = face_pt.x
                            data_arr[index + 1] = face_pt.y
                        index += 2

                index = 54
                if results.left_hand_landmarks:
                    for i in range(21):
                        lhand_pt = results.left_hand_landmarks.landmark[i]
                        if lhand_pt:
                            x = lhand_pt.x
                            y = lhand_pt.y
                            data_arr[index] = lhand_pt.x
                            data_arr[index + 1] = lhand_pt.y
                            if 0.1 <= x and x <= 0.9 and 0.1 <= y and y <= 0.9:
                                num_landmarks[0] += 1
                        index += 2

                index = 96
                if results.right_hand_landmarks:
                    for i in range(21):
                        rhand_pt = results.right_hand_landmarks.landmark[i]
                        if rhand_pt:
                            x = rhand_pt.x
                            y = rhand_pt.y
                            data_arr[index] = rhand_pt.x
                            data_arr[index + 1] = rhand_pt.y
                            if 0.1 <= x and x <= 0.9 and 0.1 <= y and y <= 0.9:
                                num_landmarks[1] += 1
                        index += 2
                
                if num_landmarks[0] > 10 or num_landmarks[1] > 10:
                    if not in_clip:
                        begin_frame_num = frame_num
                        in_clip = True
                elif in_clip:
                    strikes += 1
                    if strikes == 2:
                        break  

                if in_clip:
                    for i in range(69):
                        data.loc[row, f'x{i}'] = data_arr[2 * i]
                        data.loc[row, f'y{i}'] = data_arr[2 * i + 1]
                row += 1
                
                frame_num += int(fps * 0.25)
                video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                if cv2.waitKey(1) == ord('q'):
                    break
        video.release()
        data.to_csv(f'./mini_dataset/{video_id}.csv')