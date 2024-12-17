import cv2
import csv
import ffmpeg
import mediapipe as mp

mp_holistic = mp.solutions.holistic

with open('random_sample.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)

    for line in reader:
        filename = line[1][:-4]
        video = cv2.VideoCapture(f'../ASL_Citizen/videos/{filename}.mp4')
        fps = video.get(cv2.CAP_PROP_FPS)
        size = (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT)
        frame_num = 0
        begin_frame_num = 0
        end_frame_num = 0
        in_clip = False
        strikes = 0
        with mp_holistic.Holistic(min_detection_confidence=0.25, min_tracking_confidence=0.25, refine_face_landmarks=True) as holistic:
            while video.isOpened():
                if frame_num - begin_frame_num >= 3 * fps:
                    end_frame_num = frame_num
                    break

                ret, frame = video.read()
                if not ret:
                    break
                
                frame.flags.writeable = False
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)

                num_landmarks = [0, 0]
                if results.left_hand_landmarks:
                    for i in range(21):
                        lhand_pt = results.left_hand_landmarks.landmark[i]
                        if lhand_pt:
                            x = lhand_pt.x
                            y = lhand_pt.y
                            if 0.1 <= x and x <= 0.9 and 0.1 <= y and y <= 0.9:
                                num_landmarks[0] += 1

                if results.right_hand_landmarks:
                    for i in range(21):
                        rhand_pt = results.right_hand_landmarks.landmark[i]
                        if rhand_pt:
                            x = rhand_pt.x
                            y = rhand_pt.y
                            if 0.1 <= x and x <= 0.9 and 0.1 <= y and y <= 0.9:
                                num_landmarks[1] += 1
                
                if num_landmarks[0] > 10 or num_landmarks[1] > 10:
                    if not in_clip:
                        begin_frame_num = frame_num
                        in_clip = True
                elif in_clip:
                    strikes += 1
                    if strikes >= 12:
                        end_frame_num = frame_num
                        break

                frame_num += 1
        video.release()
        
        start = begin_frame_num/fps
        end = end_frame_num/fps
        {
            ffmpeg.input(f'../ASL_Citizen/videos/{filename}.mp4', ss=start, to=end)
            .output(f'./mini_dataset/{filename}_1.mp4', vcodec='copy', acodec='copy')
            .run()
        }