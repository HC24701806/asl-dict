import cv2
import mediapipe as mp
import time
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

features_face = [70, 105, 107, 33, 160, 158, 133, 153, 144,
                 336, 334, 300, 362, 385, 387, 263, 373, 380,
                 78, 73, 11, 303, 308, 320, 315, 85, 90]

video = cv2.VideoCapture('./ASL_Citizen/videos/9031375678366-KILL.mp4')
fps = video.get(cv2.CAP_PROP_FPS)
frame_num = 0
begin_frame_num = 0
in_clip = False
strikes = 0
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5, refine_face_landmarks=True) as holistic:
    while video.isOpened():
        if frame_num - begin_frame_num > 3 * fps:
            break

        ret, frame = video.read()
        if not ret:
            break
        
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        frame.flags.writeable = False
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)

        if not results.left_hand_landmarks and not results.right_hand_landmarks and in_clip:
            strikes += 1
            if strikes == 2:
                break
        if (results.left_hand_landmarks or results.right_hand_landmarks) and not in_clip:
            begin_frame_num = frame_num
            in_clip = True

        if in_clip:
            # Draw landmark annotation on the image.
            frame.flags.writeable = True
            image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if results.face_landmarks:
                for i in features_face:
                    face_pt = results.face_landmarks.landmark[i]
                    if face_pt:
                        image = cv2.circle(image, (int(640 * face_pt.x), int(480 * face_pt.y)), 2, (0, 0, 255), -1)

            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    results.left_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles
                    .get_default_hand_landmarks_style())
             
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    results.right_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles
                    .get_default_hand_landmarks_style())
        
            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))
        
        time.sleep(0.5)
        frame_num += int(fps * 0.25)
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        if cv2.waitKey(1) == ord('q'):
            break
video.release()