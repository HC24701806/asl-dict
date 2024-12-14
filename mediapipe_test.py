import cv2
import mediapipe as mp
from PIL import Image
import imagehash
import time
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

video = cv2.VideoCapture('./ASL_Citizen/videos/6464386294302-DISAPPOINT.mp4')
fps = video.get(cv2.CAP_PROP_FPS)
frame_num = 0
in_clip = False
hash1 = None
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while video.isOpened():
        if frame_num > 0.5 * fps:
            if frame_num > 2.5 * fps:
                break

            ret, frame = video.read()
            if not ret:
                break
            
            if hash1:
                hash2 = imagehash.phash(Image.fromarray(frame))
                if hash2 - hash1 == 0 and in_clip:
                    break
                if hash2 - hash1 > 0 and not in_clip:
                    in_clip = True
                hash1 = hash2
            else:
                hash1 = imagehash.phash(Image.fromarray(frame)) 
            
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            if in_clip:
                frame.flags.writeable = False
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)

                # Draw landmark annotation on the image.
                frame.flags.writeable = True
                image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                if results.face_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        results.face_landmarks,
                        mp_holistic.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_contours_style())
                
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        results.pose_landmarks,
                        mp_holistic.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles
                        .get_default_pose_landmarks_style())
                
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