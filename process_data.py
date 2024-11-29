import cv2
import numpy as np
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='face_landmark_model.task'),
    running_mode=VisionRunningMode.VIDEO)

with FaceLandmarker.create_from_options(options) as model:
    cam = cv2.VideoCapture(0)
    fps = cam.get(cv2.CAP_PROP_FPS)
    while True:
        ret, frame = cam.read()
        if ret:
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            res = model.detect_for_video(img, fps)

        if cv2.waitKey(1) == ord('q'):
            break
    cam.release()