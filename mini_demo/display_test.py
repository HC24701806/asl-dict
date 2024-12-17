import cv2

video = cv2.VideoCapture('test.mp4')
while video.isOpened():
    ret, frame = video.read()
    print(ret, frame)
    if not ret:
        break
    cv2.imshow('video', frame)
    if cv2.waitKey(1) == ord('q'):
        break
video.release()
cv2.destroyAllWindows()