import cv2

video = cv2.VideoCapture(0)
writer = cv2.VideoWriter('test.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 24, (640, 480))
counter = 0
while video.isOpened():
    if counter == 24:
        break
    ret, frame = video.read()
    if not ret:
        break
    cv2.imshow('vid', frame)
    writer.write(frame)
    counter += 1
    if cv2.waitKey(1) == ord('q'):
        break
video.release()
writer.release()