import cv2
from glob import glob
import os

path_to_video = "/home/gleb/ssd/projects/sport-analysis/data/runs/play_sight_raw_out.mp4"

cap = cv2.VideoCapture(path_to_video)
path_to_save = '/home/gleb/ssd/projects/sport-analysis/data/runs/frames/'

f = -1
while cap.isOpened():
    ret, frame = cap.read()
    f += 1
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    height, width, _ = frame.shape
    frame = frame[:, :]
    # if f % 10 == 0:
    #     cv2.imshow("frame", frame)
    #     cv2.waitKey(0)
    cv2.imwrite(path_to_save + f'{f}.jpg', frame)

cap.release()
# cv2.destroyAllWinndows()
print("FINISH")
# send a message




