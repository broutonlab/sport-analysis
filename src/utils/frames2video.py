import os

import cv2

image_folder = "/home/gleb/ssd/projects/sport-analysis/data/runs"
video_name = "play_sight_processed.avi"

images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
images = sorted(images)
print(images)
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape
video = cv2.VideoWriter(
    video_name, cv2.VideoWriter_fourcc(*"DIVX"), 25, (width, height)
)

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

# cv2.destroyAllWindows()
video.release()
