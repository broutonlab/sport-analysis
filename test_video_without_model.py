import argparse
import os

import cv2
import matplotlib as mpl
import matplotlib.cm as mtpltcm
import numpy as np

from src.data.load_data import get_image_and_keypoints

"""This file can be used for creating a video
if you have the same structure of data as in the
sample dataset """

parser = argparse.ArgumentParser(description=" ")
parser.add_argument(
    "--path_to_images",
    type=str,
    default="./data/raw/1_1m",
    help="path to video. (default:./data/raw/1_1m)",
)
parser.add_argument(
    "--path_out",
    type=str,
    default="./data/video.mp4",
    help="path to video. (default:./data/video_1_1_m_points.mp4)",
)
args = parser.parse_args()

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    args.path_out, cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (1920, 1080)
)

colormap = mpl.cm.jet
# add a normalization
cNorm = mpl.colors.Normalize(vmin=0, vmax=255)
# init the mapping
scalarMap = mtpltcm.ScalarMappable(norm=cNorm, cmap=colormap)

folder = args.path_to_images

paths_list = []
count = len(
    [
        f
        for f in os.listdir(args.path_to_images)
        if os.path.isfile(os.path.join(args.path_to_images, f))
    ]
)

for img in range(2, count):
    if os.path.join(folder, str(img) + ".jpg") is not None:

        im, p = get_image_and_keypoints(os.path.join(folder, str(img) + ".jpg"))

        im = np.asarray(im)

        j = 0
        for (x, y) in p:
            im = cv2.putText(
                im,
                str(j),
                (int(x), int(y) - 3),
                fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=0.5,
                color=[255, 255, 255],
            )
            cv2.circle(im, (int(x), int(y)), 2, [0, 255, 255], -1)
            j += 1
        cv2.imshow("frame", im)
        out.write(im)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Release everything if job is finished
out.release()
