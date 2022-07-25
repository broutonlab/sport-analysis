import argparse
import os

import cv2
import matplotlib as mpl
import matplotlib.cm as mtpltcm
import numpy as np
from PIL import Image

"""This file can be used for creating a video
if you have the same structure of data as in the
sample dataset """

parser = argparse.ArgumentParser(description=" ")
parser.add_argument(
    "--path_to_images",
    type=str,
    default="./data/raw/1_1m.mp4",
    help="path to video. (default:./data/raw/1_1m.mp4)",
)
parser.add_argument(
    "--path_out",
    type=str,
    default="./data/video.mp4",
    help="path to video. (default:./data/video.mp4)",
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

        im = np.asarray(
            Image.open(os.path.join(folder, str(img) + ".jpg")).convert("RGB")
        )
        print(os.path.join(folder, str(img) + ".jpg"))
        out.write(im)

# Release everything if job is finished
out.release()
