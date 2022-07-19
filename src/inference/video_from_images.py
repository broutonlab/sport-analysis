import numpy as np
import cv2
import os
import matplotlib as mpl
import matplotlib.cm as mtpltcm

# Define the codec and create VideoWriter object
from PIL import Image

# DIVX # mp4v # cv2.VideoWriter_fourcc(*'MJPG')
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("~/video_playsight.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (1920, 1080))

# import the color source
# initialize the colormap (jet)
colormap = mpl.cm.jet
# add a normalization
cNorm = mpl.colors.Normalize(vmin=0, vmax=255)
# init the mapping
scalarMap = mtpltcm.ScalarMappable(norm=cNorm, cmap=colormap)

path_to = "~/data/raw"
# [f.path for f in os.scandir(path_to) if f.is_dir()]
subfolders = ["~/data/raw/playsight", "~/data/raw/2_1m"]
paths_list = list()
for folder in subfolders:
    IMGS = [
        os.path.join(folder, f) if f.split(".")[-1] in ["jpg", "png"] else None
        for f in sorted(os.listdir(folder))
    ]
    for img in range(2, 904):
        if os.path.join(folder, str(img) + ".jpg") is not None:

            im = np.asarray(
                Image.open(os.path.join(folder, str(img) + ".jpg")).convert("RGB")
            )
            print(os.path.join(folder, str(img) + ".jpg"))
            out.write(im)
    break

# Release everything if job is finished
out.release()
