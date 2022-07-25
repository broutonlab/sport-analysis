import json
import os
import random

import numpy as np
from PIL import Image


def load_paths(path_to):
    """Get list of all names in the dataset,
    then split it into lists of train and validation

    We have images from a few videos, so we need to get images from a few
    directories
    """
    subfolders = [f.path for f in os.scandir(path_to) if f.is_dir()]
    paths_list = []

    for folder in subfolders:
        imgs = [
            os.path.join(folder, f) if f.split(".")[-1] in ["jpg", "png"] else None
            for f in os.listdir(folder)
        ]
        for img in imgs:
            paths_list.append(img)

    random.shuffle(paths_list)

    paths_list = [ele for ele in paths_list if ele is not None]
    threshold = int(len(paths_list) * 0.7)
    train, val = paths_list[:threshold], paths_list[threshold:]

    return train, val


def get_image_and_keypoints(im_path):
    """Get path to json file and image.
    Then load all the points we have and store them in one array.
    If we don't have a point, then it will have x=0, y=0
    """
    json_name = (
        "/".join(im_path.split("/")[:-1])
        + "/result_"
        + im_path.split("/")[-2]
        + ".json"
    )
    im_name = im_path.split("/")[-1]
    im = Image.open(im_path).convert("RGB")

    points = np.zeros([26, 2])
    # We have 3 default values:
    # for points 1-8 right bottom point,
    # for points 9-18 center bottom point,
    # for points 19-26 left bottom point.
    with open(json_name) as f:
        f_data = json.load(f)
        for i in range(points.shape[0]):
            if str(i + 1) in f_data[im_name]:
                points[i] = f_data[im_name][str(i + 1)]
            elif i < 9:
                points[i] = [im.size[0] - 1, im.size[1] - 1]
            elif 8 < i < 19:
                points[i] = [im.size[0] // 2, im.size[1] - 1]
            else:
                points[i] = [0, im.size[1] - 1]
    return im, points
