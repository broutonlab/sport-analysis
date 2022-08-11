import json
import os
import random

import numpy as np
from PIL import Image

from src.options.base_options import num_keypoint


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

    points = np.zeros([num_keypoint, 2])
    # We have 3 default values:
    # for points 1-8 right bottom point,
    # for points 9-18 center bottom point,
    # for points 19-26 left bottom point.
    not_need_nums = [5, 6, 9, 10, 13, 14, 17, 18, 21, 22]
    nums = {5: '7', 6: '8', 7: '11', 8: '12', 9: '15', 10: '16',
            11: '19', 12: '20', 13: '23', 14: '24', 15: '25', 16: '26'}
    nums2 = {'1': 1, '2':2, '3':3, '4':4,'7':  5, '8':  6
                , '11':  7
                , '12':  8
                , '15':  9
                , '16':  10
                , '19':  11
                , '20':  12
                , '23':  13
                , '24':  14
                , '25':  15
                , '26':  16}
    with open(json_name) as f:
        f_data = json.load(f)
        for key in f_data[im_name]:
            if key in nums2:
                points[nums2[key]-1] = f_data[im_name][key]
        # for i in range(points.shape[0]):
            #if i + 1 not in not_need_nums:
                # if str(i + 1) in f_data[im_name]:
                #     points[i] = f_data[im_name][nums[i+1]]
                # elif i < 7:
                #     points[i] = [im.size[0] - 1, im.size[1] - 1]
                # elif 6 < i < 12:
                #     points[i] = [im.size[0] // 2, im.size[1] - 1]
                # else:
                #     points[i] = [0, im.size[1] - 1]
    return im, points

if __name__ == "__main__": 
    a, b = load_paths('./data/raw')
