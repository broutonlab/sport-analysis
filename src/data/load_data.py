import json
import os
import numpy as np

from PIL import Image
import random

from src.constants import IMG_SIZE


def load_paths(path_to):
    subfolders = [f.path for f in os.scandir(path_to) if f.is_dir()]
    paths_list = list()
    for folder in subfolders:
        IMGS = [
            os.path.join(folder, f) if f.split(".")[-1] in ["jpg", "png"] else None
            for f in os.listdir(folder)
        ]
        for img in IMGS:
            paths_list.append(img)
    random.shuffle(paths_list)
    paths_list = [ele for ele in paths_list if ele is not None]
    threshold = int(len(paths_list) * 0.7)
    train, val = paths_list[:threshold], paths_list[threshold:]
    return train, val


def get_points_by_im_path(im_path):
    # получаем путь к json-у и название изображения
    json_name = (
        "/".join(im_path.split("/")[:-1])
        + "/result_"
        + im_path.split("/")[-2]
        + ".json"
    )
    im_name = im_path.split("/")[-1]
    # загружаем все имеющиеся точки
    points = np.zeros([26, 2])
    with open(json_name) as f:
        f_data = json.load(f)
        for i in range(points.shape[0]):
            if str(i + 1) in f_data[im_name]:
                points[i] = f_data[im_name][str(i + 1)]
            else:
                points[i] = [0, 0]
            # if 'playsight' in json_name:
            #     points[i][0] *= (1920 // 485)
            #     points[i][1] *= (1080 // 360)
    return points


def get_image_and_keypoints(name):
    # try find files with argument name and get it
    points = get_points_by_im_path(name)
    im = Image.open(name).convert("RGB")
    return im, points
