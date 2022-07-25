# YOLOv5 Tracking

Simplest possible example of tracking. Based on [YOLOv5-pip](https://github.com/fcakyon/yolov5-pip).

## Instructions

1. Activate venv `poetry shell`
2. Run `python demo.py <video file>`
3. Bonus: Use additional arguments `--detector_path`, `--img_size`, `--iou_thres`,`--conf_thres`, `--classes`, `--track_points` as you wish.

## Explanation

This example tracks objects using a single or double point per detection: the centroid or the two corners of the bounding boxes around objects returned by YOLOv5.

## Tracking people

![Norfair tracking pedestrians using YOLOv5](assets/demo.gif)
