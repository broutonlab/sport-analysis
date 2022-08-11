import argparse

import cv2
import matplotlib as mpl
import matplotlib.cm as mtpltcm

from src.inference.utils_instance import get_model, preprocessing, points_postprocessing
from src.models.decode import decode_card
from src.models.utils import image_to_square
from src.options.base_options import INF_IMG_SIZE

parser = argparse.ArgumentParser(description=" ")

parser.add_argument(
    "--video_in",
    type=str,
    default="./data/video.mp4",
    help="path to video. (default:./data/video.mp4)",
)
parser.add_argument(
    "--path_out",
    type=str,
    default="./data/video_out.mp4",
    help="path to output video. (default:./data/video_out.mp4)",
)
parser.add_argument(
    "--weights",
    type=str,
    default="./models/exp0/field_keypoint_best.pd",
    help="path to weights. (default:./checkpoints/field_keypoint_best.pd)",
)
parser.add_argument(
    "--num_keypoint",
    type=str,
    default=26,
    help="number of keypoints. (default: 26 )",
)
args = parser.parse_args()

model = get_model(args.weights)

cap = cv2.VideoCapture(args.video_in)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)  # DIVX # mp4v # cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter(
    args.path_out, cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (360, 540)#INF_IMG_SIZE, INF_IMG_SIZE)
)

# Initialize the colormap (jet)
colormap = mpl.cm.jet
# Add a normalization
cNorm = mpl.colors.Normalize(vmin=0, vmax=255)
# Init the mapping
scalarMap = mtpltcm.ScalarMappable(norm=cNorm, cmap=colormap)

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        image, tensor = preprocessing(frame)

        out_model = model(tensor)

        heatmaps, offsets = out_model
        # Decode results
        pred_indices_linear, pred_Sxy, head = decode_card(
            heatmaps[0].squeeze(0), offsets[0].squeeze(0), 26
        )
        pred_out = image_to_square(pred_indices_linear, pred_Sxy)

        pred_out0 = points_postprocessing(frame, image, pred_out)

        im = frame
        for i, (x, y) in enumerate(pred_out0.reshape(-1, 2)):
            im = cv2.putText(
                frame,
                str(i),
                (int(x), int(y) - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                [0, 0, 0],
                2,
                cv2.LINE_AA,
            )
            cv2.circle(im, (int(x), int(y)), 2, [0, 255, 255], -1)

        cv2.imshow("frame", im)

        out.write(im)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# Release everything when the job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
