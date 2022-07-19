import argparse
import cv2
import matplotlib as mpl
import matplotlib.cm as mtpltcm

from src.constants import INF_IMG_SIZE
from src.models.decode import decode_card
from src.inference.utils_instance import preparation, get_model

from src.models.utils import image_to_square

parser = argparse.ArgumentParser(
    description=" "
)
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
    help="path to video. (default:./data/video_out.mp4)",
)
parser.add_argument(
    "--weights",
    type=str,
    default="./checkpoints/field_keypoints_best.pd",
    help="path to video. (default:./checkpoints/field_keypoints_best.pd)",
)
args = parser.parse_args()

model = get_model(args.weights)

cap = cv2.VideoCapture(args.video_in)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # DIVX # mp4v # cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter(args.path_out, cv2.VideoWriter_fourcc(*"mp4v"),
                      25.0, (INF_IMG_SIZE, INF_IMG_SIZE))

# import the color source
# initialize the colormap (jet)
colormap = mpl.cm.jet
# add a normalization
cNorm = mpl.colors.Normalize(vmin=0, vmax=255)
# init the mapping
scalarMap = mtpltcm.ScalarMappable(norm=cNorm, cmap=colormap)

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        image, tensor = preparation(frame)

        out_model = model(tensor)

        heatmaps, offsets = out_model
        # decode results
        pred_indices_linear, pred_Sxy, head = decode_card(
            heatmaps[0].squeeze(0), offsets[0].squeeze(0)
        )
        pred_out = image_to_square(pred_indices_linear, pred_Sxy)

        for i, (x, y) in enumerate(pred_out.reshape(-1, 2)):
            image = cv2.putText(image, str(i), (int(x), int(y) - 3),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 0, 0], 2, cv2.LINE_AA)
            cv2.circle(image, (int(x), int(y)), 2, [0, 255, 255], -1)

        cv2.imshow("frame", image)

        out.write(image)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# Release everything when the job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
