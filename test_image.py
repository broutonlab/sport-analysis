import PIL.Image

from src.inference.utils_instance import get_model, preprocessing
from src.models.decode import decode_card
from src.models.utils import image_to_square
from src.visualisation.visualize import visualize_tensor

num_keypoint = 16

path_to_image = "data/raw/1_1m/3.jpg"

image = PIL.Image.open(path_to_image)

model = get_model("models/exp0/field_keypoints_best.pd")

resized_image, tensor = preprocessing(image)

out_model = model(tensor)

heatmaps, offsets = out_model
# decode results
pred_indices_linear, pred_Sxy, head = decode_card(
    heatmaps[0].squeeze(0), offsets[0].squeeze(0), num_keypoint=num_keypoint
)
pred_out = image_to_square(pred_indices_linear, pred_Sxy)

visualize_tensor(tensor, pred_out)
