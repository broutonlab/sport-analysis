import albumentations as A
import numpy as np
import torch
from torch import nn

from src.models.model import MobileNetV1
from src.options.base_options import device, INF_IMG_SIZE

augment = A.Compose([A.Resize(INF_IMG_SIZE, INF_IMG_SIZE, interpolation=3)])


def get_model(path="./checkpoints/field_keypoint_best.pd", model_version=50):
    """Change the last 2 layers
        NUM_KEYPOINT'S+1 = the number of layers in the headmap,
        the number of points you are looking for and one more for the background,
        NUM_KEYPOINT*2 = number of required coordinates, two for each point
    """
    model = MobileNetV1(model_version)
    model.heatmap = nn.Conv2d(model.last_depth, 27, 2, 2).double().to(device)
    model.offset = nn.Conv2d(model.last_depth, 52, 2, 2).double().to(device)
    model.load_state_dict(torch.load(path))

    model.to(device)
    model.eval()
    return model


def preprocessing(image):
    """."""
    img_ndarray = np.array(image)
    augment_img_keypoints = augment(image=img_ndarray)

    image_final = augment_img_keypoints["image"]
    im_final = image_final.reshape(
        1, image_final.shape[0], image_final.shape[1], image_final.shape[2]
    )
    # More on why this reshaping later.
    im_ten = torch.tensor(im_final.copy(), dtype=torch.float64).cpu()
    """ for visualize real data after code and decode
        new_points = image_to_square(real_indices_square, real_offset)
        visualize_tensor(image_final, new_points)
        print('real1', keypoint_final, 'real2', new_points)"""
    im_ten = (im_ten.permute(0, 3, 1, 2)).to(device)
    # / 255.0

    return image_final, im_ten
