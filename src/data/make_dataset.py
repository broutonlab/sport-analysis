import albumentations as A
from torch.utils.data.dataset import Dataset
import numpy as np
import torch

from src.options.base_options import IMG_SIZE, device
from src.data.load_data import get_image_and_keypoints
from src.models.decode import do_nett_data, from_inc_to_class_image, get_coords2


""" Augmentation"""
augment = A.Compose(
    [A.Resize(IMG_SIZE, IMG_SIZE, interpolation=3)],
    # A.HorizontalFlip(p=0.5)],
    keypoint_params=A.KeypointParams(format="xy"),
)


class KeypointDataset(Dataset):
    def __init__(self, samples):
        self.data = samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        :param index: int
        :return:dictionary with:
            augmented image, tensor
            points for augmented image: tensor
            headmap that matches the original points: tensor
            offset that matches the original points: tensor
        """
        img, points_list = get_image_and_keypoints(self.data[index])
        img_ndarray = np.array(img)
        # The markup can have values equal to the dimensions of the image,
        # which is not allowed for rendering,
        # so it's better to change these values a little right away
        for i in range(points_list.shape[0]):
            if points_list[i][0] == img.size[0]:
                points_list[i][0] = points_list[i][0] - 1
            if points_list[i][1] == img.size[1]:
                points_list[i][1] = points_list[i][1] - 1
        augment_img_keypoints = augment(image=img_ndarray, keypoints=points_list)
        image_final, keypoints_final = (
            augment_img_keypoints["image"],
            augment_img_keypoints["keypoints"],
        )

        keypoints_final = np.array(keypoints_final).reshape(-1)

        real_indices_square, real_sxy_square = do_nett_data(keypoints_final)
        real_headmap = from_inc_to_class_image(real_indices_square)
        real_offset = get_coords2(real_headmap, real_sxy_square).to(device)

        image_final = torch.tensor(image_final.copy(), dtype=torch.float64).cpu()
        # For visualize real points_list after code and decode
        # new_points = image_to_square(real_indices_square, real_offset)
        # visualize_tensor(image_final, new_points)
        # print('real1', keypoints_final, 'real2', new_points)
        image_final = (image_final.permute(2, 0, 1)).to(device)

        return {
            "image": image_final,
            "keypoints": torch.tensor(keypoints_final.copy(), dtype=torch.float64).to(device),
            "real_headmap": torch.tensor(real_headmap.copy(), dtype=torch.float64).to(device),
            "real_offset": real_offset.double(),
        }

    def get_index(self, name):
        """For visualisation by names,
        return the number of image by its name"""
        return self.data.index(name)

    def get_image_size(self, index):
        img, points = get_image_and_keypoints(self.data[index])
        return img.size
