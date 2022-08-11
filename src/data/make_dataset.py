import albumentations as A
import numpy as np
import torch
from torch.utils.data.dataset import Dataset

from src.data.load_data import get_image_and_keypoints
from src.models.decode import do_nett_data, from_inc_to_class_image, get_coords2
from src.options.base_options import device, IMG_SIZE


def augmentation(image, keypoint):
    """Augmentation"""
    augment = A.Compose(
        [A.Resize(IMG_SIZE, IMG_SIZE, interpolation=3)],
        # A.HorizontalFlip(p=0.5)],
        keypoint_params=A.KeypointParams(format="xy"),
    )
    return augment(image=image, keypoints=keypoint)


class KeypointDataset(Dataset):
    """."""

    def __init__(self, samples, num_keypoint=16):
        """."""
        self.data = samples
        self.num_keypoint = num_keypoint

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
        augment_img_keypoint = augmentation(image=img_ndarray, keypoint=points_list)
        image_final, keypoint_final = (
            augment_img_keypoint["image"],
            augment_img_keypoint["keypoints"],
        )

        keypoint_final = np.array(keypoint_final).reshape(-1)
        real_indices_square, real_sxy_square = do_nett_data(
            keypoint_final, self.num_keypoint
        )
        real_headmap = from_inc_to_class_image(real_indices_square, self.num_keypoint)
        real_offset = get_coords2(real_headmap, real_sxy_square, self.num_keypoint).to(
            device
        )

        image_final = torch.tensor(image_final.copy(), dtype=torch.float64).cpu()
        # For visualize real points_list after code and decode
        # new_points = image_to_square(real_indices_square, real_offset)
        # visualize_tensor(image_final, new_points)
        # print('real1', keypoints_final, 'real2', new_points)
        image_final = (image_final.permute(2, 0, 1)).to(device)

        return {
            "image": image_final,
            "keypoints": torch.tensor(keypoint_final.copy(), dtype=torch.float64).to(
                device
            ),
            "real_headmap": torch.tensor(real_headmap.copy(), dtype=torch.float64).to(
                device
            ),
            "real_offset": real_offset.double(),
        }

    def get_index(self, name):
        """For visualisation by names,
        return the number of image by its name
        """
        return self.data.index(name)

    def get_image_size(self, index):
        """."""
        img, points = get_image_and_keypoints(self.data[index])
        return img.size
