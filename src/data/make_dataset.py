import albumentations as A
from torch.utils.data.dataset import Dataset
import numpy as np
import torch

from src.constants import IMG_SIZE, device
from src.data.load_data import get_image_and_keypoints
from src.models.decode import do_nett_data, from_inc_to_class_image, get_coords2


# augmentation for resize and flip cards
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
        # img - изображение в формате PIL
        # data - точки в формате 26х2
        img, data = get_image_and_keypoints(self.data[index])
        img_ndarray = np.array(img)
        # проходим по всем точкам и проверяем на размерность
        for i in range(data.shape[0]):
            if data[i][0] == img.size[0]:
                data[i][0] = data[i][0] - 1
                # print(name_im, ' has wrong data: ', img.size[0], ' => ', data[i][0])
            if data[i][1] == img.size[1]:
                data[i][1] = data[i][1] - 1
        augment_img_keypoints = augment(image=img_ndarray, keypoints=data)
        # .transpose(-1, 0, 1)
        image_final, keypoints_final = (
            augment_img_keypoints["image"],
            augment_img_keypoints["keypoints"],
        )
        # if keypoints_final[0][0] < keypoints_final[1][0]:
        #     keypoints_final = [keypoints_final[1], keypoints_final[0], keypoints_final[3], keypoints_final[2]]
        # More on why this reshaping later.
        keypoints_final = np.array(keypoints_final).reshape(-1)
        # get indiced of cells and indents
        real_indices_square, real_sxy_square = do_nett_data(keypoints_final)

        real_headmap = from_inc_to_class_image(real_indices_square)

        real_offset = get_coords2(real_headmap, real_sxy_square).to(device)

        image_final = torch.tensor(image_final.copy(), dtype=torch.float64).cpu()
        """ for visualize real data after code and decode
        new_points = image_to_square(real_indices_square, real_offset)
        visualize_tensor(image_final, new_points)
        print('real1', keypoints_final, 'real2', new_points)"""
        image_final = (image_final.permute(2, 0, 1)).to(device)
        # / 255.0

        return {
            "image": image_final,
            "keypoints": torch.tensor(keypoints_final.copy(), dtype=torch.float64).to(device),
            "real_headmap": torch.tensor(real_headmap.copy(), dtype=torch.float64).to(device),
            "real_offset": real_offset.double(),
        }

    # для визуализации по именам, функция возвращает номер по имени изображения
    def get_index(self, name):
        return self.data.index(name)

    def get_image_size(self, index):
        img, points = get_image_and_keypoints(self.data[index])
        return img.size
