import cv2
from matplotlib import pyplot as plt
import numpy as np

from src.models.decode import decode_card
from src.models.utils import get_pred, image_to_square


def visualize(image, image_keypoint, diameter=2):
    """."""
    image = image.copy()
    j = 0
    for (x, y) in image_keypoint:
        image = cv2.putText(
            image,
            str(j),
            (int(x), int(y) - 3),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=0.5,
            color=[255, 255, 255],
        )
        cv2.circle(image, (int(x), int(y)), diameter, [0, 255, 255], -1)
        j += 1

    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.imshow(image)
    plt.show()


def visualize_tensor(tensor_image, tensor_points):
    """."""
    tensor_points = tensor_points.cpu()
    tensor_points = tensor_points.detach().numpy()
    # 2 for x and y coordinats
    tensor_points = tensor_points.reshape(-1, 2)

    tensor_image = tensor_image.cpu()
    tensor_image = (tensor_image.permute(0, 2, 3, 1)).squeeze(0)
    tensor_image = np.array(tensor_image.detach().numpy(), dtype="uint8")

    visualize(tensor_image, tensor_points)


def visualize_model(samples, num_samples, data, model):
    """."""
    selected_samples = np.random.choice(samples, num_samples, replace=False)
    for sample in selected_samples:
        # get image, image points and predict points by image name
        img, out, poi, im_orig_size, real_headmap = get_pred(sample, data, model)
        heatmaps, offsets = out
        # decode results

        pred_indices_linear, pred_sxy, head = decode_card(
            heatmaps[0].squeeze(0), offsets[0].squeeze(0)
        )
        pred_out = image_to_square(pred_indices_linear, pred_sxy)

        print(poi, "\n", pred_out)

        visualize_tensor(img, poi)
        visualize_tensor(img, pred_out)
