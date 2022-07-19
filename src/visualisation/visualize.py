import cv2
from matplotlib import pyplot as plt
import numpy as np

from src.constants import POINT
from src.data.make_dataset import get_image_and_keypoints
from src.models.decode import decode_card
from src.models.utils import get_pred, image_to_square


def visualize(image, image_keypoint, diameter=2):
    image = image.copy()
    j = 0
    for (x, y) in image_keypoint:
        image = cv2.putText(image, str(j), (int(x), int(y) - 3), fontFace=cv2.FONT_HERSHEY_PLAIN,
                            fontScale=0.5, color=[255, 255, 255])
        cv2.circle(image, (int(x), int(y)), diameter, [0, 255, 255], -1)
        j += 1

    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.imshow(image)
    plt.show()


def visualize_tensor(tensor_image, tensor_points):
    tensor_points = tensor_points.cpu()
    tensor_points = tensor_points.detach().numpy()
    # 2 for x and y coordinats
    tensor_points = tensor_points.reshape(-1, 2)  # * IMG_SIZE

    tensor_image = tensor_image.cpu()
    tensor_image = (tensor_image.permute(0, 2, 3, 1)).squeeze(0)
    #  * 255.0
    tensor_image = np.array(tensor_image.detach().numpy(), dtype="uint8")

    visualize(tensor_image, tensor_points)


def visualize_dataset(samples, num_samples):
    selected_samples = np.random.choice(samples, num_samples, replace=False)
    for sample in selected_samples:
        # get image and image points by image name
        im, data = get_image_and_keypoints(sample)
        keypoint = data["shapes"][0][POINT]
        im = np.asarray(im)
        visualize(im, keypoint)


def visualize_model(samples, num_samples, data, model):
    selected_samples = np.random.choice(samples, num_samples, replace=False)
    for sample in selected_samples:
        # get image, image points and predict points by image name
        img, out, poi, im_orig_size, real_headmap = get_pred(sample, data, model)
        heatmaps, offsets = out
        # decode results

        pred_indices_linear, pred_sxy, head = decode_card(
            heatmaps[0].squeeze(0), offsets[0].squeeze(0)
        )
        """ print('Headmaps:\n', real_headmap[4], head[4])
            print('Headmaps:\n', pred_sxy[0], pred_sxy[1],  head[0],
            pred_sxy[2],pred_sxy[3], head[1], pred_sxy[4], pred_sxy[5],
            head[2], pred_sxy[6],pred_sxy[7], head[3],  head[4])"""
        pred_out = image_to_square(pred_indices_linear, pred_sxy)

        print(poi, "\n", pred_out)

        visualize_tensor(img, poi)
        visualize_tensor(img, pred_out)


def grafics(train_loss, val_loss):
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color="orange", label="train loss")
    plt.plot(val_loss, color="red", label="validataion loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    # plt.savefig(f"{OUTPUT_PATH}/loss.png")
    plt.show()
