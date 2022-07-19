import torch

from src.constants import IMG_SIZE, NUM_KEYPOINTS, device


def get_pred(name, data, model):
    # get index of dataset item for get it
    inx = data.get_index(name)
    im_orig_size = data.get_image_size(inx)
    dat = data[inx]

    im, poi, real_headmap = (
        dat["image"].to(device),
        dat["keypoints"].to(device),
        dat["real_headmap"].to(device),
    )
    im = im.reshape(1, im.shape[0], im.shape[1], im.shape[2])

    model.eval()
    outputs = model(im)
    return im, outputs, poi, im_orig_size, real_headmap


def image_to_square(indices_square, clear_offset):
    cell_size = IMG_SIZE / clear_offset.shape[2]
    indices_square = indices_square.reshape(-1, 2)
    sxy_square = torch.empty(NUM_KEYPOINTS, dtype=torch.float64).to(device)
    index_sxy_square = 0
    for i in range(clear_offset.shape[1]):
        for j in range(clear_offset.shape[2]):
            for inx in range(indices_square.shape[0]):
                if indices_square[inx][0] == j and indices_square[inx][1] == i:
                    current_coord = 0
                    for layer in [inx * 2, inx * 2 + 1]:
                        # print('layer:', layer, '\nij:', j, i, '\nSxy[layer][j][i]:', Sxy[layer][j][i])
                        s_x_y = clear_offset[layer][j][i]
                        sxy_square[index_sxy_square] = (
                            cell_size * indices_square[inx][current_coord] + s_x_y
                        )
                        current_coord += 1
                        index_sxy_square += 1
    return sxy_square
