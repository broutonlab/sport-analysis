import numpy as np
import torch

from src.options.base_options import CELL_NUM, IMG_SIZE


def do_nett_data(points, num_keypoint):
    """."""
    cell_size = IMG_SIZE // CELL_NUM

    cells_numbers = np.array([])
    indents = np.array([])
    points = points.reshape(num_keypoint, 2)
    for i in range(num_keypoint // 2):
        x = points[i][0].item() // cell_size
        y = points[i][1].item() // cell_size
        cells_numbers = np.append(cells_numbers, [x, y])
        indents = np.append(
            indents,
            [points[i][0].item() - x * cell_size, points[i][1].item() - y * cell_size],
        )

    return cells_numbers, indents


def decode_card(scores, offsets, num_keypoint):
    """."""
    indices = get_indices(scores, num_keypoint)
    indices = indices.reshape(-1)
    if -1 in indices:
        coords = torch.empty(offsets.shape)
        head = None
    else:
        coords, head = get_coords(offsets, indices, num_keypoint)
    return indices, coords, head


def get_indices(scores, num_keypoint):
    """."""
    max_vals_wight, max_indices_wight = torch.max(scores, dim=2)

    max_vals_height, max_indices_height = torch.max(max_vals_wight, 1)

    new_max_indices_matrix = torch.empty(num_keypoint // 2, 2)
    for i in range(num_keypoint // 2):
        j = max_indices_height[i]
        new_max_indices_matrix[i][1] = j
        new_max_indices_matrix[i][0] = max_indices_wight[i][j.item()]

    return new_max_indices_matrix


def get_coords(offsets, indexes_square, num_keypoint):
    """Parameters:
        offsets - a map
        indexes_square - points in forma [x, y]
        num_keypoint - num of points
    """
    cell_size = IMG_SIZE / CELL_NUM

    # Get headmap with map of points
    sxy_where = from_inc_to_class_image(indexes_square, num_keypoint)
    sxy = torch.empty(offsets.shape)

    max_value = 1 * cell_size
    default_value = 0.5 * cell_size
    min_value = 0

    for layer in range(offsets.shape[0]):
        for i in range(offsets.shape[1]):
            for j in range(offsets.shape[2]):
                if sxy_where[layer // 2][i][j] == 1:
                    sx_or_y = min(offsets[layer][i][j], max_value)
                    sx_or_y = max(sx_or_y, min_value)
                    sxy[layer][i][j] = sx_or_y
                else:
                    sxy[layer][i][j] = default_value

    return sxy, sxy_where


def get_coords2(indices_image, indents_square, num_keypoint):
    """."""
    height = CELL_NUM

    cell_size = IMG_SIZE / height

    sxy = torch.empty(num_keypoint * 2, CELL_NUM, CELL_NUM)

    default_value = 0.5 * cell_size

    for layer in range(num_keypoint * 2):
        for i in range(sxy.shape[1]):
            for j in range(sxy.shape[2]):
                if indices_image[layer // 2][i][j] == 1:
                    sxy[layer][i][j] = indents_square[layer]
                else:
                    sxy[layer][i][j] = default_value

    return sxy


def from_inc_to_class_image(indexes, num_keypoint):
    """."""
    class_image = np.zeros((num_keypoint + 1, CELL_NUM, CELL_NUM))
    indexes = indexes.reshape(-1, 2)
    class_image[num_keypoint] = np.full((CELL_NUM, CELL_NUM), fill_value=1)
    for i in range(len(indexes)):
        class_image[i][int(indexes[i][1])][int(indexes[i][0])] = 1
        class_image[num_keypoint][int(indexes[i][1])][int(indexes[i][0])] = 0
    return class_image
