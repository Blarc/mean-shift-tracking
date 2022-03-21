import math

import cv2
import numpy as np

from ex2_utils import generate_responses_1, show_image, get_patch


def create_coordinates_kernels(size):
    # Make sure the kernel size is odd.
    if size % 2 == 0:
        size -= 1

    kernel_x = np.zeros((size, size))
    kernel_y = np.zeros((size, size))

    half = size // 2
    for index, value in enumerate(range(-half, half + 1)):
        kernel_x[:, index] = value
        kernel_y[index, :] = value

    return kernel_x, kernel_y


def mean_shift(image, patch_position, kernel_size_, epsilon):
    coords_x, coords_y = create_coordinates_kernels(kernel_size_)

    x_change = float('inf')
    y_change = float('inf')

    steps_ = [patch_position]
    while abs(x_change) > epsilon or abs(y_change) > epsilon:
        patch, _ = get_patch(image, patch_position, (kernel_size_, kernel_size_))

        x_change = np.divide(np.sum(np.multiply(coords_x, patch)), np.sum(patch))
        y_change = np.divide(np.sum(np.multiply(coords_y, patch)), np.sum(patch))

        patch_position = patch_position[0] + x_change, patch_position[1] + y_change

        floored_position = (math.floor(patch_position[0]), math.floor(patch_position[1]))
        if steps_[-1] != floored_position:
            steps_.append(floored_position)

    return int(math.floor(patch_position[0])), int(math.floor(patch_position[1])), steps_


if __name__ == '__main__':
    responses = generate_responses_1() * 600
    
    end_x, end_y, steps = mean_shift(responses, (80, 80), 5, 0.01)

    responses = cv2.circle(responses, steps[0], radius=0, color=(255, 0, 0), thickness=-1)
    for step in steps[1:]:
        responses = cv2.circle(responses, step, radius=0, color=(0, 255, 0), thickness=-1)

    show_image(cv2.resize(responses, (300, 300)), 0, "Mean shift algorithm")