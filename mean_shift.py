import math

import cv2
import numpy as np

from ex2_utils import generate_responses_1, show_image, get_patch, generate_responses_2


def create_coordinates_kernels(size):
    # Make sure the kernel size is odd.
    width = int(size[0])
    if width % 2 == 0:
        width -= 1

    height = int(size[1])
    if height % 2 == 0:
        height -= 1

    kernel_x = np.zeros((height, width))
    kernel_y = np.zeros((height, width))

    half_width = width // 2
    for index, value in enumerate(range(-half_width, half_width + 1)):
        kernel_x[:, index] = value

    half_height = height // 2
    for index, value in enumerate(range(-half_height, half_height + 1)):
        kernel_y[index, :] = value

    return kernel_x, kernel_y


def mean_shift(image, patch_position, kernel_size_, epsilon):
    coords_x, coords_y = create_coordinates_kernels(kernel_size_)

    converged = False
    iters_ = 0
    steps_ = [patch_position]
    while not converged:
        patch, _ = get_patch(image, patch_position, kernel_size_)

        x_change = np.divide(np.sum(np.multiply(coords_x, patch)), np.sum(patch))
        y_change = np.divide(np.sum(np.multiply(coords_y, patch)), np.sum(patch))

        if abs(x_change) < epsilon and abs(y_change) < epsilon:
            converged = True

        patch_position = patch_position[0] + x_change, patch_position[1] + y_change

        floored_position = (math.floor(patch_position[0]), math.floor(patch_position[1]))
        if steps_[-1] != floored_position:
            steps_.append(floored_position)

        iters_ += 1

    return int(math.floor(patch_position[0])), int(math.floor(patch_position[1])), steps_, iters_


def run_example(img):
    end_x, end_y, steps, iters = mean_shift(img, (45, 60), (5, 5), 0.01)
    responses = cv2.circle(img * 400, steps[0], radius=0, color=(255, 0, 0), thickness=-1)
    print(f'{iters} & {steps[-1]} & {img[steps[-1][1], steps[-1][0]]:.5f}')
    for step in steps[1:]:
        responses = cv2.circle(responses, step, radius=0, color=(0, 255, 0), thickness=-1)
    show_image(cv2.resize(responses, (300, 300)), 0, "Mean shift algorithm")


def run_parameters_comparison(img):
    end_x, end_y, steps, iters = mean_shift(img, (30, 60), (3, 3), 0.01)
    print(f'Kernel size 3x3 & {iters} & {steps[-1]} & {img[steps[-1][1], steps[-1][0]]:.5f} \\\\\n\\hline')
    end_x, end_y, steps, iters = mean_shift(img, (30, 60), (5, 5), 0.01)
    print(f'Kernel size 5x5 & {iters} & {steps[-1]} & {img[steps[-1][1], steps[-1][0]]:.5f} \\\\\n\\hline')
    end_x, end_y, steps, iters = mean_shift(img, (30, 60), (7, 7), 0.01)
    print(f'Kernel size 7x7 & {iters} & {steps[-1]} & {img[steps[-1][1], steps[-1][0]]:.5f} \\\\\n\\hline')
    print('\\hhline{|=|=|=|=|}')
    end_x, end_y, steps, iters = mean_shift(img, (30, 60), (5, 5), 0.01)
    print(f'Epsilon 0.01 & {iters} & {steps[-1]} & {img[steps[-1][1], steps[-1][0]]:.5f} \\\\\n\\hline')
    end_x, end_y, steps, iters = mean_shift(img, (30, 60), (5, 5), 0.05)
    print(f'Epsilon 0.05 & {iters} & {steps[-1]} & {img[steps[-1][1], steps[-1][0]]:.5f} \\\\\n\\hline')
    end_x, end_y, steps, iters = mean_shift(img, (30, 60), (5, 5), 0.1)
    print(f'Epsilon 0.1 & {iters} & {steps[-1]} & {img[steps[-1][1], steps[-1][0]]:.5f} \\\\\n\\hline')
    print('\\hhline{|=|=|=|=|}')
    end_x, end_y, steps, iters = mean_shift(img, (35, 40), (5, 5), 0.01)
    print(f'Start (35, 40) & {iters} & {steps[-1]} & {img[steps[-1][1], steps[-1][0]]:.5f} \\\\\n\\hline')
    end_x, end_y, steps, iters = mean_shift(img, (60, 30), (5, 5), 0.01)
    print(f'Start (60, 30) & {iters} & {steps[-1]} & {img[steps[-1][1], steps[-1][0]]:.5f} \\\\\n\\hline')
    end_x, end_y, steps, iters = mean_shift(img, (75, 75), (5, 5), 0.01)
    print(f'Start (75, 75) & {iters} & {steps[-1]} & {img[steps[-1][1], steps[-1][0]]:.5f} \\\\\n\\hline')


if __name__ == '__main__':
    responses = generate_responses_2()
    print(f'(50, 70): {responses[50][70]:.5f}')
    print(f'(70, 50): {responses[70][50]:.5f}')
    # run_parameters_comparison(responses)
    run_example(responses)
