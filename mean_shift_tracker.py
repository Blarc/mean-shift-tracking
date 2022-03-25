import math

import cv2
import numpy as np

from ex2_utils import Tracker, create_epanechnik_kernel, extract_histogram, get_patch, backproject_histogram
from mean_shift import create_coordinates_kernels


def mean_shift(image, patch_position, kernel_size_, q, kernel_, nbins, epsilon):
    coords_x, coords_y = create_coordinates_kernels(kernel_size_)

    x_change = float('inf')
    y_change = float('inf')

    while abs(x_change) > epsilon or abs(y_change) > epsilon:
        patch, mask = get_patch(image, patch_position, kernel_size_)
        p = extract_histogram(patch, nbins, weights=kernel_)
        v = np.sqrt(np.divide(q, p + epsilon))
        w = backproject_histogram(patch, v, nbins)

        x_change = np.divide(np.sum(np.multiply(coords_x, w)), np.sum(w))
        y_change = np.divide(np.sum(np.multiply(coords_y, w)), np.sum(w))

        patch_position = patch_position[0] + x_change, patch_position[1] + y_change

    return int(math.floor(patch_position[0])), int(math.floor(patch_position[1]))


class MeanShiftTracker(Tracker):
    def initialize(self, image, region):

        if len(region) == 8:
            x_ = np.array(region[::2])
            y_ = np.array(region[1::2])
            region = [np.min(x_), np.min(y_), np.max(x_) - np.min(x_) + 1, np.max(y_) - np.min(y_) + 1]
            
        if region[2] % 2 == 0:
            region[2] -= 1
        
        if region[3] % 2 == 0:
            region[3] -= 1

        self.window = max(region[2], region[3]) * self.parameters.enlarge_factor

        left = max(region[0], 0)
        top = max(region[1], 0)

        right = min(region[0] + region[2], image.shape[1] - 1)
        bottom = min(region[1] + region[3], image.shape[0] - 1)

        self.template = image[int(top):int(bottom), int(left):int(right)]
        self.position = (region[0] + region[2] / 2, region[1] + region[3] / 2)
        self.size = (region[2], region[3])
        self.kernel = create_epanechnik_kernel(self.size[0], self.size[1], self.parameters.kernel_sigma)
        self.q = extract_histogram(self.template, self.parameters.histogram_bins, weights=self.kernel)

    def track(self, image):
        left = max(round(self.position[0] - float(self.window) / 2), 0)
        top = max(round(self.position[1] - float(self.window) / 2), 0)

        right = min(round(self.position[0] + float(self.window) / 2), image.shape[1] - 1)
        bottom = min(round(self.position[1] + float(self.window) / 2), image.shape[0] - 1)

        if right - left < self.template.shape[1] or bottom - top < self.template.shape[0]:
            return [self.position[0] + self.size[0] / 2, self.position[1] + self.size[1] / 2, self.size[0],
                    self.size[1]]

        new_x, new_y = mean_shift(image,
                                  self.position,
                                  self.size,
                                  self.q,
                                  self.kernel,
                                  self.parameters.histogram_bins,
                                  self.parameters.epsilon)
        
        # MODEL UPDATE
        self.template, _ = get_patch(image, (new_x, new_y), self.size)
        self.q = (1 - self.parameters.update_alpha) * self.q \
                 + self.parameters.update_alpha * extract_histogram(self.template, self.parameters.histogram_bins, weights=self.kernel)
        
        self.position = (new_x, new_y)

        return [new_x, new_y, self.size[0], self.size[1]]


class MSParams:
    def __init__(self):
        self.enlarge_factor = 1.5
        self.kernel_sigma = 0.5
        self.histogram_bins = 16
        self.epsilon = 1
        self.update_alpha = 0.1
