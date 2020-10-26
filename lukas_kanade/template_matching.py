import os
import cv2

import numpy as np

from lukas_kanade.utils import *


def ssd_match(image, template):
    h_image, w_image = image.shape
    h_template, w_template = template.shape

    ssd_scores = np.zeros(shape=(h_image - h_template, w_image - w_template))

    for y in range(h_image - h_template):
        for x in range(w_image - w_template):
            crop = get_crop_by_coords(image, (x, y, w_template, h_template))
            difference = np.sum((crop - template) ** 2)
            ssd_scores[y, x] = difference

    min_ssd_index = ssd_scores.argmin()
    point_coords = np.unravel_index(min_ssd_index, ssd_scores.shape)
    return point_coords


def ncc_match(image, template):
    h_image, w_image = image.shape
    h_template, w_template = template.shape

    ncc_scores = np.zeros(shape=(h_image - h_template, w_image - w_template))

    for y in range(h_image - h_template):
        for x in range(w_image - w_template):
            crop = get_crop_by_coords(image, (x, y, w_template, h_template))
            numerator = (crop * template).sum()
            denominator = np.sqrt((crop ** 2).sum()) * np.sqrt((template ** 2).sum())
            if denominator:
                ncc_scores[y, x] = numerator / denominator
            else:
                ncc_scores[y, x] = 0

    max_ncc_index = ncc_scores.argmax()
    point_coords = np.unravel_index(max_ncc_index, ncc_scores.shape)
    return point_coords


def sad_match(image, template):
    h_image, w_image = image.shape
    h_template, w_template = template.shape

    sad_scores = np.zeros(shape=(h_image - h_template, w_image - w_template))

    for y in range(h_image - h_template):
        for x in range(w_image - w_template):
            crop = get_crop_by_coords(image, (x, y, w_template, h_template))
            difference = np.sum(np.abs(crop - template))
            sad_scores[y, x] = difference

    min_sad_index = sad_scores.argmin()
    point_coords = np.unravel_index(min_sad_index, sad_scores.shape)
    return point_coords


if __name__ == '__main__':
    path_to_dataset = 'data/Girl/img'
    images_names = sorted(os.listdir(path_to_dataset))
    print(images_names)

    for i, image_name in enumerate(images_names):

        image_path = os.path.join(path_to_dataset, image_name)

        image_bw = cv2.imread(image_path, 0)
        # plot_image(image)

        if i == 0:
            (x, y, w, h) = select_ROI(image_bw)
            template = get_crop_by_coords(image_bw, (x, y, w, h))
            print(template)
            print(template.shape)
            plot_image(template)

        # point = ssd_match(image_bw, template)
        # point = ncc_match(image_bw, template)
        point = sad_match(image_bw, template)

        image_rgb = read_bgr_image(image_path)
        visualize_bbox(image_rgb, point, template.shape[1], template.shape[0])

