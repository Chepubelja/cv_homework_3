"""
Module with all the useful utils.
"""

import os

import cv2
import numpy as np

import matplotlib.pyplot as plt


def read_rgb_image(path_to_image):
    """
    Reads RGB image.
    :param path_to_image: Path to image to read.
    :return: RGB image (np.ndarray)
    """
    image_bgr = cv2.imread(path_to_image)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb


def plot_image(image):
    """
    Plots given image.
    :param image: Image to plot (np.ndarray).
    """
    cv2.imshow('image', image)
    cv2.waitKey(0)


def plot_hist(img):
    """
    Plots image histogram per channel with cv2.
    :param img: Target image.
    """
    color = ('r', 'g', 'b')
    for i, col in enumerate(color):
        histr = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    plt.show()


def calculate_precision(matches, correct_matches):
    """
    Calculates precision using retrieved and correct matches.
    :param matches: Retrieved matches.
    :param correct_matches: Correct matches.
    :return: List of precisions.
    """
    precisions = []

    for i, match in enumerate(matches):
        if match[0] in correct_matches:
            precision = (len(precisions) + 1) / (i + 1)
            precisions.append(precision)
        else:
            continue

    return precisions


def calculate_recall(correct_matches):
    """
    Calculates recall using correct matches.
    :param correct_matches: Correct matches.
    :return: List of recalls.
    """
    recalls = []
    for i in range(len(correct_matches)):
        recalls.append((i + 1) / len(correct_matches))
    return recalls


def plot_pr_curve(query_name, precision, recall):
    """
    Plots Precision-Recall curve for the given query.
    :param query_name: Name of the query image.
    :param precision: Precision for the query image.
    :param recall: Recall for the query image.
    """
    plt.plot(recall, precision, marker='.')

    # Axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')

    # Title
    plt.title(f'Query: {query_name}')

    # Save plot
    path_to_plots = 'plots/curves'
    if not os.path.exists(path_to_plots):
        os.makedirs(path_to_plots)

    plt.savefig(f"{os.path.join(path_to_plots, query_name.split('.')[0])}_precision_recall_curve.png")

    # Show the plot
    plt.show()


def plot_matches(query, matches, path_to_data):
    """
    Plots matches for the given query image.
    :param query: Name of the query image.
    :param matches: Retrieved matches using histogram comparison method.
    :param path_to_data: Path to dataset with all images.
    """
    path_to_query = os.path.join(path_to_data, query)
    query_image = read_rgb_image(path_to_query)

    matches_imgs = [read_rgb_image(os.path.join(path_to_data, match[0])) for match in matches],
    distances = [match[1] for match in matches]

    path_to_plots = 'plots/matches'
    if not os.path.exists(path_to_plots):
        os.makedirs(path_to_plots)

    fig, ax = plt.subplots(11, 1, figsize=(12, 42))

    for i in range(11):
        if i == 0:
            ax[i].imshow(query_image, interpolation='nearest')
            ax[i].set_title(f'Query image: {query}', fontdict={'fontsize': 16, 'fontweight': 'medium'})
        else:
            ax[i].imshow(matches_imgs[0][i - 1], interpolation='nearest')
            ax[i].set_title(f'Match #{i}: {matches[i - 1][0]} - Dist {round(distances[i - 1], 3)}',
                            fontdict={'fontsize': 16, 'fontweight': 'medium'})

    plt.savefig(f"{os.path.join(path_to_plots, query.split('.')[0])}_matches.png")
    plt.show()


def calculate_hist_per_channel(image):
    """
    Calculates normalized histograms per channel (R, G, B) for the given image.
    :param image: Target image.
    :return: np.array with calculated histogram for each channel.
    """
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]

    r_calc = dict(zip(*np.unique(r.flatten(), return_counts=True)))
    r_hist = [r_calc[i] if i in r_calc else 0 for i in range(256)]
    r_normalized = r_hist / np.linalg.norm(r_hist)

    g_calc = dict(zip(*np.unique(g.flatten(), return_counts=True)))
    g_hist = [g_calc[i] if i in g_calc else 0 for i in range(256)]
    g_normalized = g_hist / np.linalg.norm(g_hist)

    b_calc = dict(zip(*np.unique(b.flatten(), return_counts=True)))
    b_hist = [b_calc[i] if i in b_calc else 0 for i in range(256)]
    b_normalized = b_hist / np.linalg.norm(b_hist)

    return np.vstack([r_normalized, g_normalized, b_normalized])


def l2_distance(hist_1, hist_2):
    """
    Calculates L2 distance (Euclidean distance) between 2 histograms
    :param hist_1: Histogram of the first image.
    :param hist_2: Histogram of the second image.
    :return: L2 distance between 2 histograms.
    """
    return np.linalg.norm(hist_1 - hist_2)
