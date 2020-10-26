import cv2


def select_ROI(frame):
    roi = cv2.selectROI(frame)
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    return roi


def plot_image(image):
    """
    Plots given image.
    :param image: Image to plot (np.ndarray).
    """
    # image_resized = cv2.resize(image, (960, 540))
    cv2.imshow('frame', image)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()


def get_crop_by_coords(image, coords):
    (x, y, w, h) = coords
    crop = image[int(y): int(y + h), int(x): int(x + w)]
    return crop


def visualize_bbox(image, point, w, h):

    # Visualizing with the bounding box over the frame
    left = point[1]
    top = point[0]
    right = left + w
    bottom = top + h
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.imshow("Frame", image)
    cv2.waitKey(40)
    cv2.destroyAllWindows()


def read_bgr_image(path_to_image):
    """
    Reads BGR image.
    :param path_to_image: Path to image to read.
    :return: BGR image (np.ndarray)
    """
    image_bgr = cv2.imread(path_to_image)
    # image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_bgr

