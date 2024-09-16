# Functions to use to handle image loading and processing
import cv2 as cv
import numpy as np
import os

def get_all(data_root: str,
            show_labels: bool = True) -> tuple[str, tuple[int, float, float, float, float], np.ndarray] | np.ndarray:
    """
    Generator, iterate through it to get all images one by one

    Parameters
    ----------
    data_root str:
        path to folder (dataset) with folders of images and labels

    show_labels bool, optional:
        return image or image with labels

    Returns
    -------
    Tuple (img_name, img_label, image as np.ndarray) if show_labels = True
    or
    Image as np.ndarray
    """

    # Paths to labels' and images' folders
    IMGS_PATH = os.path.join(data_root, 'images')
    LABELS_PATH = os.path.join(data_root, 'labels')

    # Get all images' names in directory
    images_names = os.listdir(data_root)
    labels_names = [image_name.replace('.jpg', '.txt') for image_name in images_names]

    # Iterate through images' names
    for img_name, label_name in zip(images_names, labels_names):
        if show_labels:
            # Get label content
            with open(os.path.join(LABELS_PATH, label_name)) as label:
                tmp_tuple = label.readline().split()
            label.close()
            label_content = (int(tmp_tuple[0]),      # Class type
                             float(tmp_tuple[1]),    # x_centre
                             float(tmp_tuple[2]),    # y_centre
                             float(tmp_tuple[3]),    # width
                             float(tmp_tuple[4]))   # height
            # Return labelled image
            yield img_name, label_content, cv.imread(os.path.join(IMGS_PATH, img_name))
        else:
            # Return only image
            yield cv.imread(os.path.join(IMGS_PATH, img_name))


def random(data_root: str,
           n_imgs: int = 1,
           show_labels: bool = False) -> tuple[str, tuple[int, float, float, float, float], np.ndarray] | np.ndarray:
    """
    Generator, iterate through it to get all images one by one

    Parameters
    ----------
    data_root str:
        path to folder (dataset) with folders of images and labels

    n_imgs int:
        number of images to return

    show_labels bool, optional:
        return image or image with labels

    Returns
    -------
    if show_labels = True
    tuple[img_name, img_label as tuple[object class: int, x_centre: float, y_centre: float, width: float, height: float], image as np.ndarray]

    if show_labels == False
    image as np.ndarray


    """

    # Paths to labels' and images' folders
    IMGS_PATH = os.path.join(data_root, 'images')
    LABELS_PATH = os.path.join(data_root, 'labels')

    # Get all images' names in directory
    images_names = os.listdir(IMGS_PATH)

    # Create random subset of images_names
    random_images_names = np.random.choice(images_names, size=n_imgs, replace=False)
    random_labels_names = [name.rstrip('.jpg')+'.txt' for name in random_images_names]

    # Iterate through images' names
    for img_name, label_name in zip(random_images_names, random_labels_names):
        if show_labels:
            # Get labels content
            with open(os.path.join(LABELS_PATH, label_name)) as label:
                tmp_tuple = label.readline().split()
            label.close()
            x_centre = float(tmp_tuple[1])
            y_centre = float(tmp_tuple[2])
            width = float(tmp_tuple[3])
            height = float(tmp_tuple[4])
            # Return labelled image
            yield img_name, x_centre, y_centre, width, height, cv.imread(os.path.join(IMGS_PATH, img_name))
        else:
            # Return only image
            yield cv.imread(os.path.join(IMGS_PATH,img_name))
