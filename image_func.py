# Functions to use to handle image loading and processing
import cv2 as cv
import numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib import patches as patch


def image_generator(data_root: str,
                    bOnline: bool = False,
                    random_mode: bool = False,
                    n_imgs: int = 1) -> tuple[str, np.ndarray]:

    if bOnline:
        # Use camera
        pass

    else:
        # Get all images' names in directory
        images_names = os.listdir(data_root)

        # Choose n_imgs random images if necessary
        if random_mode:
            images_names = np.random.choice(images_names, size=n_imgs, replace=False)

        # Iterate through images' names and return image
        for img_name in images_names:
            # Remove extension
            yield img_name.removesuffix('.jpg'), cv.imread(os.path.join(data_root, img_name))


def read_bbox(bboxes_root: str, bbox_name: str, image_dim=(640,640)) -> patch.Rectangle:
    """Reads bbox for given image"""
    with open(os.path.join(bboxes_root, bbox_name + '.txt')) as bbox_data:
        tup = bbox_data.readline().split()
    bbox_data.close()
    x_centre = int(float(tup[1]) * image_dim[0])
    y_centre = int(float(tup[2]) * image_dim[1])
    width = int(float(tup[3]) * image_dim[0])
    height = int(float(tup[4]) * image_dim[1])


    return patch.Rectangle(xy=(x_centre - width // 2, y_centre - height // 2),
                           width=width,
                           height=height,
                           edgecolor='g',
                           linewidth=1,
                           facecolor='none')


def draw_samples(images_path: str,
                 labels_path: str,
                 n_rows: int = 3,
                 n_cols: int = 4,
                 bbox_show: bool = True) -> None:
    """
    Draw samples from dataset
    Number of images is n_rows * n_cols

    Parameters
    ----------
    images_path str:
        path to images folder

    labels_path str:
        path to labels folder

    n_rows int:
        Number of rows of images on plot

    n_cols int:
        Number of columns of images on plot

    bbox_show bool:
        Show with or without bounding boxes

    Returns
    -------
    None
    """

    # TODO: jak znajdywać liczby całk. żeby ich stosunek był jak najbliższy podanemu?
    # TODO: chcemy zamienić n_rows i n_cols na jedno n_samples
    # Create a figure
    fig, tmp_axs = plt.subplots(figsize=(12,6), nrows=n_rows, ncols=n_cols)

    # Flatten axs list
    axs = [i for j in tmp_axs for i in j]

    del tmp_axs

    # Iterate through axs
    for ax in axs:
        image_name, image = next(image_generator(images_path, random_mode= True, n_imgs=n_cols*n_cols))
        ax.imshow(image)
        if bbox_show:
            img_dim = (len(image), len(image[0]))
            bbox: patch.Rectangle = read_bbox(labels_path, image_name, img_dim)
            ax.add_patch(bbox)

    plt.show()