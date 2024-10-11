# Functions to use to handle image loading and processing
import cv2
import numpy as np
import os


def image_generator(data_root: str,
                    bOnline: bool = False,
                    n_imgs: int | None = None):
    """Generator, returns images from given directory

    Args:
        data_root (str): path to images
        bOnline (bool, optional): if true get image from camera, else use folder. Defaults to False.
        n_imgs (int | None, optional): number of returned images. Defaults to None.

    Yields:
        np.ndarray: image
    """
    
    if bOnline:
        # Use camera
        pass

    else:
        # Get all images' names in directory
        images_names = os.listdir(data_root)

        # Choose n_imgs random images if necessary
        if type(n_imgs) == int and n_imgs > 0:
            images_names = np.random.choice(images_names, size=n_imgs, replace=False)


        # Iterate through images' names and return image
        for img_name in images_names:
            yield cv2.imread(os.path.join(data_root, img_name))
