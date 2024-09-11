import numpy as np
import os
import cv2 as cv
import decorators_custom as decor
@decor.repeat
@decor.measure_exe_time
def img_get(bOnline: bool = False, data_root: str = 'imgs') -> np.ndarray:
    """
    Generator, returns images from data_root folder one by one

    Parameters
    ----------
    bOnline: bool
        Is the camera online
    data_root: str
        Path to folder of images to return

    Returns
    -------
        Image as np.ndarray od dimensions (width x height x 3 colour channels)
    """
    if bOnline:
        # use camera
        pass
    else:
        # use saved images
        file_names = os.listdir(data_root)
        for file_name in file_names:
            yield cv.imread(os.path.join(data_root, file_name))
