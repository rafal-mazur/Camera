from detectron2.utils.visualizer import Visualizer
from detectron2.data import  DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances

import random, cv2, os
import numpy as np
import matplotlib.pyplot as plt



def draw_samples(dataset_name: str, n_imgs: int = 1, window_size_inches: tuple[float, float]=(10,6), full_screen: bool = False) -> None:
    """Draws samples from dataset registered in DatasetCatalog from detectron2"""
    
    # dataset info
    dataset = DatasetCatalog.get(dataset_name)
    dataset_metadata = MetadataCatalog.get(dataset_name)
    print(dataset)
    print()
    print(dataset_metadata)
    
    fig, ax = plt.subplots()
    mngr = plt.get_current_fig_manager() 
    mngr.window.geometry("+0+0") # type: ignore
    mngr.set_window_title('draw_samples') # type: ignore
    
    for d in random.sample(dataset, n_imgs):
        im = cv2.imread(d['file_name'])
        v = Visualizer(im[:,:,::-1], metadata=dataset_metadata)
        v = v.draw_dataset_dict(d)
        print(d)
        
        ax.imshow(v.get_image())
        
        fig.set_size_inches(window_size_inches)
        if full_screen:
            fig.canvas.manager.full_screen_toggle() # type: ignore
        
        fig.show()
        fig.waitforbuttonpress()
        ax.clear()
        

def register_dataset(dataset_name: str,
                     images_path: str,
                     annotation_json_file_name: str,
                     metadata: dict = {}) -> None:
    """Registers dataset in DatasetCatalog"""
    register_coco_instances(dataset_name, metadata, annotation_json_file_name, images_path)


def dataset_registered(dataset_name: str) -> bool:
    return True if dataset_name in DatasetCatalog.list() else False


def image_generator(data_root: str, bOnline: bool = False, n_imgs: int | None = None):
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


def choose_best_model():
    pass




if __name__ == '__main__':
    from Dataset import Dataset
    dataset = Dataset('test', 'E:\\Programowanie\\Camera-main\\LicencePlateDataset\\test', {}, 'E:\\Programowanie\\Camera-main\\LicencePlateDataset\\test_annotations.coco.json')
    draw_samples(dataset.name, 20)
