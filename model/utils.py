from detectron2.config import CfgNode, get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import  DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.model_zoo import model_zoo

import random
import cv2
import matplotlib.pyplot as plt

import model.const as c


def register_dataset(dataset_name: str,
                     images_path: str,
                     annotation_json_file_name: str,
                     metadata: dict = {},
                     replace_if_exists: bool = True) -> None:
    """Registers dataset in DatasetCatalog"""

    # Delete datasets that have overlapping name if needed
    if dataset_name not in DatasetCatalog.list():
        register_coco_instances(dataset_name, metadata, annotation_json_file_name, images_path)

    elif replace_if_exists:
        DatasetCatalog.remove(dataset_name)
        register_coco_instances(dataset_name, metadata, annotation_json_file_name, images_path)


def draw_samples(dataset_name, n_samples=1):
    """Draws samples from dataset registered in DatasetCatalog from detectron2"""
    dataset = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)

    for file in random.sample(dataset, n_samples):
        im = cv2.imread(file['file_name'], )
        v = Visualizer(im[:, :, ::-1], metadata, scale=0.8)
        v = v.draw_dataset_dict(file)

        fig, ax = plt.subplots()
        ax.imshow(v.get_image())
        fig.canvas.manager.full_screen_toggle()
        fig.show()

        fig.waitforbuttonpress()
        plt.close(fig)


def model_config(model_config_file: str = c.MODEL_CONFIG_FILE,
                 checkpoint_url: str = c.CHECKPOINT_URL,
                 train_dataset_name: str = c.TRAIN_DATASET_NAME,
                 test_dataset_name: str = c.TEST_DATASET_NAME,
                 num_classes: int = 1,
                 output_dir: str = c.OUTPUT_DIR,
                 device: str = c.DEVICE) -> CfgNode:
    """Sets model configuration"""

    if device != 'cpu' and device != 'cuda':
        raise ValueError('Given incorrect device (available devices: \'cpu\', \'cuda\')')

    if num_classes <= 0:
        raise ValueError("num_classes should be > 0")

    cfg = get_cfg()

    cfg.MODEL.DEVICE = device
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes

    cfg.merge_from_file(model_zoo.get_config_file(model_config_file))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(checkpoint_url)

    cfg.DATASETS.TRAIN = (train_dataset_name,)
    cfg.DATASETS.TEST = (test_dataset_name,)

    cfg.OUTPUT_DIR = output_dir
    return cfg
