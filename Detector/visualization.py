import torch, detectron2, cv2

from matplotlib import pyplot as plt
from detectron2 import model_zoo

# DefaultTrainer is a class for training object detector
from detectron2.engine import DefaultTrainer

# DefaultPredictor is class for inference
from detectron2.engine import DefaultPredictor

# detectron2 has its configuration format
from detectron2.config import get_cfg

# detectron2 has implemented Visualizer of object detection
from detectron2.utils.visualizer import Visualizer

# from DatasetCatalog, detectron2 gets dataset and from MetadatCatalog it gets metadata of the dataset
from detectron2.data import DatasetCatalog, MetadataCatalog

from image_func import image_generator

# BoxMode supports bounding boxes in different format
from detectron2.structures import BoxMode

from register_dataset import register_dataset

def inference_frame(dataset_name: str) -> None:
    if dataset_name in MetadataCatalog.list():
        dataset_metadata = MetadataCatalog.get(dataset_name)
    else:
        raise NameError
