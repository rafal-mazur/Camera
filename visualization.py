from detectron2.engine.defaults import build_model
import torch, detectron2, cv2

from matplotlib import pyplot as plt
from detectron2 import model_zoo

# DefaultTrainer is a class for training object model
from detectron2.engine import DefaultTrainer

# DefaultPredictor is class for inference
from detectron2.engine import DefaultPredictor

# detectron2 has its configuration format
from detectron2.config import get_cfg

# detectron2 has implemented Visualizer of object detection
from detectron2.utils.visualizer import Visualizer

# from DatasetCatalog, detectron2 gets dataset and from MetadatCatalog it gets metadata of the dataset
from detectron2.data import DatasetCatalog, MetadataCatalog

from utils import image_generator

# BoxMode supports bounding boxes in different format
from detectron2.structures import BoxMode

import torch
from my_cfg import my_cfg

def inference_frame(frame, predictor) -> None:

    pass
    
 
 
 
if __name__ == '__main__':
    pass