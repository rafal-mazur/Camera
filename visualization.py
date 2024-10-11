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

from image_func import image_generator

# BoxMode supports bounding boxes in different format
from detectron2.structures import BoxMode

import torch
from my_cfg import my_cfg

def inference_frame(frame, predictor) -> None:

    outputs = predictor(frame)

      # v = Visualizer(frame[:, :, ::-1],metadata=test_metadata, scale=0.8)
      # v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
      # img = v.get_image()
    for box in outputs["instances"].pred_boxes.to('cpu'):
        bbox = box.numpy()
        frame = cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 255), thickness=5, lineType=cv2.LINE_8)
        im = frame[:, :, ::-1]
        plt.imshow(im)
        plt.show()
    return frame

# register_dataset('my_dataset',
#                  'C:\\Projektkamerka7.09\\dataset\\train',
#                  'C:\\Projektkamerka7.09\\dataset\\train\\_annotations.coco.json')

def main():
    with open('E:\\Programowanie\\Camera-main\\model\\output\\model_0000539.pth', 'rb') as f:
        weights = torch.load(f, weights_only=True)
    
    cfg = my_cfg()
    cfg.MODEL.WEIGHTS = weights
    predictor = DefaultPredictor(cfg)
    frame = inference_frame(image_generator('E:\\Programowanie\\Camera-main\\output\\LP_detection\\model_final.pth', n_imgs=1), predictor)
    
 
 
 
if __name__ == '__main__':
    main()