import cv2
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.structures.instances import Instances
from detectron2.utils.visualizer import Visualizer


from my_cfg import my_cfg
from utils import image_generator


def predict_boxes(img: np.ndarray, predictor: DefaultPredictor):
    outputs =  predictor(img)
    boxes = []
    for box in outputs['instances'].pred_boxes.to('cpu'):
        bbox = box.numpy()
        bbox = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        boxes.append(bbox)

    return boxes

    
def get_rois(img: np.ndarray, predictor: DefaultPredictor) -> list[np.ndarray]:
    boxes = predict_boxes(img, predictor)
    rois = []
    for box in boxes:
        rois.append(img[box[1]: box[3], box[0]:box[2],:])
        
    return rois


if __name__ == '__main__':
    cfg = my_cfg()
    cfg.MODEL.WEIGHTS = 'E:\\Programowanie\\Camera-main\\models_cfg\\model_0041279.pth'
    p = DefaultPredictor(cfg)
    for img in image_generator('E:\\Programowanie\\Camera-main\\LicencePlateDataset\\test', n_imgs=10):
        for roi in get_rois(img, p):
            cv2.imshow('iamge', roi)
            cv2.waitKey(0)
            