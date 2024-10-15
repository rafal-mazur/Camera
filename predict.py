import cv2
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.structures.instances import Instances
from detectron2.utils.visualizer import Visualizer
from detectron2.structures.boxes import BoxMode


from my_cfg import my_cfg
from utils import image_generator


def _predict_boxes(img: np.ndarray, predictor: DefaultPredictor):
    """Predicts boxes predictions in BoxMode.XYXY_ABS

    Args:
        img (np.ndarray): image to make predictions on
        predictor (DefaultPredictor): predictor for image

    Returns:
        List of bounding boxes in BoxMode.XYXY_ABS
    """
    outputs =  predictor(img)
    boxes = []
    for box in outputs['instances'].pred_boxes.to('cpu'):
        bbox = box.numpy()
        bbox = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        boxes.append(bbox)

    return boxes


def get_rois(img: np.ndarray, predictor: DefaultPredictor):
    boxes = _predict_boxes(img, predictor)
    for box in boxes:
        yield img[box[1]: box[3], box[0]:box[2],:]
        


if __name__ == '__main__':
    cfg = my_cfg()
    cfg.MODEL.WEIGHTS = 'E:\\Programowanie\\Camera-main\\output\\LP_detection\\model_0035779.pth'
    p = DefaultPredictor(cfg)
    
    for image in image_generator('E:\\Programowanie\\Camera-main\\LicencePlateDataset\\test', n_imgs=10):
        for roi in get_rois(image, p):
            cv2.imshow('image', roi)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        