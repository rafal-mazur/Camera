import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import CfgNode
from detectron2.utils.visualizer import Visualizer

from my_cfg import my_cfg
from utils import image_generator
import cv2


def predict(img: np.ndarray, cfg: CfgNode | None = None):
    if cfg is None:
        cfg = my_cfg()
        cfg.MODEL.WEIGHTS = 'E:\\Programowanie\\Camera-main\\output\\LP_detection\\model_final.pth'
    
    
    predictor = DefaultPredictor(cfg)
    output = predictor(img)
    v = Visualizer(img[:,:,::-1])
    v = v.draw_instance_predictions(output['instances'].to('cpu'))
    processed_img = v.get_image()[:,:,::-1]
    cv2.imshow('X',processed_img)
    
    print(output['instances'])
    cv2.waitKey(0)
    
if __name__ == '__main__':
    for img in image_generator('E:\\Programowanie\\Camera-main\\LicencePlateDataset\\test', n_imgs=1):
        predict(img)