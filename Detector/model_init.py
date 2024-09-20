from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.modeling import build_model
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg


def set_configs(device: str = 'cpu') -> None:
    cfg = get_cfg()
    cfg.MODEL.DEVICE = device

