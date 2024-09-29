# In progress
from detectron2.config import get_cfg, CfgNode
from detectron2.model_zoo import model_zoo
from utils import *
import const as c



class Model:
    def __init__(self, train_dataset: Dataset, test_dataset: Dataset, cfg: CfgNode | None = None):

        # default config
        if cfg is None:
            self.cfg = get_cfg()
        else:
            self.cfg = cfg

        # train dataset
        self.train_dataset: Dataset = train_dataset
        self.train_dataset.register()

        # test dataset info
        self.test_dataset: Dataset = test_dataset
        self.train_dataset.register()

    def configure(self,
                  cfg: CfgNode | None = None,
                  **kwargs) -> None:
        """Sets model configuration"""
        if cfg is not None:
            self.cfg = cfg
            return

        # TODO: gdy jakis argument oprócz cfg jest none to return

        # if device != 'cpu' and device != 'cuda':
        #     raise ValueError('Given incorrect device (available devices: \'cpu\', \'cuda\')')
        #
        # if num_classes <= 0:
        #     raise ValueError("num_classes must be > 0")
        #
        # self.cfg.MODEL.DEVICE = device
        # self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
        #
        # self.cfg.merge_from_file(model_zoo.get_config_file(model_config_file))
        # self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(checkpoint_url)
        #
        # self.cfg.DATASETS.TRAIN = (self.train_dataset.name,)
        # self.cfg.DATASETS.TEST = (self.test_dataset.name,)
        #
        # self.cfg.OUTPUT_DIR = output_dir
