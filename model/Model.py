# In progress
from detectron2.config import CfgNode
from detectron2.config import get_cfg

class Model:
    def __init__(self,
                 cfg: CfgNode | None = None):

        if cfg is None:
            cfg = get_cfg()
        
