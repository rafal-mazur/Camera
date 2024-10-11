from detectron2.config import get_cfg, CfgNode
from detectron2.model_zoo import model_zoo
import torch
import defaults

def my_cfg() -> CfgNode:
    """
    Creates a custom config

    Returns:
        CfgNode: Config for model
    """
    cfg = get_cfg()
    
    cfg.merge_from_file(model_zoo.get_config_file(defaults.MODEL_CONFIG_FILE))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(defaults.MODEL_CONFIG_FILE)
    
    if torch.cuda.is_available():
        print('--Using CUDA--')
        cfg.MODEL.DEVICE = 'cuda' 
    else: 
        print('--Using CPU--')
        cfg.MODEL.DEVICE = 'cpu'
    
    # cfg.INPUT.MIN_SIZE_TRAIN = (640,)
    # cfg.INPUT.MAX_SIZE_TRAIN = 640
    # cfg.INPUT.MIN_SIZE_TEST = 640
    # cfg.INPUT.MAX_SIZE_TEST = 640
    cfg.INPUT.RANDOM_FLIP = 'none'
    
    cfg.DATASETS.TRAIN = (defaults.TRAIN_DATASET_NAME,)
    cfg.DATASETS.TEST = (defaults.VALID_DATASET_NAME,)
    
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True
    
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.01
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.RETINANET.NUM_CLASSES = 1
    
    cfg.SOLVER.MAX_ITER = 15000
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.GAMMA = 0.5
    cfg.SOLVER.STEPS = () 
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"
    cfg.SOLVER.WARMUP_ITERS = int(0.1*cfg.SOLVER.MAX_ITER)
    cfg.SOLVER.CHECKPOINT_PERIOD = 5
    cfg.SOLVER.REFERENCE_WORLD_SIZE = 1 
    
    cfg.TEST.DETECTIONS_PER_IMAGE = 100 # max 100
    cfg.TEST.EVAL_PERIOD = 0
    
    cfg.VIS_PERIOD = 0
    cfg.OUTPUT_DIR = defaults.OUTPUT_DIR
    
    
    return cfg