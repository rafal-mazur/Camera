from detectron2.config import get_cfg
import const as c
def my_cfg():
    cfg = get_cfg()
    cfg.INPUT.MIN_SIZE_TRAIN = (640,)
    cfg.DATASETS.TRAIN = (c.TRAIN_DATASET_NAME)
    cfg.DATASETS.TEST = (c.TEST_DATASET_NAME)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.MODEL.DEVICE = 'cuda'
    cfg.SOLVER.MAX_ITER = 1000
    cfg.SOLVER.REFERENCE_WORLD_SIZE = 1
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.CHECKPOINT_PERIOD = 20
    cfg.TEST.DETECTIONS_PER_IMAGE = 5
    cfg.SOLVER.STEPS = (100,)
    return cfg
    