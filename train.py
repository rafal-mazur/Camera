from detectron2.engine import DefaultTrainer

import defaults
from Dataset import Dataset
from my_cfg import my_cfg

import os, torch, yaml


def train():
    # register datasets
    train_dataset = Dataset(defaults.TRAIN_DATASET_NAME, defaults.TRAIN_DATASET_PATH, {}, defaults.TRAIN_JSON_ANNOT_FILE)
    valid_dataset = Dataset(defaults.VALID_DATASET_NAME, defaults.VALID_DATASET_PATH, {}, defaults.VALID_JSON_ANNOT_FILE)
    
    # get cfg
    cfg = my_cfg()
    
    # trainig
    trainer = DefaultTrainer(cfg)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer.resume_or_load(resume=False)
    trainer.train()
    
    # save cfg tp yaml
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')
    with open(defaults.CFG_SAVE_DIR+'\\config.yaml', 'w') as f:
        yaml.dump(cfg, f)
        f.close()
    


if __name__ == '__main__':
    train()