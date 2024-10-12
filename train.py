from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator

import defaults
from Dataset import Dataset
from my_cfg import my_cfg

import os
import torch

# class MyTrainer(DefaultTrainer):
#     @classmethod
#     def build_evaluator(cls, cfg, dataset_name, output_folder=None):
#         if output_folder is None:
#             output_folder = os.path.join(cfg.OUTPUT_DIR,"inference")
#         return COCOEvaluator(dataset_name, cfg, True, output_folder)
    
#     def build_hooks(self):
#         hooks = super().build_hooks()
#         hooks.insert(-1,LossEvalHook(
#             cfg.TEST.EVAL_PERIOD,
#             self.model,
#             build_detection_test_loader(
#                 self.cfg,
#                 self.cfg.DATASETS.TEST[0],
#                 DatasetMapper(self.cfg,True)
#             )
#         ))
#         return hooks

def train():
    # register datasets
    train_dataset = Dataset(defaults.TRAIN_DATASET_NAME, defaults.TRAIN_DATASET_PATH, {}, defaults.TRAIN_JSON_ANNOT_FILE)
    train_dataset.register()
    
    test_dataset = Dataset(defaults.VALID_DATASET_NAME, defaults.VALID_DATASET_PATH, {}, defaults.VALID_JSON_ANNOT_FILE)
    test_dataset.register()
    
    # gets cfg
    cfg = my_cfg()
    
    # trainig
    trainer = DefaultTrainer(cfg)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer.resume_or_load(resume=False)
    trainer.train()
    
    # save model
    model_save_name = 'LP_detection_model.pth'
    path = os.path.join(defaults.CFG_SAVE_PATH, model_save_name)
    torch.save(trainer.state_dict(), path)


if __name__ == '__main__':
    train()