from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultTrainer
from detectron2.modeling import build_model
from my_cfg import my_cfg
from Dataset import Dataset
import defaults

import os

def evaluate(cfg, trainer,  dataset_name, eval_output_dir):
    evaluator = COCOEvaluator(dataset_name, cfg, False, output_dir=eval_output_dir)
    val_loader = build_detection_test_loader(cfg, dataset_name) # type: ignore
    inference_on_dataset(trainer.model, val_loader, evaluator)


def evaluate_all_models(cfg, weights_dir, valid_dataset_name, eval_output_dir):
    models_list = [model_name for model_name in os.listdir(weights_dir) if model_name[-4:] == '.pth']
    for weights_file in models_list:
        cfg.MODEL.WEIGHTS = weights_file
        inference_on_dataset(build_model(cfg), build_detection_test_loader(cfg, valid_dataset_name), COCOEvaluator(valid_dataset_name, cfg, False, output_dir=eval_output_dir)) # type: ignore


def get_best_model():
    pass


if __name__ == '__main__':
    train_dataset = Dataset(defaults.TRAIN_DATASET_NAME, defaults.TRAIN_DATASET_PATH, {}, defaults.TRAIN_JSON_ANNOT_FILE)
    valid_dataset = Dataset(defaults.VALID_DATASET_NAME, defaults.VALID_DATASET_PATH, {}, defaults.VALID_JSON_ANNOT_FILE)
    cfg = my_cfg()
    cfg.MODEL.WEIGHTS = 'E:\\Programowanie\\Camera-main\\models_cfg\\model_best_for_now.pth'
    trainer = DefaultTrainer(cfg)
    evaluate_all_models(cfg, 'E:\\Programowanie\\Camera-main\\output\\LP_detection', valid_dataset.name, '.\\output\\LP_eval')