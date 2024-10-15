from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultTrainer
from detectron2.modeling import build_model
from my_cfg import my_cfg
from Dataset import Dataset
import defaults

import os

def evaluate(cfg,  dataset_name, eval_output_dir):
    evaluator = COCOEvaluator(dataset_name, cfg, False, eval_output_dir)
    val_loader = build_detection_test_loader(cfg, dataset_name) # type: ignore
    trainer = DefaultTrainer(cfg)
    inference_on_dataset(trainer.model, val_loader, evaluator) # type: ignore


def evaluate_all_models(cfg, weights_dir, valid_dataset_name, eval_output_dir):
    models_list = [model_name for model_name in os.listdir(weights_dir) if model_name[-4:] == '.pth'][::-1]
    for weights_file in models_list:
        cfg.MODEL.WEIGHTS = weights_file
        inference_on_dataset(build_model(cfg), build_detection_test_loader(cfg, valid_dataset_name), COCOEvaluator(valid_dataset_name, cfg, False, output_dir=eval_output_dir)) # type: ignore


def get_best_model():
    pass


if __name__ == '__main__':
    train_dataset = Dataset(defaults.TRAIN_DATASET_NAME, defaults.TRAIN_DATASET_PATH, {}, defaults.TRAIN_JSON_ANNOT_FILE)
    valid_dataset = Dataset(defaults.TEST_DATASET_NAME, defaults.TEST_DATASET_PATH, {}, defaults.TEST_JSON_ANNOT_FILE)
    cfg = my_cfg()
    cfg.MODEL.WEIGHTS = 'E:\\Programowanie\\Camera-main\\output\\LP_detection\\model_0007339.pth'
    evaluate(cfg, valid_dataset.name, 'E:\\Programowanie\\Camera-main\\output\\LP_eval')