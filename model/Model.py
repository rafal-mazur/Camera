import sys
# In progress
from detectron2.config import get_cfg, CfgNode
from detectron2.model_zoo import model_zoo
from detectron2.engine import DefaultTrainer
from .Dataset import Dataset

import pickle
import os
import torch


class Model:
    def __init__(self, train_dataset: Dataset, test_dataset: Dataset, cfg_save_path: str, output_dir: str, cfg: CfgNode | None = None) -> None:
        """Custom model predictor

        Args:
            train_dataset (Dataset): dataset to train the model
            test_dataset (Dataset): dataset to test the model
            cfg_save_path (str): path to .pickle file that contains pickled model configuration
            cfg (CfgNode | None, optional): pass custom model configuration. Defaults to None.
        """

        # default config
        if cfg is None:
            self.cfg = get_cfg()
        else:
            self.cfg = cfg
        
        # train dataset
        self.train_dataset: Dataset = train_dataset
        self.train_dataset.register()
        self.cfg.DATASETS.TRAIN = (self.train_dataset.name,)

        # test dataset
        self.test_dataset: Dataset = test_dataset
        self.test_dataset.register()
        self.cfg.DATASETS.TEST = (self.test_dataset.name,)
        
        self.cfg_save_path = cfg_save_path
        self.cfg.OUTPUT_DIR = output_dir
        self.output_dir = output_dir
        


    def configure(self, cfg: CfgNode | None = None, **kwargs) -> None:
        """Sets model configration

        Args:
            cfg (CfgNode | None, optional): send complete config as CfgNone. Defaults to None (pass keyword arguments in that case).
            **kwargs:
                device: 'cuda' or 'cpu' device used by model\n
                num_classes: number of classes in given dataset\n
                model_config_file: path to model config file\n
                checkpoint: file path or url to checkpoint weights file\n
                output_dir: directory for model output\n
        """
        if cfg is not None:
            self.cfg = cfg
            return
        
        if kwargs.setdefault('device') is not None:
            if kwargs['device'] != 'cpu' and kwargs['device'] != 'cuda':
                raise ValueError('Incorrect device (\'cpu\' or \'cuda\')')
            else:
                self.cfg.MODEL.DEVICE = kwargs['device']
        
        if kwargs.setdefault('num_classes') is not None:
            if kwargs['num_classes'] <= 0:
                raise ValueError('num_classes should be > 0')
            else:
                self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = kwargs['num_classes']
        
        if kwargs.setdefault('model_config_file') is not None:
            self.cfg.merge_from_file(model_zoo.get_config_file(kwargs['model_config_file']))

        if kwargs.setdefault('checkpoint') is not None:
            if os.path.isfile(kwargs['checkpoint']):    
                self.cfg.MODEL.WEIGHTS = torch.load(kwargs['checkpoint'])
            else:
                self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(kwargs['checkpoint'])
        
        if kwargs.setdefault('output_dir') is not None:
            self.cfg.OUTPUT_DIR = kwargs['output_dir']
            
        
                     
    def save_cfg(self, config_path: str | None = None) -> None:
        """Saves current model configuration

        Args:
            config_path (str | None, optional): path for the config file to be saved. Defaults to path given in initializer.
        """
        if config_path is None:
            config_path = self.cfg_save_path
            
        try:
            if os.path.isfile(config_path):
                raise FileExistsError
        except FileExistsError as e:
            print('Config file already exists. Do you want to overwrite it? (y/n)')
            while True:
                r = input()
                if r[0] == 'y':
                    break
                elif r[0] == 'n':
                    exit(0)
                else:
                    print('Unknown command (type "y" or "n")')
        
        with open(config_path, 'wb') as f:
            pickle.dump(self.cfg, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()
        
        
    def train(self):
        os.makedirs(self.output_dir, exist_ok=True)
        trainer = DefaultTrainer(self.cfg)
        trainer.resume_or_load(False)
        
        trainer.train()
    
    
    def predict(self):
        pass

