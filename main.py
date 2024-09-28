from detectron2.engine import DefaultTrainer

from model.utils import *
import model.const as c
import pickle
import os

def main():
    # Register train dataset
    register_dataset(c.TRAIN_DATASET_NAME,
                     c.TRAIN_DATASET_PATH,
                     c.TRAIN_JSON_ANNOT_FILE)

    # Register test dataset
    register_dataset(c.TEST_DATASET_NAME,
                     c.TEST_DATASET_PATH,
                     c.TEST_JSON_ANNOT_FILE)

    # load configs
    cfg = model_config()

    # save configs
    with open(c.CFG_SAVE_PATH, 'wb') as f:
        pickle.dump(cfg, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(False)

    trainer.train()

if __name__ == '__main__':
    main()