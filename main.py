import sys
import const as c
sys.path.append(c.MODEL_MODULE_PATH)

from model.Model import Model
from model.Dataset import Dataset
from my_cfg import my_cfg


def main() -> None:
    train_data: Dataset = Dataset(c.TRAIN_DATASET_NAME, c.TRAIN_DATASET_PATH, {}, c.TRAIN_JSON_ANNOT_FILE)
    test_data: Dataset = Dataset(c.TEST_DATASET_NAME, c.TEST_DATASET_PATH, {}, c.TEST_JSON_ANNOT_FILE)
    
    model = Model(train_data, test_data, c.CFG_SAVE_PATH, c.OUTPUT_DIR, my_cfg())
    model.save_cfg()
    
    model.train()

if __name__ == '__main__':
    main()