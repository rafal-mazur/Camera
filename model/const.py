# File with constants
TRAIN_DATASET_NAME: str = 'train_data'
TRAIN_DATASET_PATH: str = 'C:\\Projektkamerka7.09\\dataset\\train_imgs'
TRAIN_JSON_ANNOT_FILE: str = 'C:\\Projektkamerka7.09\\dataset\\_train_annotations.coco.json'

VALID_DATASET_NAME: str = 'valid_data'
VALID_DATASET_PATH: str = 'C:\\Projektkamerka7.09\\dataset\\valid_imgs'
VALID_JSON_ANNOT_FILE: str = 'C:\\Projektkamerka7.09\\dataset\\_valid_annotations.coco.json'

TEST_DATASET_NAME: str = 'test_data'
TEST_DATASET_PATH: str = 'C:\\Projektkamerka7.09\\dataset\\test_imgs'
TEST_JSON_ANNOT_FILE: str = 'C:\\Projektkamerka7.09\\dataset\\_test_annotations.coco.json'

DEVICE: str = 'cpu' # 'cuda' or 'cpu'

MODEL_CONFIG_FILE: str = 'COCO-Detection/retinanet_R_50_FPN_3x.yaml'
CHECKPOINT_URL: str = 'COCO-Detection/retinanet_R_50_FPN_3x.yaml'

OUTPUT_DIR: str = '.\\output\\Licence_Plate_detection'

CFG_SAVE_PATH: str = '.\\LP_cfg.pickle'