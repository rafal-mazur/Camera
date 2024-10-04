# File with constants
TRAIN_DATASET_NAME: str = 'train_data'
TRAIN_DATASET_PATH: str = 'E:\\Programowanie\\Camera-main\\LicencePlateDataset\\train'
TRAIN_JSON_ANNOT_FILE: str = 'E:\\Programowanie\\Camera-main\\LicencePlateDataset\\train_annotations.coco.json'

VALID_DATASET_NAME: str = 'valid_data'
VALID_DATASET_PATH: str = 'E:\\Programowanie\\Camera-main\\LicencePlateDataset\\valid'
VALID_JSON_ANNOT_FILE: str = 'E:\\Programowanie\\Camera-main\\LicencePlateDataset\\valid_annotations.coco.json'

TEST_DATASET_NAME: str = 'test_data'
TEST_DATASET_PATH: str = 'E:\\Programowanie\\Camera-main\\LicencePlateDataset\\test'
TEST_JSON_ANNOT_FILE: str = 'E:\\Programowanie\\Camera-main\\LicencePlateDataset\\test_annotations.coco.json'

DEVICE: str = 'cuda' # 'cuda' or 'cpu'

MODEL_CONFIG_FILE: str = 'COCO-Detection/retinanet_R_50_FPN_3x.yaml'
CHECKPOINT_URL: str = 'COCO-Detection/retinanet_R_50_FPN_3x.yaml'

OUTPUT_DIR: str = 'model\\output'
CFG_SAVE_PATH: str = 'model\\config\\LP_detection_cfg.pickle'

MODEL_MODULE_PATH: str = ".\\model"