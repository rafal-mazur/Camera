from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog

def register_dataset(dataset_name: str,
                     image_path: str,
                     annotation_json_file_name: str,
                     metadata: dict = {},
                     replace_if_exists: bool = True) -> None:

    # Names for the datasets

    # Delete datasets that have overlapping name if needed
    if dataset_name not in DatasetCatalog.list():
        register_coco_instances(dataset_name, metadata, annotation_json_file_name, image_path)

    elif replace_if_exists:
        DatasetCatalog.remove(dataset_name)
        register_coco_instances(dataset_name, metadata, annotation_json_file_name, image_path)


