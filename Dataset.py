# Checked
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog

class Dataset:
    def __init__(self, name: str, path: str, metadata: dict, json_annot_file: str) -> None:
        """Creates Dataset object and registers dataset into DatasetCatalog

        Args:
            name (str): name of dataset
            path (str): path to image directory
            metadata (dict): dataset metadata
            json_annot_file (str): path to dataset annotation file
        """
        self.name = name
        self.path = path
        self.metadata = metadata
        self.json_annot_file = json_annot_file
        register_coco_instances(self.name, self.metadata, self.json_annot_file, self.path)

    def is_registered(self) -> bool:
        return True if self.name in DatasetCatalog.list() else False

