# Checked
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog

class Dataset:
    def __init__(self, name: str, path: str, metadata: dict, json_annot_file: str) -> None:
        self.name = name
        self.path = path
        self.metadata = metadata
        self.json_annot_file = json_annot_file


    def register(self) -> None:
        """Registers dataset in DatasetCatalog"""
        register_coco_instances(self.name, self.metadata, self.json_annot_file, self.path)


    def is_registered(self) -> bool:
        return True if self.name in DatasetCatalog.list() else False

