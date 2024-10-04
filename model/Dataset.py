from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog

class Dataset:
    def __init__(self, name: str, path: str, metadata: dict, json_annot_file: str) -> None:
        self.name = name
        self.path = path
        self.metadata = metadata
        self.json_annot_file = json_annot_file


    def register(self, replace_if_exists: bool = True) -> None:
        """Registers dataset in DatasetCatalog"""
        if self.name not in DatasetCatalog.list():
            register_coco_instances(self.name, self.metadata, self.json_annot_file, self.path)
        elif replace_if_exists:
            DatasetCatalog.remove(self.name)
            register_coco_instances(self.name, self.metadata, self.json_annot_file, self.path)


    def is_registered(self) -> bool:
        return True if self.name in DatasetCatalog.list() else False

