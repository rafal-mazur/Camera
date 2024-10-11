import json

with open('E:\\Programowanie\\Camera-main\\LicencePlateDataset\\train_annotations.coco.json') as f:
    j = json.loads(f.read())
    annots = j['annotations']
    imgs = j['images']
    print(imgs[0])
    print(annots[0])
    for annot in annots:
        for img in imgs:
            if img['id'] == annot['image_id']:
                break
        else:
            print(1)