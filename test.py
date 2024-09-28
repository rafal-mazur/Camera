import json

with open('C:\\Projektkamerka7.09\\dataset\\train\\_annotations.coco.json') as file:
    j = json.loads(file.read())
    file.close()

for i1 in j['annotations']:
    print(i1['bbox'])
