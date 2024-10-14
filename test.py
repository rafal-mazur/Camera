import torch

with open('E:\\Programowanie\\Camera-main\\output\\LP_eval\\instances_predictions.pth', 'rb') as f:
    a = torch.load(f)
    print(a)