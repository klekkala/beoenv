import torch

model = torch.load('./see/model.pt', map_location="cpu")

print(type(model))

