import torch
from torchvision.models import resnet18, ResNet18_Weights
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

input_name = "data"
shape = 1, 3, 224, 224
trace = torch.jit.trace(model.eval(), torch.rand(shape).float())
torch.jit.save(trace, "resnet18.pth")
