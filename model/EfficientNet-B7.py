import torch
from torchvision import models

model = models.efficientnet_b7(pretrained=True)

model_path = 'efficientnet_b7.pth'
torch.save(model.state_dict(), model_path)

print(f"模型已保存到 {model_path}")
