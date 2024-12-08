import torch
from torchvision import models, transforms
from PIL import Image
import json

# 1. 加载预训练的 EfficientNet-B7 模型
model = models.efficientnet_b7(pretrained=True)
model.eval()  # 设置为评估模式

# 2. 读取本地图片
img_path = '../data/3.jpg'  # 替换为你的图片路径
img = Image.open(img_path)

# 3. 定义图片预处理流程
preprocess = transforms.Compose([
    transforms.Resize(600),  # EfficientNetB7 输入尺寸为 600x600
    transforms.CenterCrop(600),  # 裁剪图片到 600x600
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 图像标准化
])

# 4. 预处理图片
img_tensor = preprocess(img)
img_tensor = img_tensor.unsqueeze(0)  # 添加 batch 维度

# 5. 检查是否有可用的 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 6. 将输入图像传入模型进行预测
with torch.no_grad():  # 不需要计算梯度
    img_tensor = img_tensor.to(device)
    output = model(img_tensor)

# 7. 处理输出（获取预测的类别）
probabilities = torch.nn.functional.softmax(output[0], dim=0)

# 8. 读取本地的 ImageNet 类别标签文件
LABELS_PATH = './imagenet_class_index.json'  # 替换为你下载的文件路径
with open(LABELS_PATH, 'r') as f:
    class_idx = json.load(f)

# 9. 找到预测类别的索引
_, predicted_idx = torch.max(probabilities, 0)
predicted_label = class_idx[str(predicted_idx.item())][1]

# 10. 显示预测标签及其概率
print(f'Predicted label: {predicted_label}')
print(f'Prediction probability: {probabilities[predicted_idx].item():.4f}')
