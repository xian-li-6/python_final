import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# 设置是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义数据预处理
data_transform = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),  # 随机裁剪成224x224
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.ToTensor(),  # 转换为Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),  # 调整图像大小为256
        transforms.CenterCrop(224),  # 中心裁剪224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
}

# 数据加载函数
def load_data(data_dir, batch_size=32):
    # ImageNet 数据集路径（请根据实际情况修改路径）
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    # 加载训练集和验证集
    train_dataset = datasets.ImageFolder(train_dir, data_transform['train'])
    val_dataset = datasets.ImageFolder(val_dir, data_transform['val'])

    # 使用DataLoader加载数据
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader

# 加载预训练的 EfficientNet-B7 模型
model = models.efficientnet_b7(pretrained=True)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1000)  # ImageNet 有 1000 类

model = model.to(device)

# 设置优化器和损失函数
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Adam优化器，学习率为1e-4

# 设置学习率调度器
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# 训练和验证函数
def train_one_epoch(model, train_loader, optimizer, criterion):
    model.train()  # 设置为训练模式
    running_loss = 0.0
    correct = 0
    total = 0

    # 使用tqdm显示训练进度
    for inputs, labels in tqdm(train_loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)

        # 清空优化器梯度
        optimizer.zero_grad()

        # 前向传播
        outputs = model(inputs)

        # 计算损失
        loss = criterion(outputs, labels)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        # 计算损失和准确率
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # 计算平均损失和准确率
    avg_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy

def validate_one_epoch(model, val_loader, criterion):
    model.eval()  # 设置为评估模式
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # 在验证时不计算梯度
        for inputs, labels in tqdm(val_loader, desc="Validating"):
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播
            outputs = model(inputs)

            # 计算损失
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # 计算平均损失和准确率
    avg_loss = running_loss / len(val_loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy

# 训练和评估整个模型
def train_and_evaluate(model, data_dir, num_epochs=10, batch_size=32):
    train_loader, val_loader = load_data(data_dir, batch_size)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # 训练一轮
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")

        # 验证一轮
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion)
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")

        # 更新学习率
        scheduler.step()

        # 保存模型
        torch.save(model.state_dict(), f"efficientnet_b7_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    data_dir = 'path_to_imagenet_data'  # 修改为你存储ImageNet数据集的路径
    train_and_evaluate(model, data_dir, num_epochs=10, batch_size=32)
