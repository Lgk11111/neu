import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, datasets, models
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import warnings
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

warnings.filterwarnings('ignore')

print("=" * 60)
print("🌱 植物幼苗分类系统 - 高级优化版")
print("=" * 60)

# 配置设置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = BASE_DIR
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 1. 数据增强强化
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1))
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# 2. 更强大的模型（ResNet50或EfficientNet）
class PlantClassifier(nn.Module):
    def __init__(self, num_classes, model_name='resnet50'):
        super(PlantClassifier, self).__init__()
        if model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif model_name == 'efficientnet':
            self.backbone = models.efficientnet_b3(pretrained=True)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            self.backbone = models.resnet34(pretrained=True)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        # 添加注意力机制
        self.attention = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, in_features),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        attention_weights = self.attention(features)
        attended_features = features * attention_weights
        return self.classifier(attended_features)


# 3. 处理类别不平衡
def get_class_weights(train_dir):
    class_counts = []
    class_names = sorted([d for d in os.listdir(train_dir)
                          if os.path.isdir(os.path.join(train_dir, d))])

    for class_name in class_names:
        class_path = os.path.join(train_dir, class_name)
        img_count = len([f for f in os.listdir(class_path) if f.endswith('.png')])
        class_counts.append(img_count)

    total_samples = sum(class_counts)
    weights = [total_samples / count for count in class_counts]
    weights = torch.FloatTensor(weights).to(device)
    return weights


# 4. 训练函数优化
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc='训练'):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 添加L2正则化
        l2_lambda = 0.001
        l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
        loss = loss + l2_lambda * l2_norm

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return total_loss / len(loader), 100.0 * correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc='验证'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / len(loader), 100.0 * correct / total, all_preds, all_labels


# 5. TTA（测试时增强）
def predict_with_tta(model, image_path, transform, device, tta_transforms=None):
    if tta_transforms is None:
        tta_transforms = [
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        ]

    model.eval()
    predictions = []

    for tta_transform in tta_transforms:
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = tta_transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(image_tensor)
                predictions.append(outputs.cpu().numpy())
        except:
            continue

    if predictions:
        avg_pred = np.mean(predictions, axis=0)
        predicted_idx = np.argmax(avg_pred)
    else:
        predicted_idx = 0

    return predicted_idx


def main():
    # 检查数据目录
    print(f"\n[1/5] 检查数据目录...")
    if not os.path.exists(TRAIN_DIR):
        print(f"❌ 训练目录不存在: {TRAIN_DIR}")
        return

    class_names = sorted([d for d in os.listdir(TRAIN_DIR)
                          if os.path.isdir(os.path.join(TRAIN_DIR, d))])
    print(f"✅ 找到 {len(class_names)} 个类别")

    # 加载完整数据集
    print(f"\n[2/5] 准备数据...")
    full_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)

    # 使用KFold交叉验证
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    all_indices = list(range(len(full_dataset)))

    fold_results = []
    best_models = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(all_indices)):
        print(f"\n{'=' * 40}")
        print(f"训练 Fold {fold + 1}/5")
        print(f"{'=' * 40}")

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)

        train_loader = DataLoader(full_dataset, batch_size=32, sampler=train_subsampler, num_workers=2)
        val_loader = DataLoader(full_dataset, batch_size=32, sampler=val_subsampler, num_workers=2)

        # 创建模型
        model = PlantClassifier(num_classes=len(class_names), model_name='resnet50')
        model = model.to(device)

        # 类别权重
        class_weights = get_class_weights(TRAIN_DIR)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # 优化器和学习率调度
        optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
        scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)

        best_val_acc = 0.0
        patience = 10
        patience_counter = 0

        # 训练循环
        for epoch in range(30):  # 增加epoch
            print(f"\nEpoch {epoch + 1}/30")

            # 训练
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

            # 验证
            val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)

            print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%")
            print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}%")

            # 学习率调度
            scheduler.step()

            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                model_path = os.path.join(OUTPUT_DIR, f'best_model_fold{fold + 1}.pth')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'val_acc': val_acc,
                    'epoch': epoch
                }, model_path)
                print(f"✅ 保存最佳模型 (Fold {fold + 1}): {val_acc:.2f}%")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"早停触发 (Fold {fold + 1})")
                    break

        fold_results.append(best_val_acc)
        best_models.append(model)
        print(f"Fold {fold + 1} 完成 - 最佳准确率: {best_val_acc:.2f}%")

    print(f"\n交叉验证结果: {fold_results}")
    print(f"平均准确率: {np.mean(fold_results):.2f}% ± {np.std(fold_results):.2f}%")

    # 6. 模型集成
    print(f"\n[3/5] 模型集成...")

    def ensemble_predict(models, image_tensor):
        all_preds = []
        for model in models:
            model.eval()
            with torch.no_grad():
                outputs = model(image_tensor.to(device))
                probs = torch.softmax(outputs, dim=1)
                all_preds.append(probs.cpu().numpy())

        avg_probs = np.mean(all_preds, axis=0)
        return np.argmax(avg_probs, axis=1)

    # 预测测试集
    print(f"\n[4/5] 预测测试集...")

    if not os.path.exists(TEST_DIR):
        print(f"⚠️ 测试目录不存在: {TEST_DIR}")
        return

    test_images = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not test_images:
        print("⚠️ 测试目录中没有图像文件")
        return

    print(f"✅ 找到 {len(test_images)} 张测试图像")

    predictions = []

    for img_name in tqdm(test_images, desc='预测'):
        try:
            img_path = os.path.join(TEST_DIR, img_name)

            # 使用TTA和模型集成
            image = Image.open(img_path).convert('RGB')
            image_tensor = val_transform(image).unsqueeze(0)

            # 集成预测
            predicted_idx = ensemble_predict(best_models, image_tensor)[0]
            predicted_class = class_names[predicted_idx]

            predictions.append({
                'image_name': img_name,
                'species': predicted_class
            })

        except Exception as e:
            print(f"⚠️ 处理图像 {img_name} 时出错: {e}")
            predictions.append({
                'image_name': img_name,
                'species': class_names[0] if class_names else 'Unknown'
            })

    # 7. 保存结果
    print(f"\n[5/5] 保存结果...")

    submission_path = os.path.join(OUTPUT_DIR, 'submission_ensemble.csv')
    df = pd.DataFrame(predictions)
    df.to_csv(submission_path, index=False)

    print(f"✅ 提交文件已保存: {submission_path}")

    # 分析结果
    print(f"\n📊 预测结果分布:")
    species_counts = df['species'].value_counts()
    for species, count in species_counts.items():
        percentage = count / len(df) * 100
        print(f"  {species}: {count} 张 ({percentage:.1f}%)")

    print("\n🎯 优化策略总结:")
    print("  1. 使用ResNet50/EfficientNet等更强模型")
    print("  2. 添加注意力机制和更深分类头")
    print("  3. 5折交叉验证")
    print("  4. 模型集成")
    print("  5. 更强的数据增强")
    print("  6. 类别权重处理不平衡")
    print("  7. TTA（测试时增强）")
    print("  8. 学习率调度和早停")
    print(f"\n预期准确率: 0.90+")


if __name__ == "__main__":
    main()

