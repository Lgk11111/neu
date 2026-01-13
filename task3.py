import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import warnings
import time
from collections import Counter

warnings.filterwarnings('ignore')


# ==================== 1. 终极配置参数 ====================
class Config:
    # 路径配置
    train_dir = './train'
    test_dir = './test'
    submission_file = './submission.csv'

    # 数据参数（关键：使用更大的尺寸）
    img_size = 96  # 更大的尺寸保留更多面部细节
    num_classes = 6
    emotion_map = {0: 'Angry', 1: 'Fear', 2: 'Happy', 3: 'Sad', 4: 'Surprise', 5: 'Neutral'}

    # 训练参数（优化）
    batch_size = 32  # 合适的批次大小
    epochs = 40  # 足够的训练轮次
    learning_rate = 0.0005  # 合适的学习率
    weight_decay = 1e-4  # 正则化

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 集成学习配置
    use_ensemble = True  # 使用集成学习
    n_models = 3  # 训练3个不同的模型
    model_names = ['efficientnet_b0', 'resnet34', 'resnet50']  # 不同的模型架构

    # 数据增强强度
    augment_strength = 'strong'  # 强数据增强

    # 交叉验证
    use_cv = True
    n_folds = 5

    # 模型保存
    model_save_path = './models/'
    os.makedirs(model_save_path, exist_ok=True)


# ==================== 2. 高级数据集类（带强数据增强） ====================
class AdvancedEmotionDataset(Dataset):
    """高级数据集类，带多种数据增强"""

    def __init__(self, data_dir=None, image_paths=None, labels=None, is_train=True):
        self.is_train = is_train

        if is_train and data_dir:
            self.image_paths = []
            self.labels = []

            # 加载所有图像路径
            for emotion_idx, emotion_name in Config.emotion_map.items():
                emotion_dir = os.path.join(data_dir, emotion_name)
                if os.path.exists(emotion_dir):
                    img_files = [f for f in os.listdir(emotion_dir)
                                 if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

                    for img_file in img_files:
                        self.image_paths.append(os.path.join(emotion_dir, img_file))
                        self.labels.append(emotion_idx)

            print(f"加载了 {len(self.image_paths)} 张训练图像")
            self.show_class_distribution()
        else:
            self.image_paths = image_paths if image_paths else []
            self.labels = labels if labels is not None else [0] * len(self.image_paths)

    def show_class_distribution(self):
        """显示类别分布"""
        counter = Counter(self.labels)
        print("\n类别分布统计:")
        total = len(self.labels)
        for emotion_id, emotion_name in Config.emotion_map.items():
            count = counter.get(emotion_id, 0)
            if count > 0:
                percentage = count / total * 100
                print(f"  {emotion_name}: {count} ({percentage:.1f}%)")

        # 检查类别平衡性
        if total > 0:
            max_count = max(counter.values())
            min_count = min(counter.values())
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
            if imbalance_ratio > 2:
                print(f"\n警告: 数据集不平衡 (最大/最小 = {imbalance_ratio:.1f})")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img_path = self.image_paths[idx]

            # 读取图像
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"无法读取图像: {img_path}")

            # 转换为RGB
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            img = Image.fromarray(img)

            # 应用变换
            if self.is_train:
                img = self.strong_train_transform(img)
            else:
                img = self.val_transform(img)

            return img, self.labels[idx], os.path.basename(img_path)

        except Exception as e:
            print(f"处理图像时出错: {e}")
            dummy = torch.zeros(3, Config.img_size, Config.img_size)
            return dummy, 0, "error.jpg"

    @property
    def strong_train_transform(self):
        """强数据增强变换"""
        if Config.augment_strength == 'strong':
            return transforms.Compose([
                transforms.Resize((Config.img_size + 20, Config.img_size + 20)),  # 先放大
                transforms.RandomCrop((Config.img_size, Config.img_size)),  # 随机裁剪
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
                transforms.RandomApply([transforms.RandomPerspective(distortion_scale=0.2)], p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.3, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
            ])
        else:
            return transforms.Compose([
                transforms.Resize((Config.img_size, Config.img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

    @property
    def val_transform(self):
        """验证/测试变换"""
        return transforms.Compose([
            transforms.Resize((Config.img_size, Config.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


# ==================== 3. 高级模型架构（带注意力机制） ====================
class SEBlock(nn.Module):
    """压缩-激发注意力模块"""

    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class AttentionEmotionModel(nn.Module):
    """带注意力的情感分类模型"""

    def __init__(self, base_model_name='resnet34', num_classes=6, pretrained=True):
        super(AttentionEmotionModel, self).__init__()

        # 加载预训练模型
        if base_model_name == 'resnet34':
            backbone = models.resnet34(pretrained=pretrained)
            in_features = backbone.fc.in_features
            # 移除最后的全连接层
            self.features = nn.Sequential(*list(backbone.children())[:-2])
            # 在特定层后添加注意力
            self.se1 = SEBlock(64)
            self.se2 = SEBlock(128)
            self.se3 = SEBlock(256)
            self.se4 = SEBlock(512)

        elif base_model_name == 'resnet50':
            backbone = models.resnet50(pretrained=pretrained)
            in_features = backbone.fc.in_features
            self.features = nn.Sequential(*list(backbone.children())[:-2])
            self.se1 = SEBlock(64)
            self.se2 = SEBlock(128)
            self.se3 = SEBlock(256)
            self.se4 = SEBlock(512)

        elif base_model_name == 'efficientnet_b0':
            backbone = models.efficientnet_b0(pretrained=pretrained)
            in_features = backbone.classifier[1].in_features
            self.features = backbone.features

        else:
            raise ValueError(f"不支持的模型: {base_model_name}")

        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化分类头的权重"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)

        # 如果有注意力模块，应用它们
        if hasattr(self, 'se4'):
            # 对ResNet的特征图应用注意力
            if x.size(1) == 512:
                x = self.se4(x)
            elif x.size(1) == 256:
                x = self.se3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# ==================== 4. 高级训练策略 ====================
def train_with_cosine_annealing(model, train_loader, val_loader, num_epochs=40, model_name='model'):
    """使用Cosine退火训练"""
    print(f"\n训练 {model_name}...")

    # 损失函数（带类别权重）
    criterion = nn.CrossEntropyLoss()

    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.learning_rate,
        weight_decay=Config.weight_decay
    )

    # Cosine退火学习率调度
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,  # 初始周期
        T_mult=2,  # 周期倍增因子
        eta_min=1e-6  # 最小学习率
    )

    # 训练记录
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 40)

        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (inputs, targets, _) in enumerate(train_loader):
            inputs, targets = inputs.to(Config.device), targets.to(Config.device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

            if (batch_idx + 1) % 30 == 0:
                avg_loss = train_loss / (batch_idx + 1)
                acc = 100. * train_correct / train_total
                print(f'  Batch {batch_idx + 1}/{len(train_loader)} | '
                      f'Loss: {avg_loss:.4f} | Acc: {acc:.2f}%')

        # 学习率调度
        scheduler.step()

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, targets, _ in val_loader:
                inputs, targets = inputs.to(Config.device), targets.to(Config.device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        # 计算指标
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total

        # 保存历史
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)

        print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'model_name': model_name,
            }, os.path.join(Config.model_save_path, f'best_{model_name}.pth'))
            print(f'✓ 保存最佳{model_name} | 验证准确率: {val_acc:.2f}%')

    print(f'\n{model_name} 最佳验证准确率: {best_acc:.2f}%')
    return best_acc, history


# ==================== 5. 交叉验证训练 ====================
def cross_validation_training():
    """交叉验证训练"""
    print("=" * 70)
    print("开始交叉验证训练")
    print("=" * 70)

    # 加载完整数据集
    dataset = AdvancedEmotionDataset(data_dir=Config.train_dir, is_train=True)

    # 准备数据
    X = np.arange(len(dataset))
    y = dataset.labels

    # 分层K折交叉验证
    skf = StratifiedKFold(n_splits=Config.n_folds, shuffle=True, random_state=42)

    fold_accuracies = []
    all_histories = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n{'=' * 60}")
        print(f"Fold {fold + 1}/{Config.n_folds}")
        print(f"{'=' * 60}")

        # 创建数据子集
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)

        # 创建数据加载器
        train_loader = DataLoader(
            train_subset, batch_size=Config.batch_size,
            shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_subset, batch_size=Config.batch_size * 2,
            shuffle=False, num_workers=0
        )

        # 创建模型
        model = AttentionEmotionModel('resnet34', Config.num_classes, pretrained=True)
        model = model.to(Config.device)

        # 训练
        best_acc, history = train_with_cosine_annealing(
            model, train_loader, val_loader,
            num_epochs=Config.epochs,
            model_name=f'fold{fold + 1}_resnet34'
        )

        fold_accuracies.append(best_acc)
        all_histories.append(history)

        print(f"\nFold {fold + 1} 完成 | 最佳准确率: {best_acc:.2f}%")

    # 打印交叉验证结果
    print(f"\n{'=' * 70}")
    print("交叉验证结果")
    print(f"{'=' * 70}")
    for i, acc in enumerate(fold_accuracies):
        print(f"Fold {i + 1}: {acc:.2f}%")

    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    print(f"\n平均准确率: {mean_acc:.2f}% ± {std_acc:.2f}%")

    return mean_acc


# ==================== 6. 集成学习训练 ====================
def ensemble_training():
    """集成学习训练多个模型"""
    print("=" * 70)
    print("开始集成学习训练")
    print("=" * 70)

    # 加载完整数据集
    dataset = AdvancedEmotionDataset(data_dir=Config.train_dir, is_train=True)

    # 划分训练集和验证集
    from sklearn.model_selection import train_test_split
    indices = list(range(len(dataset)))

    train_idx, val_idx = train_test_split(
        indices, test_size=0.2,
        stratify=dataset.labels,
        random_state=42
    )

    # 创建数据加载器
    train_subset = torch.utils.data.Subset(dataset, train_idx)
    val_subset = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(
        train_subset, batch_size=Config.batch_size,
        shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_subset, batch_size=Config.batch_size * 2,
        shuffle=False, num_workers=0
    )

    # 训练多个不同的模型
    model_accuracies = []

    for i, model_name in enumerate(Config.model_names[:Config.n_models]):
        print(f"\n训练模型 {i + 1}/{Config.n_models}: {model_name}")
        print("-" * 50)

        try:
            # 创建模型
            model = AttentionEmotionModel(model_name, Config.num_classes, pretrained=True)
            model = model.to(Config.device)

            # 训练
            best_acc, history = train_with_cosine_annealing(
                model, train_loader, val_loader,
                num_epochs=Config.epochs,
                model_name=f'ensemble_{model_name}'
            )

            model_accuracies.append((model_name, best_acc))
            print(f"模型 {model_name} 完成 | 准确率: {best_acc:.2f}%")

        except Exception as e:
            print(f"训练模型 {model_name} 时出错: {e}")
            # 如果某个模型失败，使用备选模型
            if model_name != 'resnet34':
                print(f"使用 resnet34 作为替代")
                model = AttentionEmotionModel('resnet34', Config.num_classes, pretrained=True)
                model = model.to(Config.device)

                best_acc, history = train_with_cosine_annealing(
                    model, train_loader, val_loader,
                    num_epochs=Config.epochs,
                    model_name=f'ensemble_resnet34_alt{i}'
                )

                model_accuracies.append(('resnet34', best_acc))

    # 打印集成学习结果
    print(f"\n{'=' * 70}")
    print("集成学习结果")
    print(f"{'=' * 70}")
    for model_name, acc in model_accuracies:
        print(f"{model_name}: {acc:.2f}%")

    avg_acc = np.mean([acc for _, acc in model_accuracies])
    print(f"\n平均准确率: {avg_acc:.2f}%")

    return avg_acc


# ==================== 7. 集成预测 ====================
def ensemble_predict():
    """使用集成学习进行预测"""
    print("\n" + "=" * 60)
    print("集成学习预测")
    print("=" * 60)

    # 检查测试目录
    if not os.path.exists(Config.test_dir):
        print(f"测试目录不存在: {Config.test_dir}")
        return None

    # 收集测试图像
    test_images = []
    test_ids = []

    for img_name in sorted(os.listdir(Config.test_dir)):
        if img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
            test_images.append(os.path.join(Config.test_dir, img_name))
            test_ids.append(img_name)

    if not test_images:
        print("未找到测试图像")
        return None

    print(f"找到 {len(test_images)} 张测试图像")

    # 创建测试数据集
    test_dataset = AdvancedEmotionDataset(
        image_paths=test_images,
        labels=None,
        is_train=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.batch_size * 4,
        shuffle=False,
        num_workers=0
    )

    # 加载所有训练好的模型
    model_files = []
    for f in os.listdir(Config.model_save_path):
        if f.startswith('best_') and f.endswith('.pth'):
            model_files.append(f)

    if not model_files:
        print("未找到训练好的模型")
        return None

    print(f"找到 {len(model_files)} 个模型用于集成")

    # 收集所有模型的预测
    all_predictions = []

    for model_file in model_files:
        print(f"\n加载模型: {model_file}")

        checkpoint = torch.load(
            os.path.join(Config.model_save_path, model_file),
            map_location=Config.device
        )

        # 获取模型名称
        model_name = checkpoint.get('model_name', 'resnet34')

        # 创建模型
        try:
            model = AttentionEmotionModel(model_name, Config.num_classes, pretrained=False)
        except:
            # 如果创建失败，使用默认模型
            model = AttentionEmotionModel('resnet34', Config.num_classes, pretrained=False)

        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(Config.device)
        model.eval()

        # 进行预测
        predictions = []
        with torch.no_grad():
            for inputs, _, _ in test_loader:
                inputs = inputs.to(Config.device)
                outputs = model(inputs)

                # 使用softmax获取概率
                probs = torch.softmax(outputs, dim=1)
                _, preds = probs.max(1)
                predictions.extend(preds.cpu().numpy())

        all_predictions.append(predictions)
        print(f"  完成预测")

    # 集成预测（投票）
    print("\n进行集成投票...")
    all_predictions = np.array(all_predictions)
    final_predictions = []

    for i in range(len(test_ids)):
        votes = all_predictions[:, i]
        # 多数投票
        final_predictions.append(np.bincount(votes).argmax())

    # 创建提交文件
    submission_df = pd.DataFrame({
        'ID': test_ids,
        'Emotion': final_predictions
    })

    # 排序并保存
    submission_df = submission_df.sort_values('ID').reset_index(drop=True)
    submission_df.to_csv(Config.submission_file, index=False)

    print(f"\n✓ 提交文件已保存: {Config.submission_file}")
    print(f"  总预测数: {len(submission_df)}")

    # 显示预测分布
    print("\n预测分布:")
    emotion_counts = submission_df['Emotion'].value_counts().sort_index()
    for emotion_id, emotion_name in Config.emotion_map.items():
        count = emotion_counts.get(emotion_id, 0)
        percentage = count / len(submission_df) * 100
        print(f"  {emotion_name}: {count} ({percentage:.1f}%)")

    return submission_df


# ==================== 8. 主函数 ====================
def main():
    """主函数"""
    print("=" * 80)
    print("人脸情感分类 - 终极高准确率版本")
    print(f"目标: 85%+ 准确率")
    print(f"设备: {Config.device}")
    print(f"图像尺寸: {Config.img_size}x{Config.img_size}")
    print("=" * 80)

    # 检查数据
    if not os.path.exists(Config.train_dir):
        print(f"\n错误: 训练目录 '{Config.train_dir}' 不存在!")
        print("请确保数据按以下结构组织:")
        print("  ./train/Angry/      # 包含愤怒表情图像")
        print("  ./train/Fear/       # 包含恐惧表情图像")
        print("  ./train/Happy/      # 包含快乐表情图像")
        print("  ./train/Sad/        # 包含悲伤表情图像")
        print("  ./train/Surprise/   # 包含惊讶表情图像")
        print("  ./train/Neutral/    # 包含中性表情图像")
        return

    start_time = time.time()

    # 自动训练流程
    print("\n" + "=" * 80)
    print("开始自动训练流程")
    print("=" * 80)

    # 选择训练模式
    if Config.use_ensemble:
        print("\n使用集成学习模式 (训练多个模型)")
        print(f"将训练以下模型: {', '.join(Config.model_names[:Config.n_models])}")

        # 训练集成模型
        avg_acc = ensemble_training()

        print(f"\n集成学习平均准确率: {avg_acc:.2f}%")

        if avg_acc < 75:
            print("\n⚠️  准确率还有提升空间，建议:")
            print("  1. 增加训练轮次 (修改 Config.epochs = 50)")
            print("  2. 使用更大的图像尺寸 (修改 Config.img_size = 112)")
            print("  3. 添加更多数据增强")
        elif avg_acc < 85:
            print("\n✅ 准确率良好，可以尝试:")
            print("  1. 使用更多模型进行集成")
            print("  2. 尝试更复杂的模型架构")
        else:
            print("\n🎉 准确率优秀! 已达到目标")

    else:
        print("\n使用交叉验证模式")
        mean_acc = cross_validation_training()
        print(f"\n交叉验证平均准确率: {mean_acc:.2f}%")

    # 预测测试集
    if os.path.exists(Config.test_dir):
        print("\n" + "=" * 80)
        print("开始预测测试集")
        print("=" * 80)

        submission = ensemble_predict()

        if submission is not None:
            print("\n" + "=" * 80)
            print("提交准备就绪!")
            print("=" * 80)
            print(f"文件: {Config.submission_file}")
            print("格式: CSV文件，包含两列: ID, Emotion")
            print("\n将此文件上传至评测平台")
    else:
        print(f"\n测试目录不存在: {Config.test_dir}")
        print("模型已保存，可在有测试数据时进行预测")

    # 时间统计
    total_time = time.time() - start_time
    hours = total_time // 3600
    minutes = (total_time % 3600) // 60
    seconds = total_time % 60

    print(f"\n总耗时: {hours:.0f}小时 {minutes:.0f}分钟 {seconds:.0f}秒")
    print("\n" + "=" * 80)
    print("程序完成!")
    print("=" * 80)


# ==================== 9. 运行程序 ====================
if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # 运行主程序
    try:
        main()
    except Exception as e:
        print(f"\n程序出错: {e}")
        import traceback

        traceback.print_exc()

        # 提供简化的备选方案
        print("\n" + "=" * 80)
        print("如果上述方案有问题，请尝试以下简化版本:")
        print("1. 将 Config.img_size 改为 64")
        print("2. 将 Config.model_names 改为 ['resnet18']")
        print("3. 将 Config.epochs 改为 30")

        print("=" * 80)
