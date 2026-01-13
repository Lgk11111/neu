import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import warnings

warnings.filterwarnings('ignore')

# ===================== 1. 配置参数（关键修改：预测保存路径改为./image） =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_IMG_DIR = "train/image"
TRAIN_LABEL_DIR = "train/label"
TEST_IMG_DIR = "test/image"
PRED_SAVE_DIR = "./image"  # 核心修改：适配segmentation_to_csv.py的image目录
BATCH_SIZE = 4
EPOCHS = 50
LEARNING_RATE = 1e-4
IMG_SIZE = 256  # 输入网络的尺寸，最终输出还原为565x584


# ===================== 2. 自定义数据集（完全保留你的原代码） =====================
class FundusDataset(Dataset):
    def __init__(self, img_dir, label_dir=None, train_mode=True):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.train_mode = train_mode
        self.img_names = sorted(
            [f for f in os.listdir(img_dir) if f.lower().endswith('.jpg')],
            key=lambda x: int(os.path.splitext(x)[0])
        )
        # 原始尺寸记录（用于测试集还原）
        self.raw_sizes = {}
        for img_name in self.img_names:
            img_path = os.path.join(img_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                self.raw_sizes[img_name] = (img.shape[1], img.shape[0])  # (w, h)

        # 训练集数据增强
        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(5),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        # 测试集仅做归一化和缩放
        self.test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)

        # 读取灰度图
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = np.expand_dims(img, axis=-1)  # (H, W, 1)

        if self.train_mode and self.label_dir is not None:
            # 读取标签（血管=0，背景=255）
            label_path = os.path.join(self.label_dir, img_name)
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

            # 标签预处理：血管=1，背景=0（适配网络训练）
            label = (label == 0).astype(np.float32)  # 血管区域转为1

            # 数据增强（同步应用到图像和标签）
            seed = np.random.randint(2147483647)
            torch.manual_seed(seed)
            img = self.train_transform(img)
            torch.manual_seed(seed)
            label = self.train_transform(label)
            label = label.squeeze(0)  # (1, H, W) → (H, W)

            return img, label
        else:
            # 测试集仅返回图像、名称、原始尺寸（修复：返回tuple而非tensor）
            img = self.test_transform(img)
            w, h = self.raw_sizes[img_name]
            return img, img_name, (w, h)  # 关键修复：直接返回tuple


# ===================== 3. U-Net网络（完全保留你的原代码） =====================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.mpconv(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.conv(x))


class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1):
        super(UNet, self).__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


# ===================== 4. Dice损失函数（完全保留你的原代码） =====================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = pred.view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1 - dice


# ===================== 5. 训练函数（完全保留你的原代码） =====================
def train_model():
    # 加载训练集
    train_dataset = FundusDataset(TRAIN_IMG_DIR, TRAIN_LABEL_DIR, train_mode=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 初始化模型、损失、优化器
    model = UNet(n_channels=1, n_classes=1).to(DEVICE)
    criterion = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    print(f"===== 开始训练（{DEVICE}） =====")
    print(f"训练集数量：{len(train_dataset)}")
    print(f"总轮数：{EPOCHS} | 批次大小：{BATCH_SIZE}")

    # 训练循环
    best_loss = float('inf')
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            # 前向传播
            outputs = model(imgs)
            loss = criterion(outputs.squeeze(1), labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # 学习率调整
        scheduler.step()

        # 打印训练信息
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{EPOCHS}] | 平均损失：{avg_loss:.4f} | 学习率：{scheduler.get_last_lr()[0]:.6f}")

        # 保存最优模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "best_unet.pth")
            print(f"✅ 保存最优模型（损失：{best_loss:.4f}）")

    print("===== 训练完成 =====")
    return model


# ===================== 6. 测试集预测（仅修改保存路径，其余保留） =====================
def predict_test_set(model):
    # 加载测试集（关键修复：collate_fn避免tuple被转为tensor）
    def collate_fn(batch):
        imgs = torch.stack([item[0] for item in batch])
        img_names = [item[1] for item in batch]
        raw_sizes = [item[2] for item in batch]
        return imgs, img_names, raw_sizes

    test_dataset = FundusDataset(TEST_IMG_DIR, label_dir=None, train_mode=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # 加载最优权重
    model.load_state_dict(torch.load("best_unet.pth", map_location=DEVICE))
    model.eval()

    # 创建保存目录（改为./image，适配CSV脚本）
    os.makedirs(PRED_SAVE_DIR, exist_ok=True)
    # 清空旧文件（防止重复）
    for f in os.listdir(PRED_SAVE_DIR):
        if f.lower().endswith('.jpg'):
            os.remove(os.path.join(PRED_SAVE_DIR, f))

    print("\n===== 开始测试集预测 =====")
    with torch.no_grad():
        for imgs, img_names, raw_sizes in test_loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)

            # 转换为分割结果（血管=0，背景=255）
            pred = outputs.squeeze().cpu().numpy()
            pred = (pred > 0.5).astype(np.uint8)  # 预测为血管的区域=1
            pred = 255 - (pred * 255)  # 反转：血管=0，背景=255

            # 还原为原始尺寸（565x584）- 关键修复：取list中的tuple
            w, h = raw_sizes[0]
            pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)

            # 保存结果到./image目录（适配CSV脚本）
            img_name = img_names[0]
            save_path = os.path.join(PRED_SAVE_DIR, img_name)
            cv2.imwrite(save_path, pred)
            print(f"已保存：{save_path} | 原始尺寸：{w}x{h}")

    print("===== 测试集预测完成 =====")


# ===================== 7. 生成提交文件（仅优化调用逻辑，保留原脚本） =====================
def generate_submission():
    print("\n===== 生成提交文件 =====")
    try:
        # 检查image目录是否有文件
        if not os.listdir(PRED_SAVE_DIR):
            raise ValueError(f"{PRED_SAVE_DIR}目录为空！预测未生成分割图")

        # 调用外部CSV脚本（保留你的原逻辑，增加路径检查）
        os.system("python segmentation_to_csv.py")

        # 验证CSV是否生成
        if os.path.exists("submission.csv"):
            df = pd.read_csv("submission.csv")  # 临时导入pandas做验证
            print(f"✅ 提交文件生成成功：submission.csv")
            print(f"   - 数据行数：{len(df)}行（含标题共{len(df) + 1}行）")
            print(f"   - Id范围：{df['Id'].min()} ~ {df['Id'].max()}")
        else:
            raise FileNotFoundError("submission.csv未生成")

    except ImportError:
        print("⚠️  验证CSV需安装pandas：pip install pandas（不影响CSV生成）")
        print("✅ 提交文件已调用segmentation_to_csv.py生成")
    except Exception as e:
        print(f"⚠️  生成CSV失败：{str(e)}")
        print("💡 排查建议：")
        print(f"   1. 检查{PRED_SAVE_DIR}目录是否有1~20.jpg")
        print("   2. 确保segmentation_to_csv.py在当前目录")
        print("   3. 运行前关闭已打开的submission.csv")


# ===================== 8. 主函数（一键运行） =====================
if __name__ == "__main__":
    # 提前导入pandas（仅用于CSV验证，非必需）
    try:
        import pandas as pd
    except ImportError:
        print("⚠️  未安装pandas，仅影响CSV验证，不影响训练/预测")

    # 步骤1：训练模型
    model = train_model()

    # 步骤2：测试集预测（结果保存到./image）
    predict_test_set(model)

    # 步骤3：调用外部脚本生成CSV
    generate_submission()

    print("\n===== 全部流程完成 =====\n✅ 训练：best_unet.pth\n✅ 分割结果：./image\n✅ 提交文件：submission.csv")