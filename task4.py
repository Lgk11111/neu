# final_optimized.py
import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

print("=" * 60)
print("中央凹定位 - 最终优化版")
print("=" * 60)

# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 路径
base_path = r'C:\Users\Dell\Desktop\detection'
train_img_path = os.path.join(base_path, 'train')
test_img_path = os.path.join(base_path, 'test')
train_csv_path = os.path.join(base_path, 'fovea_localization_train_GT.csv')

# ==================== 最终超参数 ====================
IMG_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 100
LEARNING_RATE = 2e-5
GAUSSIAN_SIGMA = 20


# ==================== 1. 最终模型 ====================
class FinalUNet(nn.Module):
    def __init__(self):
        super().__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        # 编码器
        self.enc1 = conv_block(1, 32)
        self.enc2 = conv_block(32, 64)
        self.enc3 = conv_block(64, 128)
        self.enc4 = conv_block(128, 256)
        self.pool = nn.MaxPool2d(2)

        # 瓶颈
        self.bottleneck = conv_block(256, 512)

        # 解码器
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = conv_block(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = conv_block(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = conv_block(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = conv_block(64, 32)

        # 输出
        self.out = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out(d1)


# ==================== 2. 数据函数 ====================
def create_heatmap(size, center, sigma=GAUSSIAN_SIGMA):
    h, w = size
    x, y = center
    x = int(np.clip(x, 0, w - 1))
    y = int(np.clip(y, 0, h - 1))

    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    dist = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
    heatmap = np.exp(-dist ** 2 / (2 * sigma ** 2))

    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    return heatmap


class FinalDataset(Dataset):
    def __init__(self, img_paths, coords=None, train=True):
        self.img_paths = img_paths
        self.coords = coords
        self.train = train

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx], cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

        h, w = img.shape
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        # 简单归一化
        img = img.astype(np.float32) / 255.0
        img_tensor = torch.FloatTensor(img).unsqueeze(0)

        if self.coords is None:
            img_id = os.path.basename(self.img_paths[idx]).split('.')[0]
            return img_tensor, img_id, (w, h)

        x, y = self.coords[idx]
        scale_x = IMG_SIZE / w
        scale_y = IMG_SIZE / h
        x_scaled = x * scale_x
        y_scaled = y * scale_y

        heatmap = create_heatmap((IMG_SIZE, IMG_SIZE), (x_scaled, y_scaled))
        heatmap_tensor = torch.FloatTensor(heatmap).unsqueeze(0)

        # 简单数据增强
        if self.train and np.random.random() > 0.5:
            img_tensor = torch.flip(img_tensor, dims=[-1])
            heatmap_tensor = torch.flip(heatmap_tensor, dims=[-1])

        return img_tensor, heatmap_tensor


# ==================== 3. 最终训练 ====================
def final_train():
    print("\n" + "=" * 40)
    print("最终训练")
    print("=" * 40)

    # 加载数据
    df = pd.read_csv(train_csv_path)
    paths, coords = [], []

    for _, row in df.iterrows():
        img_id = int(row['data'])
        path = os.path.join(train_img_path, f"{img_id:04d}.jpg")
        if os.path.exists(path):
            paths.append(path)
            coords.append((float(row['Fovea_X']), float(row['Fovea_Y'])))

    print(f"加载 {len(paths)} 张图像")

    # 划分
    train_paths, val_paths, train_coords, val_coords = train_test_split(
        paths, coords, test_size=0.15, random_state=42
    )

    # 数据集
    train_set = FinalDataset(train_paths, train_coords, train=True)
    val_set = FinalDataset(val_paths, val_coords, train=False)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

    # 模型
    model = FinalUNet().to(device)

    # 优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    # 训练
    best_val = float('inf')
    train_losses, val_losses = [], []

    print(f"\n训练开始 ({EPOCHS} epochs)")

    for epoch in range(EPOCHS):
        # 训练
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{EPOCHS}')

        for imgs, heatmaps in pbar:
            imgs, heatmaps = imgs.to(device), heatmaps.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, heatmaps)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        avg_train = train_loss / len(train_loader)
        train_losses.append(avg_train)

        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, heatmaps in val_loader:
                imgs, heatmaps = imgs.to(device), heatmaps.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, heatmaps)
                val_loss += loss.item()

        avg_val = val_loss / len(val_loader)
        val_losses.append(avg_val)

        # 调整学习率
        scheduler.step(avg_val)

        # 保存最佳
        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), 'final_best_model.pth')
            print(f"✓ 保存最佳模型 (val: {best_val:.6f})")

        # 打印
        est_score = avg_val * 1200
        print(f"Epoch {epoch + 1}: train={avg_train:.6f}, val={avg_val:.6f}, est={est_score:.1f}")

        # 早停
        if avg_val < 0.008:  # 约10分
            print(f"达到优秀水平! 停止训练")
            break

    # 保存训练记录
    torch.save({
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val': best_val
    }, 'training_history.pth')

    # 绘图
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='训练')
    plt.plot(val_losses, label='验证')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()
    plt.title('训练过程')
    plt.grid(True, alpha=0.3)
    plt.savefig('final_training.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n训练完成!")
    print(f"最佳验证损失: {best_val:.6f}")
    print(f"预计分数: {best_val * 1200:.1f}")

    return model


# ==================== 4. 最终预测 ====================
def final_predict(model):
    print("\n" + "=" * 40)
    print("最终预测")
    print("=" * 40)

    # 检查测试目录
    if not os.path.exists(test_img_path):
        print(f"测试目录不存在: {test_img_path}")
        return None

    # 收集测试图像
    predictions = {}

    for img_id in range(81, 101):
        path = os.path.join(test_img_path, f"{img_id:04d}.jpg")

        if os.path.exists(path):
            # 读取
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                predictions[img_id] = (1446, 1056)  # 默认值
                continue

            h, w = img.shape
            img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            # 预处理
            img_norm = img_resized.astype(np.float32) / 255.0
            img_tensor = torch.FloatTensor(img_norm).unsqueeze(0).unsqueeze(0).to(device)

            # 预测
            model.eval()
            with torch.no_grad():
                heatmap = model(img_tensor).squeeze().cpu().numpy()

            # 改进的坐标提取
            def get_refined_coord(heatmap):
                # 找到峰值
                peak_idx = np.argmax(heatmap)
                y_peak, x_peak = np.unravel_index(peak_idx, heatmap.shape)

                # 区域加权平均
                threshold = heatmap.max() * 0.3
                mask = heatmap > threshold

                if np.sum(mask) > 5:
                    yy, xx = np.where(mask)
                    weights = heatmap[yy, xx]

                    # 加权平均
                    x_avg = np.sum(xx * weights) / np.sum(weights)
                    y_avg = np.sum(yy * weights) / np.sum(weights)
                else:
                    x_avg, y_avg = x_peak, y_peak

                return x_avg, y_avg

            # 获取坐标
            x_scaled, y_scaled = get_refined_coord(heatmap)

            # 缩放回原始尺寸
            x = x_scaled * (w / IMG_SIZE)
            y = y_scaled * (h / IMG_SIZE)

            predictions[img_id] = (float(x), float(y))
            print(f"图像 {img_id:04d}: X={x:8.1f}, Y={y:8.1f}")
        else:
            predictions[img_id] = (1446, 1056)
            print(f"图像 {img_id:04d}: 文件不存在，使用默认值")

    # 生成CSV
    data = []
    for img_id in range(81, 101):
        x, y = predictions[img_id]
        data.append({'ImageID': f'{img_id}_Fovea_X', 'value': x})
        data.append({'ImageID': f'{img_id}_Fovea_Y', 'value': y})

    df = pd.DataFrame(data)
    df.to_csv('final_submission.csv', index=False)

    print(f"\n✅ CSV已保存: final_submission.csv")

    # 统计
    x_vals = df[df['ImageID'].str.contains('_X')]['value'].values
    y_vals = df[df['ImageID'].str.contains('_Y')]['value'].values

    print(f"X: {x_vals.min():.0f}-{x_vals.max():.0f} (均值: {x_vals.mean():.0f})")
    print(f"Y: {y_vals.min():.0f}-{y_vals.max():.0f} (均值: {y_vals.mean():.0f})")

    return df


# ==================== 5. 主函数 ====================
def main():
    print("中央凹定位 - 最终版")
    print("=" * 60)

    # 训练
    model = final_train()

    # 加载最佳模型
    print("\n加载最佳模型...")
    model.load_state_dict(torch.load('final_best_model.pth', map_location=device))
    model.eval()

    # 预测
    df = final_predict(model)

    if df is not None:
        print("\n前10行结果:")
        print(df.head(10))

    print("\n" + "=" * 60)
    print("✅ 所有流程完成!")
    print("生成的文件:")
    print("  1. final_best_model.pth - 最终模型")
    print("  2. final_training.png - 训练曲线")
    print("  3. training_history.pth - 训练历史")
    print("  4. final_submission.csv - 提交文件")
    print("=" * 60)


# ==================== 运行 ====================
if __name__ == "__main__":
    main()
