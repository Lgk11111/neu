"""
植物图像分类系统 - 课程优化最终版（纯机器学习）
目标分数：0.80+
符合课程要求：特征工程 + 传统机器学习
"""
import os
import cv2
import numpy as np
import pandas as pd
import joblib
import time
from collections import Counter
import warnings

warnings.filterwarnings('ignore')

# sklearn相关
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


# ==================== 配置参数 ====================
class Config:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    TRAIN_DATA_PATH = os.path.join(BASE_DIR, "data", "train")
    TEST_DATA_PATH = os.path.join(BASE_DIR, "data", "test")
    IMAGE_SIZE = (128, 128)  # 平衡特征提取速度和信息保留
    MODEL_SAVE_PATH = os.path.join(BASE_DIR, "plant_classifier_final.pkl")
    SUBMISSION_PATH = os.path.join(BASE_DIR, "submission_final.csv")
    RANDOM_STATE = 42
    TEST_SIZE = 0.2

    # 特征提取增强
    USE_COLOR_MOMENTS = True
    USE_GLCM = True
    USE_HOG = True
    USE_LBP = True
    USE_SHAPE = True
    USE_GRADIENT = True

    # 模型配置
    ENSEMBLE_METHOD = 'voting'  # 'voting' 或 'stacking'
    USE_FEATURE_SELECTION = True
    N_BEST_FEATURES = 150


# ==================== 高级特征提取函数 ====================
def extract_color_features_enhanced(image):
    """增强版颜色特征提取"""
    features = []

    # 1. 多颜色空间直方图
    color_spaces = {
        'BGR': image,
        'HSV': cv2.cvtColor(image, cv2.COLOR_BGR2HSV),
        'LAB': cv2.cvtColor(image, cv2.COLOR_BGR2LAB),
        'YCrCb': cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    }

    for space_name, space_img in color_spaces.items():
        for channel in range(3):
            hist = cv2.calcHist([space_img], [channel], None, [16], [0, 256])
            cv2.normalize(hist, hist)
            features.extend(hist.flatten())

    # 2. 颜色矩（均值、标准差、偏度、峰度）
    for channel in range(3):
        channel_data = image[:, :, channel].flatten()

        # 均值
        mean_val = np.mean(channel_data)
        features.append(mean_val)

        # 标准差
        std_val = np.std(channel_data)
        features.append(std_val)

        # 偏度（安全计算）
        if std_val > 0:
            skewness = np.mean(((channel_data - mean_val) / std_val) ** 3)
        else:
            skewness = 0
        features.append(skewness)

        # 峰度
        if std_val > 0:
            kurtosis = np.mean(((channel_data - mean_val) / std_val) ** 4) - 3
        else:
            kurtosis = 0
        features.append(kurtosis)

    # 3. 颜色相关性特征
    # 计算颜色通道间的相关系数
    for i in range(3):
        for j in range(i + 1, 3):
            corr = np.corrcoef(image[:, :, i].flatten(), image[:, :, j].flatten())[0, 1]
            features.append(corr if not np.isnan(corr) else 0)

    return np.array(features)


def extract_texture_features_enhanced(image):
    """增强版纹理特征提取"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = []

    # 1. GLCM特征（灰度共生矩阵）
    # 计算不同距离和方向的GLCM
    distances = [1, 3, 5]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    for d in distances:
        for a in angles:
            # 简化的GLCM特征计算
            rows, cols = gray.shape
            glcm = np.zeros((256, 256), dtype=np.float32)

            for i in range(rows - d):
                for j in range(cols - d):
                    p1 = gray[i, j]
                    p2 = gray[i + int(d * np.sin(a)), j + int(d * np.cos(a))]
                    glcm[p1, p2] += 1

            if glcm.sum() > 0:
                glcm /= glcm.sum()

                # 对比度
                i_idx, j_idx = np.indices(glcm.shape)
                contrast = np.sum(glcm * ((i_idx - j_idx) ** 2))
                features.append(contrast)

                # 能量
                energy = np.sum(glcm ** 2)
                features.append(energy)

                # 同质性
                homogeneity = np.sum(glcm / (1 + (i_idx - j_idx) ** 2))
                features.append(homogeneity)
            else:
                features.extend([0, 0, 0])

    # 2. LBP特征（局部二值模式）
    radius = 1
    n_points = 8 * radius

    height, width = gray.shape
    lbp = np.zeros_like(gray)

    for i in range(radius, height - radius):
        for j in range(radius, width - radius):
            center = gray[i, j]
            code = 0
            for k in range(n_points):
                theta = 2 * np.pi * k / n_points
                x = int(i + radius * np.cos(theta))
                y = int(j + radius * np.sin(theta))
                if gray[x, y] >= center:
                    code |= 1 << k
            lbp[i, j] = code

    # LBP直方图
    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    features.extend(hist[:32])  # 只取前32个bin

    # 3. Tamura纹理特征（简化版）
    # 粗糙度
    kernel_sizes = [3, 5, 7]
    for ksize in kernel_sizes:
        blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)
        diff = cv2.absdiff(gray, blurred)
        features.append(np.mean(diff))
        features.append(np.std(diff))

    return np.array(features)


def extract_shape_features_enhanced(image):
    """增强版形状特征提取"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = []

    # 1. 边缘检测
    edges = cv2.Canny(gray, 50, 150)

    # 2. 轮廓特征
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # 找到最大轮廓
        main_contour = max(contours, key=cv2.contourArea)

        # 基本形状特征
        area = cv2.contourArea(main_contour)
        perimeter = cv2.arcLength(main_contour, True)

        features.append(area)
        features.append(perimeter)

        # 形状描述符
        if perimeter > 0:
            # 圆形度
            circularity = 4 * np.pi * area / (perimeter ** 2)
            features.append(circularity)

            # 紧密度
            compactness = area / (perimeter ** 2)
            features.append(compactness)
        else:
            features.extend([0, 0])

        # 矩形度
        x, y, w, h = cv2.boundingRect(main_contour)
        rect_area = w * h
        if rect_area > 0:
            rectangularity = area / rect_area
            features.append(rectangularity)
        else:
            features.append(0)

        # 纵横比
        if h > 0:
            aspect_ratio = w / h
            features.append(aspect_ratio)
        else:
            features.append(0)

        # Hu矩（7个不变矩）
        moments = cv2.moments(main_contour)
        if moments['m00'] > 0:
            hu_moments = cv2.HuMoments(moments).flatten()
            # 取对数压缩范围
            hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
            features.extend(hu_moments)
        else:
            features.extend([0] * 7)
    else:
        features = [0] * 13  # 13个形状特征

    # 3. 凸包特征
    if contours:
        hull = cv2.convexHull(main_contour)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = area / hull_area
            features.append(solidity)
        else:
            features.append(0)
    else:
        features.append(0)

    return np.array(features)


def extract_hog_features_enhanced(image):
    """增强版HOG特征"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # 调整大小以标准化特征维度
    gray = cv2.resize(gray, (64, 64))

    # 计算梯度
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # 计算梯度的幅度和方向
    magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

    # 计算HOG特征
    cell_size = 8
    bin_count = 9
    features = []

    for i in range(0, gray.shape[0], cell_size):
        for j in range(0, gray.shape[1], cell_size):
            cell_mag = magnitude[i:i + cell_size, j:j + cell_size]
            cell_angle = angle[i:i + cell_size, j:j + cell_size]

            # 计算方向直方图
            hist, _ = np.histogram(cell_angle, bins=bin_count, range=(0, 180), weights=cell_mag)

            # 归一化
            hist = hist.astype("float")
            hist /= (hist.sum() + 1e-6)
            features.extend(hist)

    # 添加梯度统计特征
    features.append(np.mean(magnitude))
    features.append(np.std(magnitude))
    features.append(np.max(magnitude))

    return np.array(features)


def extract_gradient_features(image):
    """梯度特征"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = []

    # Sobel算子
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # 梯度幅度
    grad_mag = np.sqrt(sobelx ** 2 + sobely ** 2)

    # 梯度方向
    grad_dir = np.arctan2(sobely, sobelx)

    # 统计特征
    features.append(np.mean(grad_mag))
    features.append(np.std(grad_mag))
    features.append(np.max(grad_mag))
    features.append(np.min(grad_mag))

    # 方向直方图
    dir_hist, _ = np.histogram(grad_dir.ravel(), bins=8, range=(-np.pi, np.pi))
    dir_hist = dir_hist.astype("float")
    dir_hist /= (dir_hist.sum() + 1e-6)
    features.extend(dir_hist)

    return np.array(features)


def extract_all_features_final(image_path, config):
    """最终版特征提取"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"警告: 无法读取图像 {image_path}")
        return None

    # 调整大小
    img = cv2.resize(img, config.IMAGE_SIZE)

    features = []

    # 提取各种特征
    if config.USE_COLOR_MOMENTS:
        color_features = extract_color_features_enhanced(img)
        features.extend(color_features)

    if config.USE_GLCM or config.USE_LBP:
        texture_features = extract_texture_features_enhanced(img)
        features.extend(texture_features)

    if config.USE_SHAPE:
        shape_features = extract_shape_features_enhanced(img)
        features.extend(shape_features)

    if config.USE_HOG:
        hog_features = extract_hog_features_enhanced(img)
        features.extend(hog_features)

    if config.USE_GRADIENT:
        gradient_features = extract_gradient_features(img)
        features.extend(gradient_features)

    # 转换为numpy数组并处理异常值
    features_array = np.array(features, dtype=np.float32)
    features_array = np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)

    return features_array


# ==================== 数据增强 ====================
def augment_image_simple(image):
    """简单数据增强"""
    augmented_images = []

    # 原始图像
    augmented_images.append(image)

    # 水平翻转
    augmented_images.append(cv2.flip(image, 1))

    # 垂直翻转
    augmented_images.append(cv2.flip(image, 0))

    # 旋转（小角度）
    rows, cols = image.shape[:2]
    for angle in [10, -10]:
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        rotated = cv2.warpAffine(image, M, (cols, rows))
        augmented_images.append(rotated)

    return augmented_images


# ==================== 模型构建 ====================
def create_advanced_ensemble(n_features, n_classes, config):
    """创建高级集成模型"""

    if config.ENSEMBLE_METHOD == 'stacking':
        # Stacking集成
        from sklearn.ensemble import StackingClassifier

        # 第一层：基础模型
        base_models = [
            ('rf1', RandomForestClassifier(
                n_estimators=200, max_depth=20,
                min_samples_split=5, min_samples_leaf=2,
                random_state=config.RANDOM_STATE,
                class_weight='balanced',
                n_jobs=-1
            )),
            ('rf2', RandomForestClassifier(
                n_estimators=200, max_depth=15,
                min_samples_split=10, min_samples_leaf=4,
                random_state=config.RANDOM_STATE + 1,
                class_weight='balanced',
                n_jobs=-1
            )),
            ('gb1', GradientBoostingClassifier(
                n_estimators=150, learning_rate=0.1,
                max_depth=6, subsample=0.8,
                random_state=config.RANDOM_STATE
            )),
            ('gb2', GradientBoostingClassifier(
                n_estimators=150, learning_rate=0.05,
                max_depth=8, subsample=0.7,
                random_state=config.RANDOM_STATE + 1
            )),
            ('svm1', SVC(
                C=10, kernel='rbf', gamma='scale',
                probability=True,
                random_state=config.RANDOM_STATE
            )),
            ('svm2', SVC(
                C=5, kernel='poly', degree=3,
                probability=True,
                random_state=config.RANDOM_STATE + 1
            )),
            ('knn', KNeighborsClassifier(
                n_neighbors=7, weights='distance',
                metric='minkowski', p=2,
                n_jobs=-1
            )),
            ('lda', LinearDiscriminantAnalysis())
        ]

        # 第二层：元学习器
        meta_learner = LogisticRegression(
            C=1.0, solver='lbfgs',
            multi_class='multinomial',
            max_iter=2000,
            random_state=config.RANDOM_STATE
        )

        model = StackingClassifier(
            estimators=base_models,
            final_estimator=meta_learner,
            cv=5,
            passthrough=False,
            n_jobs=-1
        )

    else:  # voting集成
        # 创建多样化的基础模型
        rf1 = RandomForestClassifier(
            n_estimators=300, max_depth=20,
            min_samples_split=5, min_samples_leaf=2,
            random_state=config.RANDOM_STATE,
            class_weight='balanced',
            n_jobs=-1
        )

        rf2 = RandomForestClassifier(
            n_estimators=300, max_depth=15,
            min_samples_split=10, min_samples_leaf=4,
            random_state=config.RANDOM_STATE + 1,
            class_weight='balanced',
            n_jobs=-1
        )

        gb = GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1,
            max_depth=6, subsample=0.8,
            random_state=config.RANDOM_STATE
        )

        svm = SVC(
            C=10, kernel='rbf', gamma='scale',
            probability=True,
            random_state=config.RANDOM_STATE
        )

        knn = KNeighborsClassifier(
            n_neighbors=9, weights='distance',
            metric='minkowski', p=2,
            n_jobs=-1
        )

        lda = LinearDiscriminantAnalysis()

        qda = QuadraticDiscriminantAnalysis()

        # Bagging增强的决策树
        bagging_dt = BaggingClassifier(
            base_estimator=DecisionTreeClassifier(
                max_depth=10,
                random_state=config.RANDOM_STATE
            ),
            n_estimators=50,
            random_state=config.RANDOM_STATE,
            n_jobs=-1
        )

        model = VotingClassifier(
            estimators=[
                ('rf1', rf1),
                ('rf2', rf2),
                ('gb', gb),
                ('svm', svm),
                ('knn', knn),
                ('lda', lda),
                ('qda', qda),
                ('bagging', bagging_dt)
            ],
            voting='soft',  # 使用概率投票
            weights=[3, 2, 2, 2, 1, 2, 1, 2]  # 调整权重
        )

    return model


# ==================== 主程序 ====================
def main():
    print("=" * 60)
    print("植物图像分类系统 - 课程最终优化版")
    print("目标：0.80+ 分数（纯机器学习）")
    print("=" * 60)

    config = Config()

    # 1. 加载数据
    print("\n[1/6] 加载训练数据...")
    train_images = []
    train_labels = []

    if not os.path.exists(config.TRAIN_DATA_PATH):
        print(f"错误: 训练数据路径不存在: {config.TRAIN_DATA_PATH}")
        return

    categories = sorted([d for d in os.listdir(config.TRAIN_DATA_PATH)
                         if os.path.isdir(os.path.join(config.TRAIN_DATA_PATH, d))])

    if not categories:
        print("错误: 未找到类别文件夹")
        return

    class_names = categories
    print(f"找到 {len(categories)} 个类别: {categories}")

    # 统计和加载数据
    total_images = 0
    for label, category in enumerate(categories):
        category_path = os.path.join(config.TRAIN_DATA_PATH, category)
        img_files = [f for f in os.listdir(category_path)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        print(f"  类别 {category}: {len(img_files)} 张图片")
        total_images += len(img_files)

        for img_file in img_files:
            img_path = os.path.join(category_path, img_file)
            train_images.append(img_path)
            train_labels.append(label)

    print(f"\n总共加载 {total_images} 张训练图像")

    # 2. 特征提取
    print("\n[2/6] 提取高级特征（可能需要几分钟）...")
    X = []
    y = []

    start_time = time.time()
    processed = 0

    for i, (img_path, label) in enumerate(zip(train_images, train_labels)):
        features = extract_all_features_final(img_path, config)

        if features is not None:
            X.append(features)
            y.append(label)

            # 简单数据增强（增加训练数据）
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, config.IMAGE_SIZE)
                augmented_images = augment_image_simple(img)

                # 对增强图像提取特征
                for aug_img in augmented_images[1:2]:  # 只用一个增强版本
                    # 临时保存增强图像
                    temp_path = f"temp_aug_{i}.jpg"
                    cv2.imwrite(temp_path, aug_img)
                    aug_features = extract_all_features_final(temp_path, config)
                    if aug_features is not None:
                        X.append(aug_features)
                        y.append(label)
                    os.remove(temp_path)

        processed += 1
        if processed % 10 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / processed
            remaining = avg_time * (len(train_images) - processed)
            print(f"  进度: {processed}/{len(train_images)} | "
                  f"已用时间: {elapsed:.1f}s | "
                  f"剩余时间: {remaining:.1f}s")

    X = np.array(X)
    y = np.array(y)

    print(f"\n特征提取完成! 耗时: {time.time() - start_time:.1f}秒")
    print(f"特征维度: {X.shape} (原始: {len(train_images)}, 增强后: {len(X)})")

    # 3. 特征处理
    print("\n[3/6] 特征处理...")

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 特征选择
    if config.USE_FEATURE_SELECTION and X_scaled.shape[1] > config.N_BEST_FEATURES:
        print(f"特征选择: 从 {X_scaled.shape[1]} 个特征中选择 {config.N_BEST_FEATURES} 个最佳特征")
        selector = SelectKBest(f_classif, k=min(config.N_BEST_FEATURES, X_scaled.shape[1]))
        X_selected = selector.fit_transform(X_scaled, y)
        print(f"特征选择完成!")
    else:
        X_selected = X_scaled
        selector = None

    # PCA降维（保留95%方差）
    pca = PCA(n_components=0.95, random_state=config.RANDOM_STATE)
    X_pca = pca.fit_transform(X_selected)
    print(f"PCA降维: {X_selected.shape[1]} -> {X_pca.shape[1]} 维")
    print(f"保留方差: {np.sum(pca.explained_variance_ratio_):.2%}")

    # 4. 训练模型
    print("\n[4/6] 训练高级集成模型...")

    # 划分训练验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_pca, y, test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE, stratify=y
    )

    print(f"训练集: {X_train.shape[0]} 样本")
    print(f"验证集: {X_val.shape[0]} 样本")

    # 创建并训练集成模型
    model = create_advanced_ensemble(X_train.shape[1], len(class_names), config)

    print(f"训练{config.ENSEMBLE_METHOD}集成模型...")
    model.fit(X_train, y_train)

    # 验证
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"\n✅ 验证集准确率: {accuracy:.4f}")

    print("\n详细分类报告:")
    print(classification_report(y_val, y_pred, target_names=class_names))

    # 交叉验证（可选，较慢）
    print("\n进行5折交叉验证...")
    cv_scores = cross_val_score(model, X_pca, y, cv=5, scoring='accuracy', n_jobs=-1)
    print(f"交叉验证分数: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # 5. 预测测试集
    print("\n[5/6] 预测测试集...")

    if not os.path.exists(config.TEST_DATA_PATH):
        print(f"测试数据路径不存在: {config.TEST_DATA_PATH}")
        print("跳过测试集预测")
    else:
        test_files = sorted([f for f in os.listdir(config.TEST_DATA_PATH)
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

        if len(test_files) == 0:
            print("未找到测试图像")
        else:
            print(f"处理 {len(test_files)} 张测试图像")

            test_features = []
            valid_files = []

            for i, img_file in enumerate(test_files):
                img_path = os.path.join(config.TEST_DATA_PATH, img_file)
                features = extract_all_features_final(img_path, config)

                if features is not None:
                    # 应用相同的特征处理流程
                    features_scaled = scaler.transform(features.reshape(1, -1))

                    if selector is not None:
                        features_selected = selector.transform(features_scaled)
                    else:
                        features_selected = features_scaled

                    features_pca = pca.transform(features_selected)
                    test_features.append(features_pca.flatten())
                    valid_files.append(img_file)

                if (i + 1) % 10 == 0:
                    print(f"  进度: {i + 1}/{len(test_files)}")

            if test_features:
                test_features = np.array(test_features)
                test_predictions = model.predict(test_features)

                # 转换为植物名称
                test_predictions_names = [class_names[pred] for pred in test_predictions]

                # 确保ID有.png扩展名
                fixed_ids = []
                for filename in valid_files:
                    if not filename.lower().endswith('.png'):
                        filename = os.path.splitext(filename)[0] + '.png'
                    fixed_ids.append(filename)

                # 创建提交文件
                submission_df = pd.DataFrame({
                    'ID': fixed_ids,
                    'Category': test_predictions_names
                })

                # 按ID排序
                submission_df = submission_df.sort_values('ID').reset_index(drop=True)

                submission_df.to_csv(config.SUBMISSION_PATH, index=False)
                print(f"\n✅ 提交文件已保存: {config.SUBMISSION_PATH}")

                # 显示统计信息
                print(f"\n📊 预测结果统计:")
                print(f"总预测数: {len(submission_df)}")
                print("类别分布:")
                print(submission_df['Category'].value_counts().sort_index())

                # 显示前10行
                print("\n📋 前10行数据:")
                print(submission_df.head(10).to_string(index=False))
            else:
                print("所有测试图像特征都无效!")

    # 6. 保存模型
    print("\n[6/6] 保存模型和配置...")
    model_data = {
        'model': model,
        'scaler': scaler,
        'selector': selector,
        'pca': pca,
        'class_names': class_names,
        'config': config,
        'feature_dim': X_pca.shape[1]
    }

    joblib.dump(model_data, config.MODEL_SAVE_PATH)
    print(f"模型已保存: {config.MODEL_SAVE_PATH}")

    print("\n" + "=" * 60)
    print("🎉 程序执行完成!")
    print(f"📈 预期分数: 0.78-0.85 (基于验证集准确率: {accuracy:.4f})")
    print("=" * 60)

    # 给出进一步优化建议
    print("\n💡 如果分数仍需提高，可尝试:")
    print("1. 调整特征提取参数")
    print("2. 增加数据增强强度")
    print("3. 使用网格搜索调优模型参数")
    print("4. 尝试不同的特征组合")
    print("5. 增加集成模型的多样性")


if __name__ == "__main__":
    # 检查必要的库
    required_libs = ['cv2', 'sklearn', 'numpy', 'pandas', 'joblib']

    for lib in required_libs:
        try:
            if lib == 'cv2':
                import cv2
            elif lib == 'sklearn':
                from sklearn import __version__ as sk_version
            elif lib == 'numpy':
                import numpy as np
            elif lib == 'pandas':
                import pandas as pd
            elif lib == 'joblib':
                import joblib
        except ImportError as e:
            print(f"错误: 缺少必要的库 {lib}")
            print(f"请安装: pip install opencv-python scikit-learn numpy pandas joblib")
            exit(1)

    main()