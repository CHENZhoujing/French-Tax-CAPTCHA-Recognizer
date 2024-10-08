# -*- coding: utf-8 -*-

import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image


# 1. 加载数据
def load_data(data_dir):
    X = []
    y = []
    # 获取数字文件夹列表（0-9）
    for label in sorted(os.listdir(data_dir)):
        label_path = os.path.join(data_dir, label)
        if os.path.isdir(label_path) and label.isdigit():
            # 遍历每个数字文件夹内的图像文件
            for img_name in os.listdir(label_path):
                img_path = os.path.join(label_path, img_name)
                try:
                    # 打开图像
                    img = Image.open(img_path)
                    # 确保图像大小为 (10, 15)，如果已经是该尺寸，可省略 resize
                    img = img.resize((10, 15))  # 注意顺序 (宽, 高)
                    # 将图像转换为灰度图（如果需要）
                    img = img.convert('L')
                    img_array = np.array(img)
                    # 如果需要归一化，请取消下一行的注释
                    img_array = img_array / 255.0
                    X.append(img_array)
                    y.append(int(label))
                    print(label)
                except Exception as e:
                    print(f"无法加载图像 {img_path}，错误：{e}")
    X = np.array(X)
    y = np.array(y)
    return X, y


# 指定数据集路径
data_dir = 'digital_dataset'  # 替换为您的数据集路径

# 加载数据
X, y = load_data(data_dir)

# 2. 调整数据形状
# 添加通道维度，形状为 (样本数, 高, 宽, 通道数)
X = X.reshape(-1, 10, 15, 1)

# 如果需要，将像素值归一化到 [0, 1] 范围内
# X = X / 255.0

# 3. 划分数据集
# 将数据集划分为训练集和测试集（80% 训练，20% 测试）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 构建模型
model = Sequential([
    Conv2D(25, (4, 4), activation='relu', input_shape=(10, 15, 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')  # 假设有 10 个类别（数字 0-9）
])

# 5. 编译模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 6. 训练模型
epochs = 300  # 根据需要调整
batch_size = 2  # 根据数据集大小调整

history = model.fit(X_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=0.2,  # 从训练集中划分 10% 作为验证集
                    verbose=1)

# 7. 评估模型
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print('\n测试准确率:', test_acc)

# 8. 保存模型
model.save('digit_recognition_model.h5')

# 9. 可视化训练结果
# 绘制准确率曲线
plt.figure(figsize=(8, 4))
plt.plot(history.history['accuracy'], label='训练准确率')
plt.plot(history.history['val_accuracy'], label='验证准确率')
plt.title('模型准确率')
plt.xlabel('迭代次数')
plt.ylabel('准确率')
plt.legend()
plt.show()

# 绘制损失曲线
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='训练损失')
plt.plot(history.history['val_loss'], label='验证损失')
plt.title('模型损失')
plt.xlabel('迭代次数')
plt.ylabel('损失')
plt.legend()
plt.show()
