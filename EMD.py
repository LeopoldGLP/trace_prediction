'''
@Project ：lstm.py 
@File    ：EMD.py
@IDE     ：PyCharm 
@Author  ：Goulipeng
@e-mail  ；goulipeng@nuaa.edu.cn
@Date    ：2023/12/10 20:15 
'''

import numpy as np
import matplotlib.pyplot as plt
from PyEMD import EMD

# 生成示例数据
np.random.seed(42)
time = np.linspace(0, 10, 1000)
trajectory_data = np.sin(2 * np.pi * 1 * time) + np.random.normal(scale=0.2, size=len(time))
wind_direction = np.sin(2 * np.pi * 0.5 * time) + np.random.normal(scale=0.1, size=len(time))
wind_speed = np.abs(np.sin(2 * np.pi * 0.8 * time) + np.random.normal(scale=0.15, size=len(time)))
heading = np.sin(2 * np.pi * 0.3 * time) + np.random.normal(scale=0.05, size=len(time))

# 合并数据
combined_data = np.vstack([trajectory_data, wind_direction, wind_speed, heading]).T  # 转置

# 初始化EMD对象
emd = EMD()

# 分解数据
imfs = emd(combined_data[:, 0])  # 只选择航迹定位数据进行分解

# 绘制结果
plt.figure(figsize=(12, 8))
plt.subplot(len(imfs)+1, 1, 1)
plt.plot(time, combined_data[:, 0], 'r')  # 使用第一列数据（航迹定位数据）
plt.title('Original Trajectory Data')

for i in range(len(imfs)):
    plt.subplot(len(imfs)+1, 1, i+2)
    plt.plot(time, imfs[i], 'b')
    plt.title(f'IMF {i+1}')

plt.tight_layout()
plt.show()

