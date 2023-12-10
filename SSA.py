'''
@Project ：lstm.py 
@File    ：SSA.py
@IDE     ：PyCharm 
@Author  ：Goulipeng
@e-mail  ；goulipeng@nuaa.edu.cn
@Date    ：2023/12/10 20:55 
'''
import numpy as np
import matplotlib.pyplot as plt
import ssa

# 生成模拟的航迹数据
np.random.seed(42)
num_points = 100
time = np.arange(num_points)
latitude = 30 + 2 * np.sin(0.2 * time) + 0.5 * np.random.normal(size=num_points)
longitude = -120 - 2 * np.cos(0.2 * time) + 0.5 * np.random.normal(size=num_points)
wind_direction = 180 + 20 * np.sin(0.1 * time) + 5 * np.random.normal(size=num_points)
wind_speed = 10 + 2 * np.sin(0.15 * time) + 1 * np.random.normal(size=num_points)
heading = 90 + 10 * np.cos(0.1 * time) + 2 * np.random.normal(size=num_points)

# 将数据组合成一个矩阵，每一列是一个变量
data_matrix = np.vstack([latitude, longitude, wind_direction, wind_speed, heading]).T

# 对航迹数据进行 SSA 分解
num_components = 5  # 选择的成分数量
components, singular_values = ssa(data_matrix, n_components=num_components)

# 绘制原始航迹数据和分解的成分
labels = ['Latitude', 'Longitude', 'Wind Direction', 'Wind Speed', 'Heading']

plt.figure(figsize=(12, 8))
for i in range(data_matrix.shape[1]):
    plt.subplot(data_matrix.shape[1] + 1, 1, i+1)
    plt.plot(time, data_matrix[:, i], label=labels[i])
    plt.legend()

# 绘制 SSA 成分
for i in range(num_components):
    plt.subplot(data_matrix.shape[1] + 1, 1, data_matrix.shape[1] + i + 1)
    plt.plot(time, components[:, i], label=f'Component {i+1}')
    plt.legend()

plt.tight_layout()
plt.show()
