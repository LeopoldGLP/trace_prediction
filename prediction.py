'''
@Project ：PositionPredicted 
@File    ：prediction.py
@IDE     ：PyCharm 
@Author  ：Goulipeng
@e-mail  ；goulipeng@nuaa.edu.cn
@Date    ：2023/10/26 15:00 
'''
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np

# 假设这些是您的数据
X_train = np.random.rand(100, 10, 5)  # 训练集数据，形状为 (样本数, 时间步数, 特征数)
y_train = np.random.rand(100, 2)  # 训练集标签，形状为 (样本数, 输出维度)

X_val = np.random.rand(20, 10, 5)  # 验证集数据
y_val = np.random.rand(20, 2)  # 验证集标签

X_test = np.random.rand(30, 10, 5)  # 测试集数据
y_test = np.random.rand(30, 2)  # 测试集标签

# 创建模型
model = Sequential()
model.add(LSTM(units=hidden_units, input_shape=(10, 5), return_sequences=True))  # 注意，这里假设时间步数为10，特征数为5
model.add(LSTM(units=hidden_units, return_sequences=True))  # 添加更多LSTM层，视需要而定
model.add(Dense(2))  # 输出层，假设输出维度为2

# 编译模型
model.compile(loss='mean_squared_error', optimizer='adam')

# 训练模型
num_epochs = 10
batch_size = 32
model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(X_val, y_val))

# 评估模型
loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')

# 预测新数据
X_new_data = np.random.rand(5, 10, 5)  # 新数据，形状为 (样本数, 时间步数, 特征数)
predicted_positions = model.predict(X_new_data)
print('Predicted Positions:')
print(predicted_positions)
