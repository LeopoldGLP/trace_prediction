import numpy as np
import pandas as pd
from pyts.decomposition import SingularSpectrumAnalysis
from PyEMD import EMD
import tensorflow as tf
from tensorflow.keras import layers

# 假设你有一个包含航迹数据的DataFrame，列名分别为 'longitude', 'latitude', 'altitude', 'airspeed', 'vertical_speed', 'heading'
# 我们用随机数据生成一个示例DataFrame
np.random.seed(42)
data = np.random.rand(100, 6)
columns = ['longitude', 'latitude', 'altitude', 'airspeed', 'vertical_speed', 'heading']
trajectory_df = pd.DataFrame(data, columns=columns)

# 对经度纬度进行SSA处理
def ssa_processing(data):
    ssa = SingularSpectrumAnalysis(window_size=5)
    ssa_components = ssa.fit_transform(data)
    return ssa_components

# 对其他四个特征进行EDA处理
def eda_processing(data):
    eda = EMD(r=2, n_neighbors=10, threshold=0.1)
    eda_components = eda.fit_transform(data)
    return eda_components

# 划分训练集和测试集
train_size = int(0.8 * len(trajectory_df))
train_data = trajectory_df[:train_size]
test_data = trajectory_df[train_size:]

# 提取训练集的特征和目标
train_features = train_data[['longitude', 'latitude', 'altitude', 'airspeed', 'vertical_speed', 'heading']].values
train_target = train_data[['longitude', 'latitude']].values

# 提取测试集的特征和目标
test_features = test_data[['longitude', 'latitude', 'altitude', 'airspeed', 'vertical_speed', 'heading']].values
test_target = test_data[['longitude', 'latitude']].values

# 对特征进行SSA处理
# ssa_features = np.hstack([ssa_processing(train_features[:, :2]), eda_processing(train_features[:, 2:])])
# ssa_test_features = np.hstack([ssa_processing(test_features[:, :2]), eda_processing(test_features[:, 2:])])

# 建立Informer模型
class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert (
            self.head_dim * num_heads == embed_dim
        ), "Embedding dimension needs to be divisible by number of heads"

        self.wq = layers.Dense(embed_dim)
        self.wk = layers.Dense(embed_dim)
        self.wv = layers.Dense(embed_dim)
        self.dense = layers.Dense(embed_dim)

    def call(self, value, key, query, mask):
        batch_size = tf.shape(query)[0]

        # Linear layers
        query = self.wq(query)
        key = self.wk(key)
        value = self.wv(value)

        # Split the embedding into num_heads and reshape for scaled dot product attention
        query = tf.reshape(query, (batch_size, -1, self.num_heads, self.head_dim))
        key = tf.reshape(key, (batch_size, -1, self.num_heads, self.head_dim))
        value = tf.reshape(value, (batch_size, -1, self.num_heads, self.head_dim))

        # Transpose to align dimensions for multiplication
        query = tf.transpose(query, perm=[0, 2, 1, 3])
        key = tf.transpose(key, perm=[0, 2, 3, 1])
        value = tf.transpose(value, perm=[0, 2, 3, 1])

        # Scaled dot-product attention
        scaled_attention_logits = tf.einsum("ijkl,ijlm->ijkm", query, key) / tf.math.sqrt(
            tf.cast(self.head_dim, tf.float32)
        )
        if mask is not None:
            scaled_attention_logits += mask * -1e9

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.einsum("ijkm,ijml->ijkl", attention_weights, value)

        # Reshape and concatenate the heads
        output = tf.reshape(output, (batch_size, -1, self.embed_dim))
        output = self.dense(output)
        return output
class FeedForwardNetwork(layers.Layer):
    def __init__(self, embed_dim, ff_dim):
        super(FeedForwardNetwork, self).__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ff_dim

        self.dense1 = layers.Dense(ff_dim, activation="relu")
        self.dense2 = layers.Dense(embed_dim)

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        return x

class InformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(InformerBlock, self).__init__()
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = FeedForwardNetwork(embed_dim, ff_dim)

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training, mask):
        attn_output = self.attention(inputs, inputs, inputs, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class Informer(tf.keras.Model):
    def __init__(self, embed_dim, num_heads, ff_dim, num_blocks, mlp_dim, dropout=0.1, mlp_dropout=0.1):
        super(Informer, self).__init__()

        self.num_blocks = num_blocks

        self.blocks = [InformerBlock(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_blocks)]

        self.ffn_layer1 = FeedForwardNetwork(embed_dim, mlp_dim[0])
        self.ffn_layer2 = FeedForwardNetwork(mlp_dim[0], mlp_dim[1])

        self.final_layer = layers.Dense(1)

    def call(self, inputs, training=True):
        # mask = tf.math.reduce_any(tf.math.not_equal(inputs, 0), axis=-1, keepdims=True)
        mask = tf.cast(tf.math.reduce_any(tf.math.not_equal(inputs, 0), axis=-1, keepdims=True), dtype=tf.float32)

        for i in range(self.num_blocks):
            inputs = self.blocks[i](inputs, training, mask)

        x = tf.reduce_mean(inputs, axis=1)  # Global average pooling

        x = self.ffn_layer1(x)
        x = self.ffn_layer2(x)
        x = self.final_layer(x)

        return x

# 创建Informer模型实例
embed_dim = 64
num_heads = 4
ff_dim = 32
num_blocks = 2
mlp_dim = (64, 32)
dropout = 0.1
mlp_dropout = 0.1

informer_model = Informer(embed_dim, num_heads, ff_dim, num_blocks, mlp_dim, dropout, mlp_dropout)

# 编译模型
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
informer_model.compile(optimizer=optimizer, loss='mean_squared_error')

# 训练模型
informer_model.fit(train_features, train_target, epochs=10, batch_size=32, validation_split=0.2)

# 在测试集上进行预测
predictions = informer_model.predict(test_features)
