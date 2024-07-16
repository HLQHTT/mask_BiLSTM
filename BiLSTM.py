import os
# This guide can only be run with the tensorflow backend.
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import pandas as pd
import numpy as np
#np.object = object
seed = 2
np.random.seed(seed)  # 指定随机数种子
from data_generate import data_generate, ESOL_data_generate
import matplotlib.pyplot as plt
#import keras
from tensorflow import keras
from keras import layers
from keras.constraints import max_norm
import tensorflow as tf
from keras import initializers, constraints,activations,regularizers
from keras import backend as K
from keras.layers import Layer, Lambda
from sklearn.model_selection import train_test_split
from attention import Attention


#dropout_rate = 0.25
batch_size = 128
epochs = 40
activation = 'relu'
#索引词表的最大个数42，单个SMILES最大长度100，但是加上一头一尾，所以总长102
top_words=38 
#光敏分子数据集
max_words=306
#ESOL数据集
# max_words=102

@tf.keras.utils.register_keras_serializable(package="my_self_attention")
class self_attention(Layer):
    #返回值：返回的不是attention权重，而是每个timestep乘以权重后相加得到的向量。
    #输入:输入是rnn的timesteps，也是最长输入序列的长度
    def __init__(self, step_dim, 
                 features_dim, 
                 #mask,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')
 
        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
 
        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)
 
        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = features_dim
        #self.mask = mask
        super(self_attention, self).__init__(**kwargs)
 
    def build(self, input_shape):
        assert len(input_shape) == 3
 
        self.W = self.add_weight(shape=(input_shape[-1],),initializer=self.init,name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,constraint=self.W_constraint)
        self.features_dim = input_shape[-1]
 
        if self.bias:
            self.b = self.add_weight(shape=(input_shape[1],),initializer='zero', name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,constraint=self.b_constraint)
        else:
            self.b = None
        self.built = True
 
    # def compute_mask(self, input, input_mask=None):
    #     return None     ## 后面的层不需要mask了，所以这里可以直接返回none
 
    def call(self, x, mask=None):
        features_dim = self.features_dim    ## 这里应该是 step_dim是我们指定的参数，它等于input_shape[1],也就是rnn的timesteps
        step_dim = self.step_dim
        
        # 输入和参数分别reshape再点乘后，tensor.shape变成了(batch_size*timesteps, 1),之后每个batch要分开进行归一化
         # 所以应该有 eij = K.reshape(..., (-1, timesteps))
 
        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),K.reshape(self.W, (features_dim, 1))), (-1, step_dim))
        if self.bias:
            eij += self.b        
        eij = K.tanh(eij)    #RNN一般默认激活函数为tanh, 对attention来说激活函数差别不大，因为要做softmax
        a = K.exp(eij)
        if mask is not None:    ## 如果前面的层有mask，那么后面这些被mask掉的timestep肯定是不能参与计算输出的，也就是将他们attention权重设为0
            a *= K.cast(mask, K.floatx())   ## cast是做类型转换，keras计算时会检查类型，可能是因为用gpu的原因
 
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)      # a = K.expand_dims(a, axis=-1) , axis默认为-1， 表示在最后扩充一个维度。比如shape = (3,)变成 (3, 1)
        ## 此时a.shape = (batch_size, timesteps, 1), x.shape = (batch_size, timesteps, units)
        weighted_input = x * a    
        # weighted_input的shape为 (batch_size, timesteps, units), 每个timestep的输出向量已经乘上了该timestep的权重
        # weighted_input在axis=1上取和，返回值的shape为 (batch_size, 1, units)
        return K.sum(weighted_input, axis=1)
    
    def get_config(self):
        config = super(self_attention, self).get_config()
        config.update(
            {
                "step_dim": self.step_dim,
                #"mask": self.mask,
                "W_regularizer": self.W_regularizer,
                "b_regularizer": self.b_regularizer,
                "W_constraint": self.W_constraint,
                "b_constraint": self.b_constraint,
                "bias": self.bias,
                "features_dim": self.features_dim
            }
        )
        return config
    
    @classmethod
    def form_config(cls, config):
        return cls(**config)

 
    def compute_output_shape(self, input_shape):    ## 返回的结果是c，其shape为 (batch_size, units)
        return input_shape[0],  self.features_dim

@tf.keras.utils.register_keras_serializable(package="my_removemask")
class RemoveMask(keras.layers.Layer):
    def __init__(self, return_masked=False, no_mask=False, **kwargs):
        super(RemoveMask, self).__init__(**kwargs)
        self.supports_masking = True
        self.no_mask = no_mask
        #self.return_masked = return_masked

    def get_config(self):
        config = super(RemoveMask, self).get_config()
        config.update(
            {
                "no_mask": self.no_mask,
                #"return_masked": self.return_masked
            }
        )
        return config

    def compute_mask(self, inputs, mask=None):
        return None

def r2_keras(y_true, y_pred):
    y_true = tf.reshape(y_true,(-1,1))
    y_pred = tf.reshape(y_pred,(-1,1))
    SS_res = tf.reduce_sum(tf.square(y_true - y_pred))
    SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return (1 - SS_res / (SS_tot + 10e-8))


def plot_loss(history):
    # 显示训练和验证损失图表
    plt.subplots(1,2,figsize=(10,3))
    plt.subplot(121)
    loss = history.history["loss"]
    epochs = range(1, len(loss)+1)
    val_loss = history.history["val_loss"]
    plt.plot(epochs, loss, "bo", label="Training Loss")
    plt.plot(epochs, val_loss, "r", label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()  
    plt.subplot(122)
    r2 = history.history["r2_keras"]
    val_r2 = history.history["val_r2_keras"]
    plt.plot(epochs, r2, "b-", label="Training r2")
    plt.plot(epochs, val_r2, "r--", label="Validation r2")
    plt.title("Training and Validation r2")
    plt.xlabel("Epochs")
    plt.ylabel("r2")
    plt.legend()
    plt.tight_layout()
    plt.show()

def return_pd(history, seed):
    #输出训练过程中的损失和R2
    loss = pd.DataFrame(history.history["loss"])
    val_loss = pd.DataFrame(history.history["val_loss"])
    r2 = pd.DataFrame(history.history["r2_keras"])
    val_r2 = pd.DataFrame(history.history["val_r2_keras"])
    loss.columns = ['train_loss']
    val_loss.columns = ['val_loss']
    r2.columns = ['train_r2']
    val_r2.columns = ['val_r2']
    pd.concat([loss, val_loss, r2, val_r2], axis=1).to_csv('./deep learning/mask_BiLSTM/data/train_loss_r2/train_loss_r2_{}.csv'.format(seed))



def build_model(top_words=top_words,max_words=max_words,hidden_dim=[32], model='attention+BiLSTM'): 
    if model == 'attention+BiLSTM':    
        
        inputs = layers.Input(name='inputs',shape=[max_words,], dtype='int32')
        layer = layers.Embedding(top_words+1, 64, mask_zero=True, input_length= max_words, 
                           embeddings_regularizer=keras.regularizers.l2(1e-5),
                           embeddings_constraint=keras.constraints.max_norm(3))(inputs)
        layer = layers.Masking(mask_value=0)(layer) 
        bilstm = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(layer) 
        bilstm = layers.Dropout(0.25)(bilstm)
        bilstm = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(bilstm)
    
        layer = RemoveMask()(bilstm)
        layer = layers.Dropout(0.25)(layer)
    
        ## 注意力机制 
        attention = self_attention(step_dim=max_words, features_dim = 0, 
                                   #mask=mask
                                   )(bilstm)
        layer = layers.Dense(256, activation='relu')(attention)
        layer = layers.Dropout(0.25)(layer)
        layer = layers.Dense(64,
                     activation='relu'
                     )(layer)
        layer = layers.Dropout(0.5)(layer)
        output = layers.Dense(1)(layer)
        model = keras.Model(inputs=inputs, outputs=output) 

    elif model=='attention+BiGRU':
        inputs = layers.Input(name='inputs',shape=[max_words,], dtype='int32')
        layer = layers.Embedding(top_words+1, 32, mask_zero=True, input_length= max_words, 
                           embeddings_regularizer=keras.regularizers.l2(1e-5),
                           embeddings_constraint=keras.constraints.max_norm(3))(inputs)
        attention_probs = layers.Dense(32, activation='relu', name='attention_vec')(layer)
        attention_mul =  layers.Multiply()([layer, attention_probs])
        mlp = layers.Dense(64,activation='relu')(attention_mul) #原始的全连接
        mlp = layers.Dropout(0.25)(mlp)
        #bat=BatchNormalization()(mlp)
        #act=Activation('relu')
        gru=layers.Bidirectional(layers.GRU(32))(mlp)
        gru = layers.Dropout(0.25)(gru)
        mlp = layers.Dense(16,activation='relu')(gru)
        mlp = layers.Dropout(0.5)(mlp)
        output = layers.Dense(1)(mlp)
        model = keras.Model(inputs=[inputs], outputs=output)

    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
    0.001,
    decay_steps=3000,
    decay_rate=0.96,
    staircase=True)

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)    
        
    model.compile(loss="mse", optimizer=optimizer, metrics=[r2_keras])
    return model

#定义训练函数
def train_fuc(top_words=top_words, max_words=max_words,batch_size=32,epochs=10,hidden_dim=[32],show_loss=False, model='attention+BiLSTM'):
    #构建模型
    model=build_model(max_words=max_words, top_words=top_words,model=model)
    print(model.summary())
    history=model.fit(xtrain, ytrain, batch_size=batch_size, epochs=epochs, validation_split=0.2, verbose=1)
    print('————————————训练完毕————————————')
    # 评估模型
    loss, r2_keras = model.evaluate(xtest, ytest,batch_size=batch_size)
    print("测试数据集的R2 = {:.4f}".format(r2_keras))
    
    # return_pd(history, 39)

    if show_loss:
        plot_loss(history)

    return r2_keras

#定义预测函数
def predict_fuc(top_words=top_words, max_words=max_words,batch_size=32,epochs=10,hidden_dim=[32],xtrain=None, ytrain=None, model='attention+BiLSTM'):
    #构建模型
    model=build_model(max_words=max_words, top_words=top_words,model=model)
    #训练模型
    model.fit(xtrain, ytrain, batch_size=batch_size, epochs=epochs, validation_split=0.2, verbose=1)
    print('————————————模型建立完毕————————————')
    # 预测
    #y_predict = model.predict(x_predict)
    #pd.DataFrame(model.predict(y_predict)).to_csv('../pred_val.csv')
    return model
    



if __name__ == "__main__":
    #光敏分子数据集
    xtrain, ytrain, xtest, ytest, _, __ = data_generate(9)
    #ESOL数据集
    # for i in [9, 29, 49, 69, 89]:
    # xtrain, ytrain, xtest, ytest = ESOL_data_generate(9)
    train_fuc(batch_size=batch_size, epochs=epochs, show_loss=True, model='attention+BiLSTM')
    # model = predict_fuc(batch_size=batch_size, epochs=epochs, xtrain=xtrain, ytrain=ytrain)
    # 保存模型权重为HDF5文件
    # model.save("./deep learning/mask_BiLSTM/model/BiLSTM_attention_9.h5")
    # print("-----模型保存成功-----")
    



