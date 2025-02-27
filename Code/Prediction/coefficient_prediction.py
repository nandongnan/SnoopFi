import numpy as np
import pandas as pd
import scipy.io as scio
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import glob

from sklearn import model_selection

from keras.layers import GRU,LSTM,Attention
from tensorflow.keras.losses import CosineSimilarity
from tensorflow.keras.losses import MeanSquaredError
import keras.backend as K
from tensorflow import keras
from tensorflow.keras import layers,optimizers,datasets,Sequential
from tensorflow.keras.layers import Dense, Activation, InputLayer, BatchNormalization, Dropout
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from keras.layers import Layer
from keras import initializers, regularizers, constraints

import itertools
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# for num in range(9,10):
#     for cishu in range(1,2):
for num in range(1,2):
    for cishu in range(1,2):
        #version7 实际数据(输入由3个特征变成6个特征, 输出为52个子载波)
        #样本数*6
        coefficient_data_real = scio.loadmat('your key.mat')
        #样本数*52
        predict_data_complex = scio.loadmat('obfuscation response.mat')

        coefficient_data_real = coefficient_data_real['coefficient_data']
        predict_data_complex = predict_data_complex['ultimateData']
        # predict_data_complex = predict_data_complex['predict_data']

        # print(coefficient_data.shape)
        # print(predict_data.shape)

        # print(predict_real.shape)
        # print(predict_img.shape)

        print(coefficient_data_real.shape)
        print(predict_data_complex.shape)


        # 将数据集拆分为训练集和测试集
        X_train, X_test, y_train, y_test = model_selection.train_test_split(coefficient_data_real, predict_data_complex, test_size = 0.2, random_state = 1234)

        print('Shapes: X_train: {}, X_test: {}, y_train: {}, y_test: {}'.format(X_train.shape, X_test.shape, y_train.shape, y_test.shape))


        # 获取数据集的长度
        data_len = len(X_train)

        # 设置要抽样的百分比
        # sample_percentage = np.arange(0.8, 1, 0.02)
        # sample_percentage = np.arange(0.025, 0.8, 0.02)
        sample_percentage = [1]
        test_mse = []
        test_score =[]






        cosine_similarity = []

        # 绘制混淆矩阵
        def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
            """
            - cm : 计算出的混淆矩阵的值
            - classes : 混淆矩阵中每一行每一列对应的列
            - normalize : True:显示百分比, False:显示个数
            """
            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                print("显示百分比：")
                np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
                print(cm)
            else:
                print('显示具体数字：')
                print(cm)
            plt.imshow(cm, interpolation='nearest', cmap=cmap)
            plt.title(title)
            plt.colorbar()
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=45)
            plt.yticks(tick_marks, classes)

            plt.ylim(len(classes) - 0.5, -0.5)
            fmt = '.2f' if normalize else 'd'
            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, format(cm[i, j], fmt),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            # plt.savefig('D:\\Research\\CSI Sensing\系数预测\\result\\ratio-data-result\\fig\\confusion.jpg', bbox_inches='tight')
            plt.show()

        # def cosine_similarity_loss(y_true, y_pred):
        #     dot_product = tf.reduce_sum(y_true * y_pred, axis=-1)
        #     true_norm = tf.norm(y_true, axis=-1)
        #     pred_norm = tf.norm(y_pred, axis=-1)
        #     cosine_similarity = dot_product / (true_norm * pred_norm)
        #     return cosine_similarity


        def cosine_similarity_loss(y_true, y_pred):
            # 从每个信号中减去直流分量（均值）
            y_true_centered = y_true - tf.reduce_mean(y_true, axis=-1, keepdims=True)
            y_pred_centered = y_pred - tf.reduce_mean(y_pred, axis=-1, keepdims=True)

            dot_product = tf.reduce_sum(y_true_centered * y_pred_centered, axis=-1)  # 计算去直流分量后的点积
            true_norm = tf.norm(y_true_centered, axis=-1)  # 计算去直流分量后的真实信号的范数
            pred_norm = tf.norm(y_pred_centered, axis=-1)  # 计算去直流分量后的预测信号的范数

            cosine_similarity = dot_product / (true_norm * pred_norm)  # 计算余弦相似度
            return cosine_similarity


        def rmse(y_true, y_pred):
            return K.sqrt(K.mean(K.square(y_pred - y_true)))


        for ratio in sample_percentage:
            X_train, X_test, y_train, y_test = model_selection.train_test_split(coefficient_data_real, predict_data_complex,
                                                                                test_size=0.2, random_state=1234)

            # print('Shapes: X_train: {}, X_test: {}, y_train: {}, y_test: {}'.format(X_train.shape, X_test.shape, y_train.shape,
            #                                                                         y_test.shape))
            # 获取数据集的长度
            data_len = len(X_train)

            # 计算要抽样的数据点数量

            sample_size = int(data_len * ratio)
            # 生成随机索引
            random_indices = np.random.choice(data_len, size=sample_size, replace=False)

            # 使用随机索引从数据和标签中获取子集
            X_train = X_train[random_indices]
            y_train = y_train[random_indices]

            print('Radio Shapes: X_train: {}, X_test: {}, y_train: {}, y_test: {}'.format(X_train.shape, X_test.shape, y_train.shape,
                                                                                    y_test.shape))

            # X_train = tf.cast(X_train, tf.float32)
            # X_test = tf.cast(X_test, tf.float32)
            y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)
            y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)

            X_train = X_train.reshape(len(X_train), 1, 6)
            X_test = X_test.reshape(len(X_test), 1, 6)

            # X_train = X_train.reshape(len(X_train), 1, 3)
            # X_test = X_test.reshape(len(X_test), 1, 3)

            model = Sequential()
            # model.add(GRU(256, activation = 'relu', input_shape=(1, 3)))
            model.add(GRU(256, activation='relu', input_shape=(1, 6)))
            # model.add(GRU(256, activation = 'relu', return_sequences = True))
            # model.add(GRU(256, activation = 'tanh', return_sequences = True))
            # model.add(GRU(256, activation='relu'))
            # model.add(Dense(64, input_dim = 3, activation='relu'))  # 第一个隐藏层，包含64个神经元
            # model.add(Dense(32, activation='relu'))  # 第二个隐藏层，包含32个神经元
            model.add(Dense(512, activation='relu'))
            model.add(Dense(256, activation='relu'))
            model.add(Dense(128, activation='relu'))
            #model.add(Dense(64, activation='relu'))
            # model.add(Dense(56))  # 输出层，输出56个数值
            model.add(Dense(52))  # 输出层，输出52个数值

            # model.summary()

            # 编译模型
            # model.compile(loss='mean_squared_error', optimizer='adam')  # 使用均方误差损失函数和Adam优化器
            # model.compile(optimizer='adam', loss= CosineSimilarity(axis=1))
            # model.compile(optimizer='adam', loss= 'mse', metrics =[cosine_similarity_loss] )
            model.compile(optimizer='adam', loss='mse', metrics=[rmse])

            # 训练模型
            model.fit(X_train, y_train, epochs=150, batch_size=16, validation_split=0.1, shuffle=True,
                      verbose=1)  # 使用80%的数据作为训练集，20%作为验证集

            train_loss = model.history.history['loss']
            val_loss = model.history.history['val_loss']

            epochs_range = range(len(train_loss))  # 横坐标，网络循环了几次

            # 损失曲线
            plt.plot(epochs_range, train_loss, label='Training_loss')
            plt.plot(epochs_range, val_loss, label='validation_loss')
            plt.legend()
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.title('Loss')

            # 使用模型进行预测
            y_predict = model.predict(X_test)  # 得到预测结果，shape为(1, 56)

            test_number = len(X_test)
            nmse_all = np.zeros(test_number)

            print(y_predict.dtype)
            print(y_test.dtype)

            for i in range(test_number):
                e1 = np.square(np.linalg.norm(y_test[i, :] - y_predict[i, :], ord=2, axis=None, keepdims=False))
                e2 = np.square(np.linalg.norm(y_test[i, :], ord=2, axis=None, keepdims=False))
                nmse_all[i] = (e1 / e2)
            nmse_average = np.mean(nmse_all)
            score = model.evaluate(X_test, y_test)

            cosine_similarity.append(cosine_similarity_loss(y_predict, y_test))


            print('Result: NMSE: {}, score: {}, cosine_similarity:{}'.format(nmse_average, score, cosine_similarity_loss(y_predict, y_test)))

            test_mse.append(nmse_average)
            test_score.append(score)

            # loss = np.array(loss)
            # val_loss = np.array(val_loss)
            m=int(0.9*int(data_len * ratio))
            # 保存训练好的模型
            model.save("6feature_model1.h5",
                       options='utf-8')
            print(test_mse)
            print(cosine_similarity)















