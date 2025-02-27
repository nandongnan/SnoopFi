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



data1File = scio.loadmat('data of position1')
data2File = scio.loadmat('data of position2')
data3File = scio.loadmat('data of position3')
data4File = scio.loadmat('data of position4')
data5File = scio.loadmat('data of position5')
data6File = scio.loadmat('data of position6')
data7File = scio.loadmat('data of position7')
data8File = scio.loadmat('data of position8')



local1_data = abs(data1File['merged_p'].transpose())[:100,:]
local2_data = abs(data2File['merged_p'].transpose())[:100,:]
local3_data = abs(data3File['merged_p'].transpose())[:100,:]
local4_data = abs(data4File['merged_p'].transpose())[:100,:]
local5_data = abs(data5File['merged_p'].transpose())[:100,:]
local6_data = abs(data6File['merged_p'].transpose())[:100,:]
local7_data = abs(data7File['merged_p'].transpose())[:100,:]
local8_data = abs(data8File['merged_p'].transpose())[:100,:]



label1 = np.zeros((len(local1_data),1), dtype = np.int32)
label2 = np.zeros((len(local2_data),1), dtype = np.int32)
label3 = np.zeros((len(local3_data),1), dtype = np.int32)
label4 = np.zeros((len(local4_data),1), dtype = np.int32)
label5 = np.zeros((len(local5_data),1), dtype = np.int32)
label6 = np.zeros((len(local6_data),1), dtype = np.int32)
label7 = np.zeros((len(local7_data),1), dtype = np.int32)
label8 = np.zeros((len(local8_data),1), dtype = np.int32)


label2 [label2 == 0] = 1
label3 [label3 == 0] = 2
label4 [label4 == 0] = 3
label5 [label5 == 0] = 4
label6 [label6 == 0] = 5
label7 [label7 == 0] = 6
label8 [label8 == 0] = 7


#合并数据
X_data = np.concatenate((local1_data, local2_data, local3_data, local4_data, local5_data, local6_data, local7_data, local8_data), axis=0)
y_label = np.concatenate((label1, label2, label3, label4, label5, label6, label7, label8), axis=0)


print(X_data.shape)
print(y_label.shape)


#打乱数据
# permutation = np.random.permutation(y_label.shape[0])
# shuffled_dataset = X_data[permutation, :]
# shuffled_labels = y_label[permutation]

# X_tensor = tf.convert_to_tensor(X_train, dtype=tf.float32)

# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = model_selection.train_test_split(X_data, y_label, test_size = 0.2, random_state = 1234)

print('Shapes: X_train: {}, X_test: {}, y_train: {}, y_test: {}'.format(X_train.shape, X_test.shape, y_train.shape, y_test.shape))



# 获取数据集的长度
data_len = len(X_train)

# 设置要抽样的百分比（这里是百分之一）
# sample_percentage = [0.15,0.2];
sample_percentage = [1]
test_acc = []

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

        scio.savemat('C:\\Users\\user\\Desktop\\机器学习\\解混结果\\Rx1_confusion_matrix.mat', {'conf': cm})
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
    plt.savefig('C:\\Users\\user\\Desktop\\机器学习\\解混结果\\Rx1_confusion.jpg', bbox_inches='tight')
    plt.show()

for ratio in sample_percentage:
    # 计算要抽样的数据点数量
    sample_size = int(data_len * ratio)
    # 生成随机索引
    random_indices = np.random.choice(data_len, size=sample_size, replace=False)

    # 使用随机索引从数据和标签中获取子集
    sampled_data = X_train[random_indices]
    sampled_labels = y_train[random_indices]
    sampled_data = sampled_data.reshape(len(sampled_data), 1, 52)
    sampled_labels = to_categorical(sampled_labels, 8)

    X_train = X_train.reshape(len(X_train), 1, 52)
    X_test = X_test.reshape(len(X_test), 1, 52)

    # X_train = X_train.reshape(len(X_train), 1, 52)
    # X_test = X_test.reshape(len(X_test), 1, 52)

    y_train = to_categorical(y_train, 8)
    y_test = to_categorical(y_test, 8)

    # 构建深度学习网络
    model = Sequential()
    # model.add(keras.layers.core.Masking(mask_value=0., input_shape=(2, 60)))
    model.add(GRU(512, activation='relu', input_shape=(1, 52)))
    # model.add(GRU(1024, activation = 'relu', input_shape=(1, 52), return_sequences = True))
    # model.add(Dropout(0.2))
    model.add(BatchNormalization())
    # model.add(Dropout(0.2))
    # model.add(GRU(256, activation = 'relu'))
    # model.add(BatchNormalization())
    # model.add(GRU(256, activation = 'tanh', return_sequences = True))
    # model.add(GRU(1024, activation = 'relu', return_sequences = True))
    # model.add(GRU(1024, activation = 'relu', return_sequences = True))
    # model.add(GRU(1024, activation='relu'))

    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(8, activation='softmax'))  # 输出层，输出8个数值

    model.summary()

    # （4）网络配置
    # 设置动态学习率指数衰减
    exponential_decay = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.0001,  # 初始学习率
        decay_steps=2,  # 衰减步长
        decay_rate=0.96)  # 衰减率

    # 编译
    model.compile(optimizer=optimizers.Adam(learning_rate=exponential_decay),
                  loss=tf.losses.CategoricalCrossentropy(from_logits=True),  # 交叉熵损失
                  metrics=['accuracy'])  # 准确率指标

    # 早停策略
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_accuracy',  # 验证集的准确率作为指标
        patience=15,  # 最多忍受多少个次循环没有改进
        restore_best_weights=True)  # 发生早停时，自动寻找最优的monitor参数

    # （5）网络训练
    # 指定训练集、验证集、迭代次数
    # 训练目标和验证目标需要时one_hot编码后的
    tf.config.experimental_run_functions_eagerly(True)
    network = model.fit(x=sampled_data,
                        y=sampled_labels,  # 训练集
                        validation_split=0.2,
                        epochs=60,  # 迭代多少次
                        batch_size=32,
                        # callbacks = early_stopping, # 回调函数，在训练过程中的适当时机被调用
                        shuffle=True,  # 每轮迭代之前洗牌
                        verbose=1  # 0为不在标准输出流输出日志信息，1:显示进度条，2:每个epoch输出一行记录
                        )

    # （6）模型评估

    # ==1== 计算准确率

    train_acc = model.history.history['accuracy']
    val_acc = model.history.history['val_accuracy']

    # ==2== 损失
    train_loss = model.history.history['loss']
    val_loss = model.history.history['val_loss']

    # ==3== 曲线图
    epochs_range = range(len(train_acc))  # 横坐标，网络循环了几次

    # 准确率曲线
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_acc, label='Training_acc')
    plt.plot(epochs_range, val_acc, label='validation_acc')
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title('Accuracy')

    # 损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_loss, label='Training_loss')
    plt.plot(epochs_range, val_loss, label='validation_loss')
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title('Loss')
    plt.show()

    prediction = model.predict(X_test)

    # convert back to number
    y = tf.argmax(y_test, axis=1)
    pred = tf.argmax(prediction, axis=1)
    pred_list = pred.numpy().tolist()
    y_list = y.numpy().tolist()
    true_num = 0
    for i in range(len(pred_list)):
        if pred[i] == y_list[i]:
            true_num = true_num + 1
    print("测试集样本数量：{}, 识别对的数量:{}".format(len(y_list), true_num))
    # print('Shapes: X_train: {}, X_test: {}, y_train: {}, y_test: {}'.format(X_train.shape, X_test.shape, y_train.shape, y_test.shape))

    print("Test set: acc=", (float(true_num) / len(pred_list)))

    test_acc.append((float(true_num) / len(pred_list)))

    test_pred = pred_list
    test_target = y_list
    class_names = [1, 2, 3, 4, 5, 6, 7, 8]
    classes = np.arange(8)
    classNum = 8  # 识别数

    conf_numpy = confusion_matrix(test_target, test_pred)
    print(conf_numpy)


    plot_confusion_matrix(conf_numpy, classes=classes, normalize=True, title='confusion matrix')
    # plot_confusion_matrix(cnf_matrix, classes=attack_types, normalize=False, title='Normalized confusion matrix')


print(test_acc)
# scio.savemat('test_acc.mat', {'test_acc': test_acc})
