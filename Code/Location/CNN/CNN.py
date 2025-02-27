import numpy as np
import pandas as pd
import scipy.io as scio
import tensorflow as tf
import matplotlib.pyplot as plt
import gzip
import os
import csv
import glob

from sklearn import model_selection

from keras.layers import GRU,LSTM,Attention
from tensorflow.keras.losses import CosineSimilarity
from tensorflow.keras.losses import MeanSquaredError
import keras.backend as K

from tensorflow import keras
from tensorflow.keras import layers,optimizers,datasets,Sequential
from tensorflow.keras.layers import Dense, Activation, InputLayer

from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical


import h5py
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D

# 混淆矩阵
import itertools
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras import regularizers

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
    # matplotlib版本问题，如果不加下面这行代码，则绘制的混淆矩阵上下只能显示一半，有的版本的matplotlib不需要下面的代码，分别试一下即可
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
    # plt.savefig('D:\Research\CSI Sensing\定位识别\localization3.0-reshape\\results\confusion_Rx2.jpg', bbox_inches='tight')
    plt.show()


numm = 100


data1File = scio.loadmat('data of position1')
data2File = scio.loadmat('data of position1')
data3File = scio.loadmat('data of position1')
data4File = scio.loadmat('data of position1')
data5File = scio.loadmat('data of position1')
data6File = scio.loadmat('data of position1')
data7File = scio.loadmat('data of position1')
data8File = scio.loadmat('data of position1')


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


for cishu in range(1,2):
    # 将数据集拆分为训练集和测试集
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X_data, y_label, test_size = 0.2, random_state = 1234)

    print('Shapes: X_train: {}, X_test: {}, y_train: {}, y_test: {}'.format(X_train.shape, X_test.shape, y_train.shape, y_test.shape))

    y_train = to_categorical(y_train, 8)
    y_test = to_categorical(y_test, 8)


    # Define CNN model
    # earlystopping_cb = tf.keras.callbacks.EarlyStopping(monitor='accuracy',patience=3,mode='max')
    num_neurons_first_layer = [None, 52, 1]

    model = Sequential()
    # model.add(Conv1D(30, 5, padding='same', activation='relu',input_shape=num_neurons_first_layer))
    model.add(Conv1D(30, 5, padding='same', activation='relu'))
    model.add(Conv1D(50, 5, padding='same', activation='relu'))
    model.add(Flatten())
    model.add(Dense(100, activation='relu',kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(50, activation='relu',kernel_regularizer=regularizers.l2(0.01)))
    #model.add(Dense(25, activation='relu', kernel_regularizer=regularizers.l2(0.01)))###
    model.add(Dense(8, activation='softmax'))

    # ==4== 指定输入层
    model.build(input_shape=[None, 52, 1])

    # ==5== 查看网络结构
    model.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # 设置您希望的学习率
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])

    #model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

    # Train the CNN
    history = model.fit(
        X_train, y_train,
        batch_size=100, epochs=900, shuffle=True, validation_split = 0.2,
    #     callbacks=[earlystopping_cb],
        verbose=1)

    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(train_loss) + 1)

    # 绘制损失图
    plt.plot(epochs, train_loss, 'g', label='loss')
    plt.plot(epochs, val_loss, 'b', label='val_loss')
    plt.title('loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()



    # Evaluate results
    results = model.evaluate(X_test, y_test)
    print(results)

    # 测试结果（ours）
    prediction = model.predict(X_test)

    # convert back to number
    y = tf.argmax(y_test, axis = 1)
    pred = tf.argmax(prediction, axis = 1)

    pred_list = pred.numpy().tolist()
    y_list = y.numpy().tolist()
    true_num = 0
    for i in range(len(pred_list)):
            if pred_list[i]==y_list[i]:
                true_num = true_num+1
    print("测试集样本数量：{}, 识别对的数量:{}".format(len(y_list), true_num))
    # print('Shapes: X_train: {}, X_test: {}, y_train: {}, y_test: {}'.format(X_train.shape, X_test.shape, y_train.shape, y_test.shape))

    print("Test set: acc=" , (float(true_num)/len(pred_list)))



    test_pred = pred_list
    test_target = y_list
    class_names = [1,2,3,4,5,6,7,8]
    classes = np.arange(8)
    classNum = 8 #识别设备数

    conf_numpy = confusion_matrix(test_target, test_pred)

    # cm_percentage = conf_numpy.astype('float') / conf_numpy.sum(axis=1)[:, np.newaxis]
    # cm_percentage = np.round(cm_percentage, 3)
    plot_confusion_matrix(conf_numpy, classes=classes, title='confusion matrix')#normalize=True

