from keras.layers import Input, Dense
from keras.models import Model,load_model
from keras.optimizers import SGD
from keras.utils import to_categorical,plot_model
from mnist import MNIST
import matplotlib.pyplot as plt
import rebuild_matrix as rebulid
import numpy as np


def model_load():
    model = load_model("./temp/keras_model")
    return model

def load_data_s():
    images, labels = MNIST('./python-mnist/data', mode='vanilla', return_type='numpy').load_training()
    feature = np.load("./temp/kmeans_feature_7x7x7_512.npy")
    k_feature = np.load("./temp/train_feature.npy")

    label = np.copy(labels)
    image = np.empty(shape = images[0].shape)
    for i in range(k_feature.shape[0]):
        image = np.vstack((image,rebulid.rebuild_matrix_c(feature,k_feature[i],7,7,7).flatten()))
        label = np.hstack((label,labels[i]))
        print(i)
    label = to_categorical(label)
    return image,label

def load_data():
    _, labels = MNIST('./python-mnist/data', mode='randomly_binarized', return_type='numpy').load_training()
    image = np.load("./temp/train_feature_" + str(0) + ".npy")
    label = np.copy(labels)
    for i in range(1,5):
        image = np.vstack((image,np.load("./temp/train_feature_" + str(i) + ".npy")))
        label = np.hstack((label,labels))
    label = to_categorical(label)
    image = image / image.max()
    return image,label

def train_data():
    # images, labels = load_data()
    images, labels = MNIST('./python-mnist/data', mode='randomly_binarized', return_type='numpy').load_training()
    labels = to_categorical(labels)
    inputs = Input(shape=(784,))
    x = Dense(64, activation='tanh')(inputs)
    x = Dense(128, activation='tanh')(x)
    x = Dense(64, activation='tanh')(x)
    predictions = Dense(10, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=predictions)
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='mean_squared_error', metrics=['accuracy'])
    history = model.fit(images, labels, validation_split=0.25, epochs=100, batch_size=16, verbose=1)  # starts training
    # predict = model.predict(test_feature)
    # evaluate = model.evaluate(test_feature,labels_t)
    model.save("./temp/keras_model")


if __name__=='__main__':
    train_data()
