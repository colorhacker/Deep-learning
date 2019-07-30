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


def load_data():
    images, labels = MNIST('./python-mnist/data', mode='vanilla', return_type='numpy').load_training()
    images = np.vstack((images,np.load("./temp/train_feature_28_56.npy")))
    labels = np.hstack((labels,labels))
    images = images/images.max()
    labels = to_categorical(labels)
    print(images.shape)
    print(labels.shape)
    return images, labels


def train_data():
    images, labels = load_data()
    # images, labels = MNIST('./python-mnist/data', mode='randomly_binarized', return_type='numpy').load_training()
    # labels = to_categorical(labels)
    inputs = Input(shape=(784,))
    x = Dense(392, activation='tanh')(inputs)
    x = Dense(196, activation='tanh')(x)
    x = Dense(98, activation='tanh')(x)
    predictions = Dense(10, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=predictions)
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='mean_squared_error', metrics=['accuracy'])
    history = model.fit(images, labels, validation_split=0.25, epochs=100, batch_size=256, verbose=1)  # starts training
    # predict = model.predict(test_feature)
    # evaluate = model.evaluate(test_feature,labels_t)
    model.save("./temp/keras_model")


if __name__ == '__main__':
    train_data()
