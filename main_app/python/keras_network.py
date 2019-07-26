from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import SGD
from keras.utils import to_categorical
from mnist import MNIST
import matplotlib.pyplot as plt
import numpy as np


if __name__=='__main__':
    images, labels = MNIST('./python-mnist/data', mode='randomly_binarized', return_type='numpy').load_training()
    labels = to_categorical(labels)
    train_feature = np.load("./temp/train_feature.npy")
    train_feature = train_feature / train_feature.max()
    inputs = Input(shape=(16,))
    x = Dense(64, activation='tanh')(inputs)
    x = Dense(256, activation='tanh')(x)
    x = Dense(512, activation='tanh')(x)
    x = Dense(256, activation='tanh')(x)
    x = Dense(64, activation='tanh')(x)
    predictions = Dense(10, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=predictions)
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='mean_squared_error', metrics=['accuracy'])
    history = model.fit(train_feature, labels, validation_split=0.25, epochs=100, batch_size=16, verbose=1)  # starts training
    #print(dir(history))
    #print(history.history)
