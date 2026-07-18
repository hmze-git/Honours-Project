import tensorflow as tf
from tensorflow.keras import layers,models
import cupy as np
import math
import time
import pickle

#following the structure by singh et. al. make the hidden size 128 as that is the size of their first dense layer
class TFLSTMCNN:
    def __init__(self,inputShape,hiddenSize,numClasses=3,learnRate=0.01):
        self.LSTMCNN=models.Sequential()
        self.LSTMCNN.add(layers.Timelayers.TimeDistributed(layers.Conv2D(32,kernel_size=(2,2),strides=1,padding='valid',input_shape=inputShape)))
        self.LSTMCNN.add(layers.Timelayers.TimeDistributed(layers.Conv2D(32,kernel_size=(2,2),strides=1,padding='same')))
        #add batch normalisation once batching added in
        self.LSTMCNN.add(layers.Timelayers.TimeDistributed(layers.MaxPool2D(2,2)))


        self.LSTMCNN.add(layers.Timelayers.TimeDistributed(layers.Conv2D(64,kernel_size=(2,2),strides=1,padding='valid')))
        self.LSTMCNN.add(layers.Timelayers.TimeDistributed(layers.Conv2D(64,kernel_size=(2,2),strides=1,padding='same')))
        #add batch normalisation once batching added in
        self.LSTMCNN.add(layers.Timelayers.TimeDistributed(layers.MaxPool2D(2,2)))


        self.LSTMCNN.add(layers.Timelayers.TimeDistributed(layers.Conv2D(128,kernel_size=(2,2),strides=1,padding='valid')))
        self.LSTMCNN.add(layers.Timelayers.TimeDistributed(layers.Conv2D(128,kernel_size=(2,2),strides=1,padding='same')))
        #add batch normalisation once batching added in
        self.LSTMCNN.add(layers.Timelayers.TimeDistributed(layers.MaxPool2D(2,2)))

        self.LSTMCNN.add(layers.Timelayers.TimeDistributed(layers.Flatten()))
        self.LSTMCNN.add(layers.Timelayers.TimeDistributed(layers.Dense(256,activation='relu')))
        self.LSTMCNN.add(layers.Timelayers.TimeDistributed(layers.Dense(128,activation='relu')))
        self.LSTMCNN.add(layers.LSTM(hiddenSize,))


        #final 2 dense classificaiton heads
        self.LSTMCNN.add(layers.Dense(128,activation='relu'))
        self.LSTMCNN.add(layers.Dropout(0.25))
        self.LSTMCNN.add(layers.Dense(numClasses,activation='softmax'))

        self.LSTMCNN.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=learnRate, momentum=0.9),
            loss='sparse_categorical_accuracy',
            metrics=['accuracy']
        )


    def  train(self,epochs,xSet,ySet):
        self.LSTMCNN.fit(
            x=xSet,
            y=ySet,
            batch_size=0,
            epochs=epochs
        )

