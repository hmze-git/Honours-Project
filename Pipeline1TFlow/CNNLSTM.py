import tensorflow as tf
from tensorflow.keras import layers,models
from tensorflow.keras.metrics import Precision,Recall
import math
import time
import pickle
import numpy as np
from matplotlib import pyplot as plt

#following the structure by singh et. al. make the hidden size 128 as that is the size of their first dense layer
class TFLSTMCNN:
    def __init__(self,inputShape,hiddenSize,numClasses=3,learnRate=0.0005):
        self.LSTMCNN=models.Sequential()
        #self.LSTMCNN.add(layers.Input(shape=inputShape))
        #self.LSTMCNN.add(layers.TimeDistributed(layers.Conv2D(8,kernel_size=(3,3),strides=1,padding='valid',activation='relu')))
        #self.LSTMCNN.add(layers.TimeDistributed(layers.Conv2D(8,kernel_size=(3,3),strides=1,padding='same',activation='relu')))
        #add batch normalisation once batching added in
        #self.LSTMCNN.add(layers.TimeDistributed(layers.BatchNormalization()))
        #self.LSTMCNN.add(layers.TimeDistributed(layers.MaxPool2D(2,2)))


        #self.LSTMCNN.add(layers.TimeDistributed(layers.Conv2D(16,kernel_size=(3,3),strides=1,padding='valid',activation='relu')))
        #self.LSTMCNN.add(layers.TimeDistributed(layers.Conv2D(16,kernel_size=(3,3),strides=1,padding='same',activation='relu')))
        #add batch normalisation once batching added in
        #self.LSTMCNN.add(layers.TimeDistributed(layers.BatchNormalization()))
        #self.LSTMCNN.add(layers.TimeDistributed(layers.MaxPool2D(2,2)))


        #self.LSTMCNN.add(layers.TimeDistributed(layers.Conv2D(32,kernel_size=(3,3),strides=1,padding='valid',activation='relu')))
        #self.LSTMCNN.add(layers.TimeDistributed(layers.Conv2D(32,kernel_size=(3,3),strides=1,padding='same',activation='relu')))
        #add batch normalisation once batching added in
        #self.LSTMCNN.add(layers.TimeDistributed(layers.BatchNormalization()))
        #self.LSTMCNN.add(layers.TimeDistributed(layers.MaxPool2D(2,2)))

        #self.LSTMCNN.add(layers.TimeDistributed(layers.Flatten()))
        
        #self.LSTMCNN.add(layers.Ti meDistributed(layers.GlobalAveragePooling2D()))
        #self.LSTMCNN.add(layers.TimeDistributed(layers.Dense(64,activation='relu')))
        #self.LSTMCNN.add(layers.TimeDistributed(layers.BatchNormalization()))



        # VGGNET HERE
        print(inputShape)

        baseCNN=tf.keras.applications.VGG16(
            input_shape=inputShape[1:],
            weights="imagenet",
            include_top=False,
            pooling='avg'
        )
        baseCNN.trainable = False



        #overfitting is happening somewhere here 
        #so look into that
        #either lstm sequence is too long, hidden size also might be too big or the dropout needs to be more aggresive
        self.LSTMCNN.add(layers.TimeDistributed(baseCNN))
        self.LSTMCNN.add(layers.TimeDistributed(layers.BatchNormalization()))



        self.LSTMCNN.add(layers.LSTM(hiddenSize,unroll=True,recurrent_dropout=0.3))
        self.LSTMCNN.add(layers.Dropout(0.25))

        #final 2 dense classificaiton heads
        self.LSTMCNN.add(layers.Dense(32,activation='relu'))
        self.LSTMCNN.add(layers.Dropout(0.25))
        self.LSTMCNN.add(layers.Dense(numClasses,activation='softmax'))

        self.LSTMCNN.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learnRate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        


    def  train(self,epochs,dataset,validSet):
        callack=tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=5)
        hist=self.LSTMCNN.fit(
            dataset,
            validation_data=validSet,
            epochs=epochs,
            callbacks=[callack]
        )
        print(hist.history.keys())
        plt.plot(hist.history['accuracy'])
        plt.plot(hist.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train','val'])
        plt.savefig('NormVsValACc.png')
        plt.clf()

        plt.plot(hist.history['loss'])
        plt.plot(hist.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train','val'])
        plt.savefig('NormVsValLoss.png')




    


