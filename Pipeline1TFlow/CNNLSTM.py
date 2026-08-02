import tensorflow as tf
from tensorflow.keras import layers,models
from tensorflow.keras.metrics import Precision,Recall
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
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



        self.LSTMCNN.add(layers.LSTM(hiddenSize,unroll=True,recurrent_dropout=0.3,kernel_regularizer='l2'))
        self.LSTMCNN.add(layers.Dropout(0.25))

        #final 2 dense classificaiton heads
        #mayb add another head above this with size 128 or 64
        self.LSTMCNN.add(layers.Dense(32,activation='relu',kernel_regularizer='l2'))
        self.LSTMCNN.add(layers.Dropout(0.25))
        self.LSTMCNN.add(layers.Dense(numClasses,activation='softmax'))

        self.LSTMCNN.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learnRate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        


    def  train(self,epochs,dataset,validSet):
        earlyStop=tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True)
        checkPoint=tf.keras.callbacks.ModelCheckpoint('LSTMVGG16.keras',save_best_only=True)
        hist=self.LSTMCNN.fit(
            dataset,
            validation_data=validSet,
            epochs=epochs,
            callbacks=[earlyStop,checkPoint],
            verbose=2
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

        self.confusionMatrix(validSet)

    def evaluate(self,dataset):
        pass
    def confusionMatrix(self,validationDataset):

       
        preds=self.LSTMCNN.predict(validationDataset)

        predLabels=np.argmax(preds,axis=1)
        trueLabels=np.concatenate([y.numpy() for x,y in validationDataset],axis=0)

        cm=confusion_matrix(trueLabels,predLabels)

        disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=[0,1,2])
        disp.plot(cmap='Blues')
        plt.savefig('confMatrix.png')
        plt.show()

    


