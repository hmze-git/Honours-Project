import tensorflow as tf
from tensorflow.keras import layers,models
from tensorflow.keras.metrics import Precision,Recall
import math
import time
import pickle
import numpy as np

#following the structure by singh et. al. make the hidden size 128 as that is the size of their first dense layer
class TFLSTMCNN:
    def __init__(self,inputShape,hiddenSize,numClasses=3,learnRate=0.001):
        self.LSTMCNN=models.Sequential()
        self.LSTMCNN.add(layers.Input(shape=inputShape))
        self.LSTMCNN.add(layers.TimeDistributed(layers.Conv2D(8,kernel_size=(3,3),strides=1,padding='valid',activation='relu')))
        self.LSTMCNN.add(layers.TimeDistributed(layers.Conv2D(8,kernel_size=(3,3),strides=1,padding='same',activation='relu')))
        #add batch normalisation once batching added in
        self.LSTMCNN.add(layers.TimeDistributed(layers.BatchNormalization()))
        self.LSTMCNN.add(layers.TimeDistributed(layers.MaxPool2D(2,2)))


        self.LSTMCNN.add(layers.TimeDistributed(layers.Conv2D(16,kernel_size=(3,3),strides=1,padding='valid',activation='relu')))
        self.LSTMCNN.add(layers.TimeDistributed(layers.Conv2D(16,kernel_size=(3,3),strides=1,padding='same',activation='relu')))
        #add batch normalisation once batching added in
        self.LSTMCNN.add(layers.TimeDistributed(layers.BatchNormalization()))
        self.LSTMCNN.add(layers.TimeDistributed(layers.MaxPool2D(2,2)))


        self.LSTMCNN.add(layers.TimeDistributed(layers.Conv2D(32,kernel_size=(3,3),strides=1,padding='valid',activation='relu')))
        self.LSTMCNN.add(layers.TimeDistributed(layers.Conv2D(32,kernel_size=(3,3),strides=1,padding='same',activation='relu')))
        #add batch normalisation once batching added in
        self.LSTMCNN.add(layers.TimeDistributed(layers.BatchNormalization()))
        self.LSTMCNN.add(layers.TimeDistributed(layers.MaxPool2D(2,2)))

        #self.LSTMCNN.add(layers.TimeDistributed(layers.Flatten()))
        
        self.LSTMCNN.add(layers.TimeDistributed(layers.GlobalAveragePooling2D()))
        self.LSTMCNN.add(layers.TimeDistributed(layers.Dense(64,activation='relu')))
        self.LSTMCNN.add(layers.TimeDistributed(layers.BatchNormalization()))

        self.LSTMCNN.add(layers.LSTM(hiddenSize,unroll=True))


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
        self.LSTMCNN.fit(
            dataset,
            validation_data=validSet,
            epochs=epochs
        )

    def checkGradient(self,dataset):

        for xB,yB in dataset.take(20):

            with tf.GradientTape() as tape:
                preds=self.LSTMCNN(xB,training=True)
                loss=tf.keras.losses.sparse_categorical_crossentropy(yB,preds)
                loss=tf.reduce_mean(loss)
            
            grads=tape.gradient(loss,self.LSTMCNN.trainable_variables)
            print(f"\nLoss on this batch: {loss.numpy():.6f}\n")

            # Step 3: group gradient norms by top-level layer name
            layerGradNorms = {}
            for var, grad in zip(self.LSTMCNN.trainable_variables, grads):
                layerName = var.name.split('/')[0]
                if grad is None:
                    print(f"WARNING: {var.name} has NO gradient (None)")
                    continue
                gradNorm = tf.norm(grad).numpy()
                layerGradNorms.setdefault(layerName, []).append(gradNorm)

            # Step 4: print summary, in the order layers were added to the model
            print(f"{'Layer':35s} {'Mean Grad Norm':>15s} {'Max Grad Norm':>15s} {'Min Grad Norm':>15s}")
            print("-" * 82)
            for layer, norms in layerGradNorms.items():
                print(f"{layer:35s} {np.mean(norms):>15.8f} {np.max(norms):>15.8f} {np.min(norms):>15.8f}")
