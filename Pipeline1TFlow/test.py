from CNNLSTM import TFLSTMCNN
import tensorflow as tf
from tensorflow.keras import layers,models
import numpy as np
import pickle
xTrain=np.load('InputFiltered.npy',mmap_mode='r')
yTrain=np.load('OutputTags.npy')
frameLengthPV=np.load('RealFrameLen.npy')

#xTrain=np.asarray(xTrain)
#yTrain=np.asarray(yTrain)
#frameLengthPV=np.asarray(frameLengthPV)

xTrain=xTrain.astype(np.float64)
yTrain=yTrain.astype(np.int32)
frameLengthPV=frameLengthPV.tolist()
hiddenSize=64
numClasses=3


inShpae=xTrain.shape[1:] #remvove axis 0 from shape it only wants frame shape
LCNN=TFLSTMCNN(inShpae,hiddenSize,numClasses)


def generator():
    for x in range(len(xTrain)):
        yield xTrain[x],yTrain[x]


dataset=tf.data.Dataset.from_generator(
    generator,
    output_signature=(
        tf.TensorSpec(shape=xTrain.shape[1:],dtype=tf.float32),
        tf.TensorSpec(shape=(),dtype=tf.int32)
    )
  )

dataset=dataset.batch(1)


print(dataset)

LCNN.train(5,dataset)


#with open('Save_Model_epoch_2','rb') as f:
 #           model=pickle.load(f)
#
  #          p=model.predict(xTrain[50,:,:,:],frameLengthPV[50])

   #         print(np.argmax(p))
    #        print(yTrain[50])
        