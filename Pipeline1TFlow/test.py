from CNNLSTM import TFLSTMCNN
import tensorflow as tf
from tensorflow.keras import layers,models
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
xTrain=np.load('InputFiltered.npy',mmap_mode='r')
yTrain=np.load('OutputTags.npy')

xValid=np.load('validationFiltered.npy',mmap_mode='r')
yValid=np.load('validationTags.npy')

frameLengthPV=np.load('RealFrameLen.npy')

#xTrain=np.asarray(xTrain)
#yTrain=np.asarray(yTrain)
#frameLengthPV=np.asarray(frameLengthPV)

frameLengthPV=frameLengthPV.tolist()
hiddenSize=64
numClasses=3



inShpae=xTrain.shape[1:] #remvove axis 0 from shape it only wants frame shape
LCNN=TFLSTMCNN(inShpae,hiddenSize,numClasses)


def generator():
    for x in range(len(xTrain)):
        yield xTrain[x],yTrain[x]


def validationGenerator():
    for x in range(len(xValid)):
        yield xValid[x],yValid[x]

dataset=tf.data.Dataset.from_generator(
    generator,
    output_signature=(
        tf.TensorSpec(shape=xTrain.shape[1:],dtype=tf.float32),
        tf.TensorSpec(shape=(),dtype=tf.int32)
    )
  )

validationSet=tf.data.Dataset.from_generator(
    validationGenerator,
    output_signature=(
         tf.TensorSpec(shape=xValid.shape[1:],dtype=tf.float32),
        tf.TensorSpec(shape=(),dtype=tf.int32)
    )
)


dataset = dataset.shuffle(buffer_size=1050)
dataset = dataset.batch(batch_size=4)
dataset = dataset.prefetch(tf.data.AUTOTUNE)



validationSet = validationSet.shuffle(buffer_size=225)
validationSet = validationSet.batch(batch_size=4)
validationSet = validationSet.prefetch(tf.data.AUTOTUNE)





LCNN.train(150,dataset,validationSet)


preds=LCNN.LSTMCNN.predict(validationSet)

predLabels=np.argmax(preds,axis=1)
trueLabels=np.concatenate([y.numpy() for x,y in validationSet],axis=0)

cm=confusion_matrix(predLabels,trueLabels)

disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=[0,1,2])
disp.plot(cmap='Blues')
plt.savefig('confMatrix.png')
plt.show()


#with open('Save_Model_epoch_2','rb') as f:
 #           model=pickle.load(f)
#
  #          p=model.predict(xTrain[50,:,:,:],frameLengthPV[50])

   #         print(np.argmax(p))
    #        print(yTrain[50])
        