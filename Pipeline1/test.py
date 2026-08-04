from LSTMCNN import LSTMCNN
import numpy as np
import pickle
xTrain=np.load('InputFiltered.npy',mmap_mode='r') #mmap set r later for now test
yTrain=np.load('OutputTags.npy')
frameLengthPV=np.load('RealFrameLen.npy')

yTrain=np.asarray(yTrain)
frameLengthPV=np.asarray(frameLengthPV)


yTrain=yTrain.astype(np.int32)
frameLengthPV=frameLengthPV.tolist()
hiddenSize=64
numClasses=3


LCNN=LSTMCNN(xTrain,yTrain,frameLengthPV,(64,64,3),hiddenSize,numClasses)

LCNN.train(5,0.01)

#with open('Save_Model_epoch_2','rb') as f:
 #           model=pickle.load(f)
#
  #          p=model.predict(xTrain[50,:,:,:],frameLengthPV[50])

   #         print(np.argmax(p))
    #        print(yTrain[50])
        