from CNN import CNN
from LSTM import LSTMCell
from ConvLayer import ConVLayer
from PoolLayer import PoolLayer
from DesnseLayer import DenseLayer
from LSTMDesnseLayer import LSTMDenseLayer
import cupy as np
import math
import time

spatialExtractor=CNN(3)




xTrain=np.load('InputFiltered.npy')
yTrain=np.load('OutputTags.npy')
frameLengthPV=np.load('RealFrameLen.npy')

xTrain=np.asarray(xTrain)
yTrain=np.asarray(yTrain)
frameLengthPV=np.asarray(frameLengthPV)

xTrain=xTrain.astype(np.float64)
yTrain=yTrain.astype(np.int32)
frameLengthPV=frameLengthPV.tolist()




Conv1=ConVLayer(8,2,(64, 64,3))
Conv2=ConVLayer(8,2,Conv1.outputShape)
#Pool1=PoolLayer(2,Conv2.outputShape)


Conv3=ConVLayer(16,2,Conv2.outputShape)
Conv4=ConVLayer(16,2,Conv3.outputShape)
#Pool2=PoolLayer(2,Conv4.outputShape)


Conv5=ConVLayer(32,2,Conv4.outputShape)
Conv6=ConVLayer(32,2,Conv5.outputShape)
#Pool3=PoolLayer(2,Conv6.outputShape)

D1=DenseLayer(128,math.prod(Conv6.outputShape),None,'ReLu',None)
D2=DenseLayer(64,D1.numOutNeurons,None,'ReLu',None)


spatialExtractor.addlayer(Conv1)
spatialExtractor.addlayer(Conv2)
#spatialExtractor.addlayer(Pool1)
spatialExtractor.addlayer(Conv3)
spatialExtractor.addlayer(Conv4)
#spatialExtractor.addlayer(Pool2)
spatialExtractor.addlayer(Conv5)
spatialExtractor.addlayer(Conv6)
#spatialExtractor.addlayer(Pool3)
spatialExtractor.addlayer(D1)
spatialExtractor.addlayer(D2)

hiddenSize=64
numClasses=3

temporal=LSTMCell(D2.numOutNeurons,hiddenSize)

classifier1=LSTMDenseLayer(32,hiddenSize,None,'ReLu')
classifier2=LSTMDenseLayer(numClasses,classifier1.numOutNeurons,None,'SoftMax',None)


def train(xTrain,yTrain,epochs,LR):

    numVideos,numFrames,numRows,numCols,depth=xTrain.shape

   
    for e in range(epochs):
        predicions=[]
        totalLoss=0
        startTime=time.time()
        for v in range(2):
            startTimeVid=time.time()
            DHPrevT=np.zeros(hiddenSize)
            DCellPrevT=np.zeros(hiddenSize)

            H=np.zeros(hiddenSize,)
            C=np.zeros(hiddenSize,)
            cArray=[]
            for f in range(frameLengthPV[v]):
                frame=xTrain[v,f,::]

                #startCNN=time.time()
                lstmInput=spatialExtractor.forward(frame)
                #endCNN=time.time()

                #print(f"time for one CNN run {endCNN-startCNN}")


                #start=time.time()
                lstmCache=temporal.forward(lstmInput,C,H)
                #end=time.time()

               #print(f'Time for one LSTM run {end-start}')

                H=lstmCache.get('hnew')
                C=lstmCache.get('cnew')
              
                cArray.append(lstmCache)

            secondLastOutput=classifier1.forward(H)
            prediction=classifier2.forward(secondLastOutput)
            predicions.append(prediction)
            totalLoss+=sparseCategoricalCrossEntropyLoss(prediction,yTrain[v])
            
            DHC2=classifier2.backward(prediction,yTrain[v])
            DHC1=classifier1.backward(DHC2,yTrain[v]) #RELU so doesnt need the label

            for f in reversed(range(frameLengthPV[v])):
        
                if f==numFrames-1:
                    DHTOT=DHC1+DHPrevT
                else:
                    DHTOT=DHPrevT

                DCellPrevT,DHPrevT,DXt= temporal.backward(cArray[f],DHTOT,DCellPrevT)
                spatialExtractor.backward(f,DXt,None)# CNN never final layer so doesnt need it

            
            temporal.update(LR)
            temporal.zeroDeriGrad()
            spatialExtractor.update(LR)
            spatialExtractor.resetCacheWeights()
            nd=time.time()
            print(f"One vid done in {nd-startTimeVid}")
        endTimeEpoch=time.time()
        print(f"end time 1 epch 2 vids {endTimeEpoch-startTime}")
        print(f"Epoch {e}: avg loss = {totalLoss/numVideos:.4f}")
        correct = sum(np.argmax(pred) == true for pred, true in zip(predicions, yTrain))
        print(f"Accuracy: {correct/numVideos*100:.1f}%")  



            

        
def sparseCategoricalCrossEntropyLoss(prediction,trueLabelIndex):

        epsilon=1e-12

        #prediction array has all the probabilities
        # true label as index will select the probability that is computed for true label
        # if its high loss is low
        # if it is low loss is high (very wrong)
        return -np.log(prediction[trueLabelIndex]+epsilon)

train(xTrain,yTrain,5,0.1)