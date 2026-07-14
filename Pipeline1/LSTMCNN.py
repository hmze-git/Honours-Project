from CNN import CNN
from LSTM import LSTMCell
from ConvLayer import ConVLayer
from PoolLayer import PoolLayer
from DesnseLayer import DenseLayer
from LSTMDesnseLayer import LSTMDenseLayer
import cupy as np
import math
import time
import pickle





class LSTMCNN:
    def __init__(self,xData,yData,FrameLen,inputShape,hiddenSize,numClasses,):
            self.spatialExtractor=CNN(numClasses)

            self.xTrain=xData
            self.yTrain=yData
            self.frameLengthPV=FrameLen

            self.Conv1=ConVLayer(8,2,inputShape)
            self.Conv2=ConVLayer(8,2,self.Conv1.outputShape)
            self.Pool1=PoolLayer(2,self.Conv2.outputShape)


            self.Conv3=ConVLayer(16,2,self.Pool1.outputShape)
            self.Conv4=ConVLayer(16,2,self.Conv3.outputShape)
            self.Pool2=PoolLayer(2,self.Conv4.outputShape)


            self.Conv5=ConVLayer(32,2,self.Pool2.outputShape)
            self.Conv6=ConVLayer(32,2,self.Conv5.outputShape)
            self.Pool3=PoolLayer(2,self.Conv6.outputShape)

            self.D1=DenseLayer(128,math.prod(self.Pool3.outputShape),None,'ReLu',None)
            self.D2=DenseLayer(64,self.D1.numOutNeurons,None,'ReLu',None)
            self.spatialExtractor.addlayer(self.Conv1)
            self.spatialExtractor.addlayer(self.Conv2)
            self.spatialExtractor.addlayer(self.Pool1)
            self.spatialExtractor.addlayer(self.Conv3)
            self.spatialExtractor.addlayer(self.Conv4)
            self.spatialExtractor.addlayer(self.Pool2)
            self.spatialExtractor.addlayer(self.Conv5)
            self.spatialExtractor.addlayer(self.Conv6)
            self.spatialExtractor.addlayer(self.Pool3)
            self.spatialExtractor.addlayer(self.D1)
            self.spatialExtractor.addlayer(self.D2)

            self.hiddenSize=hiddenSize
            self.numClasses=numClasses

            self.temporal=LSTMCell(self.D2.numOutNeurons,hiddenSize)

            self.classifier1=LSTMDenseLayer(32,hiddenSize,None,'ReLu')
            self.classifier2=LSTMDenseLayer(numClasses,self.classifier1.numOutNeurons,None,'SoftMax',None)
            self.accuracy=[]
            self.loss=[]
            self.numEpochs=0


    def train(self,epochs,LR):

        numVideos,numFrames,numRows,numCols,depth=self.xTrain.shape
        self.numEpochs=epochs
    
        for e in range(epochs):
            predicions=[]
            totalLoss=0
            startTime=time.time()
            for v in range(numVideos):

                DHPrevT=np.zeros(self.hiddenSize)
                DCellPrevT=np.zeros(self.hiddenSize)

                H=np.zeros(self.hiddenSize,)
                C=np.zeros(self.hiddenSize,)
                cArray=[]
                for f in range(self.frameLengthPV[v]):
                    frame=self.xTrain[v,f,:,:,:]

                    #startCNN=time.time()
                    lstmInput=self.spatialExtractor.forward(frame)
                    #endCNN=time.time()

                    #print(f"time for one CNN run {endCNN-startCNN}")


                    #start=time.time()
                    lstmCache=self.temporal.forward(lstmInput,C,H)
                    #end=time.time()

                #print(f'Time for one LSTM run {end-start}')

                    H=lstmCache.get('hnew')
                    C=lstmCache.get('cnew')
                
                    cArray.append(lstmCache)

                secondLastOutput=self.classifier1.forward(H)
                prediction=self.classifier2.forward(secondLastOutput)
                predicions.append(prediction)
                totalLoss+=self.sparseCategoricalCrossEntropyLoss(prediction,self.yTrain[v])
                
                DHC2=self.classifier2.backward(prediction,self.yTrain[v])
                DHC1=self.classifier1.backward(DHC2,self.yTrain[v]) #RELU so doesnt need the label

                for f in reversed(range(self.frameLengthPV[v])):
            
                    if f==self.frameLengthPV[v]-1:
                        DHTOT=DHC1+DHPrevT
                    else:
                        DHTOT=DHPrevT

                    DCellPrevT,DHPrevT,DXt= self.temporal.backward(cArray[f],DHTOT,DCellPrevT)
                    self.spatialExtractor.backward(f,DXt,None)# CNN never final layer so doesnt need it

                self.classifier2.updateParameters(LR)
                self.classifier1.updateParameters(LR)
                self.temporal.update(LR)
                self.spatialExtractor.update(LR)
                self.classifier2.clearDerivativeCache()
                self.classifier1.clearDerivativeCache()
                self.temporal.zeroDeriGrad()
                self.spatialExtractor.resetCacheWeights()
                
            endTimeEpoch=time.time()
            print(f"End Time epoch  {e} for {numVideos} vids {endTimeEpoch-startTime}")
            print(f"Epoch {e}: avg loss = {totalLoss/numVideos}")
            numCorrect=0
            for p,trueLabel in zip(predicions,self.yTrain):
                if np.argmax(p)==trueLabel:
                    numCorrect+=1
            self.accuracy.append(numCorrect)
            print(f"Accuracy: {(numCorrect/numVideos)*100}%") 
            self.saveModel(f'Save_Model_epoch_{e}') 


    def predict(self,Xinput,frameLength):
                H=np.zeros(self.hiddenSize,)
                C=np.zeros(self.hiddenSize,)
                cArray=[]
                for f in range(frameLength):
                    frame=Xinput[f,::]

                    #startCNN=time.time()
                    lstmInput=self.spatialExtractor.forward(frame)

                    lstmCache=self.temporal.forward(lstmInput,C,H)

                    H=lstmCache.get('hnew')
                    C=lstmCache.get('cnew')
                
                    cArray.append(lstmCache)

                secondLastOutput=self.classifier1.forward(H)
                prediction=self.classifier2.forward(secondLastOutput)
                return prediction

    def saveModel(self,path):
        with open(path,'wb') as f:
            pickle.dump(self,f)
        

            
    def sparseCategoricalCrossEntropyLoss(self,prediction,trueLabelIndex):

            epsilon=1e-12

            #prediction array has all the probabilities
            # true label as index will select the probability that is computed for true label
            # if its high loss is low
            # if it is low loss is high (very wrong)
            return -np.log(prediction[trueLabelIndex]+epsilon)

