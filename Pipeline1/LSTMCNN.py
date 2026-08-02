from CNN import CNN
from LSTM import LSTMCell
from ConvLayer import ConVLayer
from PoolLayer import PoolLayer
from DesnseLayer import DenseLayer
from LSTMDesnseLayer import LSTMDenseLayer
import cupy as cp
import numpy as np
import math
import time
import pickle
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay





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
            self.valAccuracy=[]
            self.loss=[]
            self.valLoss=[]
            self.trainTimes=[]
            self.numEpochs=0


    def train(self,epochs,LR):

        numVideos,numFrames,numRows,numCols,depth=self.xTrain.shape
        self.numEpochs=epochs

        numVideos=1050
        maxCount=5 #only do 150 videos so the permuations can be on full set that way we randomly get entries and dont just take first batch
        for e in range(epochs):
            
            totalLoss=0
            startTime=time.time()
            
          

            shuffledIndex=np.random.permutation(numVideos)
            #stick top 150 videos for now because otherwise it takes too long with 224x224
            counter=0
            visitedIndices=[]
            for v in shuffledIndex:
                if counter==maxCount:
                     break
                v=int(v)
                visitedIndices.append(v)

                DHPrevT=cp.zeros(self.hiddenSize)
                DCellPrevT=cp.zeros(self.hiddenSize)

                H=cp.zeros(self.hiddenSize,)
                C=cp.zeros(self.hiddenSize,)
                cArray=[]
                for f in range(self.frameLengthPV[v]):
                    frame=cp.asarray(self.xTrain[v,f,:,:,:])
                    frame=frame.astype(cp.float64)

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

                #increment the count
                counter+=1
                
          


            numCorrect=0
            #this way compare only what we want 
            for v in visitedIndices:

                prediction=self.predict(self.xTrain[v],self.frameLengthPV[v])
                totalLoss+=self.sparseCategoricalCrossEntropyLoss(prediction,self.yTrain[v])

                if cp.argmax(prediction)==self.yTrain[v]:
                        numCorrect+=1

            print(f"Epoch {e}: avg loss ={totalLoss/maxCount}")
            print(f"Accuracy: {(numCorrect/maxCount)*100}%") 
            self.accuracy.append(numCorrect/maxCount)
            self.loss.append(float(totalLoss/maxCount))

            self.validation(self.xTrain,self.yTrain,self.frameLengthPV,visitedIndices,e)
            endTimeEpoch=time.time()


            timeTake=endTimeEpoch-startTime
            self.trainTimes.append(timeTake)
            print(f"End Time epoch  {e} for {maxCount} vids {timeTake}")     
         
           
            #self.saveModel(f'Save_Model_epoch_{e}') 
        self.plotCurves()

    def predict(self,Xinput,frameLength):
                H=cp.zeros(self.hiddenSize,)
                C=cp.zeros(self.hiddenSize,)
                cArray=[]
                for f in range(frameLength):
                    frame=cp.asarray(Xinput[f,::])
                    frame=frame.astype(cp.float64)

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

    def validation(self,xValid,yValid,xVidLength,visitedVideos,epoch):
        numVideos,numFrames,numRows,numCols,depth=xValid.shape

        count=0
        shuffledIndex=np.random.permutation(numVideos)
        numCorrect=0
        totalLoss=0
        for v in shuffledIndex:
            v=int(v)
            if count==len(visitedVideos):
                break

            if v in visitedVideos:
                 continue
            

            prediction=self.predict(xValid[v],xVidLength[v])
            totalLoss+=self.sparseCategoricalCrossEntropyLoss(prediction,yValid[v])
            if cp.argmax(prediction)==yValid[v]:
                numCorrect+=1
            count+=1
        print(f"Epoch {epoch}: avg loss ={totalLoss/len(visitedVideos)}")
        print(f"Validation Accuracy: {(numCorrect/len(visitedVideos))*100}%") 
        self.valAccuracy.append(numCorrect/len(visitedVideos))
        self.valLoss.append(float(totalLoss/len(visitedVideos)))
        

        
        
    def plotCurves(self):
        plt.plot(self.accuracy)
        plt.plot(self.valAccuracy)
        plt.title('Model Accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train','val'])
        plt.savefig('NormVsValACc.png')
        plt.clf()

        plt.plot(self.loss)
        plt.plot(self.valLoss)

        plt.title('Model Loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train','val'])
        plt.savefig('NormVsValLoss.png')

        #hardcoded but should be full dataset size sort out later
        shuffledIndex=np.random.permutation(1050)


        counter=0
        predLabel=[]
        trueLabels=[]
        for v in shuffledIndex:
            v=int(v)

            if counter==5: #temp hardcoded val fix later 
                 break

            p=self.predict(self.xTrain[v],self.frameLengthPV[v])
            predLabel.append(int(cp.argmax(p)))
            trueLabels.append(self.yTrain[v]) #passed in as numpy array so no need to convert it 
            counter+=1
     

        cm=confusion_matrix(trueLabels,predLabel)
        disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=[0,1,2])
        disp.plot(cmap='Blues')
        plt.savefig('confMatrix.png')
        plt.show()

            
    def sparseCategoricalCrossEntropyLoss(self,prediction,trueLabelIndex):

            epsilon=1e-12

            #prediction array has all the probabilities
            # true label as index will select the probability that is computed for true label
            # if its high loss is low
            # if it is low loss is high (very wrong)
            return -cp.log(prediction[trueLabelIndex]+epsilon)

