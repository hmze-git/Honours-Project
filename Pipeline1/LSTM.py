import numpy as np
import math

class LSTMCell:
    #hidden state (short term) and cell state(long term) share a size
    def __init__(self,inputSize,hiddenSize):

        #vector will fix itself on multiply

        scale=0.1
        #Forget gate block
        ############
        self.inputVect=np.random.randn(inputSize)*scale
        self.WeightFGHide=np.random.randn(hiddenSize,hiddenSize)*scale
        self.WeightFGInput=np.random.randn(inputSize,hiddenSize)*scale
        self.BiasFG=np.zeros(hiddenSize) # another vector
        ##################

        #InputGate block
        #################
        self.WeightInGHide=np.random.randn(hiddenSize,hiddenSize)*scale
        self.WeightInGInput=np.random.randn(inputSize,hiddenSize)*scale
        self.BiasInG=np.zeros(hiddenSize)
        #################

        #Candidate gate
        #################
        self.WeightCGHide=np.random.randn(hiddenSize,hiddenSize)*scale
        self.WeightCGInput=np.random.randn(inputSize,hiddenSize)*scale
        self.BiasCG=np.zeros(hiddenSize)
        #################



        #Output Gate
        #################
        self.WeightOutGInput=np.random.randn(inputSize,hiddenSize)*scale
        self.WeightOutGHide=np.random.randn(hiddenSize,hiddenSize)*scale
        self.BiasOutG=np.zeros(hiddenSize)
         #################


        #Weight and Bias gradients to accumulate across the timesteps

        #output gate
        #################
        self.DwXo=np.zeros((inputSize,hiddenSize))
        self.DwHo=np.zeros((hiddenSize,hiddenSize))
        self.Dbiaso=np.zeros(hiddenSize)
        #################

        #candidate gate
        #################
        self.DwXc=np.zeros((inputSize,hiddenSize))
        self.DwHc=np.zeros((hiddenSize,hiddenSize))
        self.Dbiasc=np.zeros(hiddenSize)
        #################
 
        #input gate
        #################
        self.DwXi=np.zeros((inputSize,hiddenSize))
        self.DwHi=np.zeros((hiddenSize,hiddenSize))
        self.Dbiasi=np.zeros(hiddenSize)
        #################

        #forget gate
        #################
        self.DwXf=np.zeros((inputSize,hiddenSize))
        self.DwHf=np.zeros((hiddenSize,hiddenSize))
        self.Dbiasf=np.zeros(hiddenSize)
        #################


    def forward(self,Xinput,cellHighway,hiddenHighway):

        oldHiddenHighway=hiddenHighway
        oldCellHighway=cellHighway
        # xinput@weightforget works because xinput is a vector so multiplication works without issue
        cellForgetVals=self.sigmoid((Xinput@self.WeightFGInput + hiddenHighway@self.WeightFGHide) + self.BiasFG)

        cellHighway=cellHighway*cellForgetVals


        #input/candidate gates

        forgetPortion=self.sigmoid((Xinput@self.WeightInGInput +oldHiddenHighway@self.WeightInGHide)+self.BiasInG)

        candidatePortion=self.tanh((Xinput@self.WeightCGInput+oldHiddenHighway@self.WeightCGHide)+self.BiasCG)

        finalVal=forgetPortion*candidatePortion

        cellHighway=cellHighway+finalVal


        #outputGate
        
        OutGPortion=self.sigmoid((Xinput@self.WeightOutGInput +oldHiddenHighway@self.WeightOutGHide)+self.BiasOutG)

        potentialShortMem=self.tanh((cellHighway))


        hiddenHighway=OutGPortion*potentialShortMem

        cacheObject={
            'x':Xinput,
            'hOld':oldHiddenHighway,
            'cOld':oldCellHighway,
            'forgetG':cellForgetVals,
            'inputG':forgetPortion,
            'candG':candidatePortion,
            'outG':OutGPortion,
            'hnew':hiddenHighway,
            'cnew':cellHighway

        }

        return cacheObject
        
        
    def backward(self,cachedValues,DH,DHCell):

        #loss function to be used is Sparse Categorical Cross Entropy Loss

        #output gate derivative
        DO=DH*self.tanh(cachedValues.get('cnew'))

        #cekk state gradient loss
        DCell=DHCell+DH*cachedValues.get('outG')*(1-(self.tanh(cachedValues.get('cnew')))**2)


        DI=DCell*cachedValues.get('candG')

        DCG=DCell*cachedValues.get('inputG')
        

        DF=DCell*cachedValues.get('cOld')

 

        DFGpre=DF*cachedValues.get('forgetG')*(1-cachedValues.get('forgetG'))
        DIGpre=DI*cachedValues.get('inputG')*(1-cachedValues.get('inputG'))
        DCGpre=DCG*(1-(cachedValues.get('candG')**2))
        DOgpre=DO*cachedValues.get('outG')*(1-cachedValues.get('outG'))
        #DHCell to be passed into "previous"
        #check this if gradient is wonky
        DCellPrevT=DCell*cachedValues.get('forgetG')

        #change in loss by changing the hidden state
        #since hidden is affected by all weights from all the gates sum them up
        #note that use preactivation so first apply derivative of the activation functions for each gate
        DHPrevT=self.WeightFGHide@DFGpre +self.WeightCGHide@DCGpre+self.WeightInGHide@DIGpre+self.WeightOutGHide@DOgpre

    # weight Gradients
    #outputGate weight grad

        
        self.DwXo+=np.outer(cachedValues.get('x'),DO*cachedValues.get('outG')*(1-cachedValues.get('outG')))
        self.DwHo+=np.outer(cachedValues.get('hOld'),DO*cachedValues.get('outG')*(1-cachedValues.get('outG')))
        self.Dbiaso+=DO*cachedValues.get('outG')*(1-cachedValues.get('outG'))

    #candidate weight grad
        self.DwXc+=np.outer(cachedValues.get('x'),DCG*(1-(cachedValues.get('candG')**2)))
        self.DwHc+=np.outer(cachedValues.get('hOld'),DCG*(1-(cachedValues.get('candG')**2)))
        self.Dbiasc+=DCG*(1-(cachedValues.get('candG')**2))

   #input weight grad
        self.DwXi+=np.outer(cachedValues.get('x'),DI*cachedValues.get('inputG')*(1-cachedValues.get('inputG')))
        self.DwHi+=np.outer(cachedValues.get('hOld'),DI*cachedValues.get('inputG')*(1-cachedValues.get('inputG')))
        self.Dbiasi+=DI*cachedValues.get('inputG')*(1-cachedValues.get('inputG'))

   #forget Gate weight grad
        self.DwXf+=np.outer(cachedValues.get('x'),DF*cachedValues.get('forgetG')*(1-cachedValues.get('forgetG')))
        self.DwHf+=np.outer(cachedValues.get('hOld'),DF*cachedValues.get('forgetG')*(1-cachedValues.get('forgetG')))
        self.Dbiasf+=DF*cachedValues.get('forgetG')*(1-cachedValues.get('forgetG'))
        return DCellPrevT,DHPrevT


    def update(self,learnRate):
        
        self.WeightOutGHide-=learnRate*self.DwHo
        self.WeightOutGInput-=learnRate*self.DwXo
        self.BiasOutG-=learnRate*self.Dbiaso


        self.WeightCGHide-=learnRate*self.DwHc
        self.WeightCGInput-=learnRate*self.DwXc
        self.BiasCG-=learnRate*self.Dbiasc


        self.WeightInGHide-=learnRate*self.DwHi
        self.WeightInGInput-=learnRate*self.DwXi
        self.BiasInG-=learnRate*self.Dbiasi


        self.WeightFGHide-=learnRate*self.DwHf
        self.WeightFGInput-=learnRate*self.DwXf
        self.BiasFG-=learnRate*self.Dbiasf


    def zeroDeriGrad(self):
          #output gate
        #################
        self.DwXo[:]=0
        self.DwHo[:]=0
        self.Dbiaso[:]=0
        #################

        #candidate gate
        #################
        self.DwXc[:]=0
        self.DwHc[:]=0
        self.Dbiasc[:]=0
        #################
 
        #input gate
        #################
        self.DwXi[:]=0
        self.DwHi[:]=0
        self.Dbiasi[:]=0
        #################

        #forget gate
        #################
        self.DwXf[:]=0
        self.DwHf[:]=0
        self.Dbiasf[:]=0

    def sigmoid (self,z):
        return 1/(1+np.exp(-z))
    
    def tanh (self,z):
        return (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))
        