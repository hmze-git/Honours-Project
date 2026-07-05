import numpy as np


class LSTMCell:
    #hidden state (short term) and cell state(long term) share a size
    def __init__(self,inputSize,hiddenSize):

        #vector will fix itself on multiply


        #Forget gate block
        ############
        self.inputVect=np.zeros(inputSize)
        self.WeightFGHide=np.zeros((hiddenSize,hiddenSize))
        self.WeightFGInput=np.zeros((inputSize,hiddenSize))
        self.BiasFG=np.zeros(hiddenSize) # another vector
        ##################

        #InputGate block
        #################
        self.WeightInGHide=np.zeros((hiddenSize,hiddenSize))
        self.WeightInGInput=np.zeros((inputSize,hiddenSize))
        self.BiasInG=np.zeros(hiddenSize)
        #################

        #Candidate gate
        #################
        self.WeightCGHide=np.zeros((hiddenSize,hiddenSize))
        self.WeightCGInput=np.zeros((inputSize,hiddenSize))
        self.BiasCG=np.zeros(hiddenSize)
        #################



        #Output Gate
        #################
        self.WeightOutGInput=np.zeros((inputSize,hiddenSize))
        self.WeightOutGHide=np.zeros((hiddenSize,hiddenSize))
        self.BiasOutG=np.zeros(hiddenSize)

    def forward(self,Xinput,cellHighway,hiddenHighway):

        print(Xinput.shape)    
        print(hiddenHighway.shape)
        print(cellHighway.shape)
        # xinput@weightforget works because xinput is a vector so multiplication works without issue
        cellForgetVals=self.sigmoid((Xinput@self.WeightFGInput + hiddenHighway@self.WeightFGHide) + self.BiasFG)

        cellHighway=cellHighway*cellForgetVals


        #input/candidate gates

        forgetPortion=self.sigmoid((Xinput@self.WeightInGInput +hiddenHighway@self.WeightInGHide)+self.BiasInG)

        candidatePortion=self.tanh((Xinput@self.WeightCGInput+hiddenHighway@self.WeightCGHide)+self.BiasCG)

        finalVal=forgetPortion*candidatePortion

        cellHighway=cellHighway+finalVal


        #outputGate
        
        forgetOutPortion=self.sigmoid((Xinput@self.WeightOutGInput +hiddenHighway@self.WeightOutGHide)+self.BiasOutG)

        potentialShortMem=self.tanh((cellHighway))


        hiddenHighway=forgetOutPortion*potentialShortMem

        return hiddenHighway,cellHighway
        
        

        
    def sigmoid (self,z):
        return 1/(1+np.exp(-z))
    
    def tanh (self,z):
        return (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))
        