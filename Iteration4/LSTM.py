import numpy as np


class LSTMCell:
    #hidden state (short term) and cell state(long term) share a size
    def __init__(self,inputSize,hiddenSize):

        #vector will fix itself on multiply


        #Forget gate block
        ############
        self.inputVect=np.zeros(inputSize)
        self.WeightF=np.zeros((hiddenSize,hiddenSize))
        self.WeightFInput=np.zeros((inputSize,hiddenSize))
        self.BiasF=np.zeros(hiddenSize) # another vector
        ##################

        