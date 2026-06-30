import numpy as np
from ConvLayer import ConVLayer
from DesnseLayer import DenseLayer
from PoolLayer import PoolLayer


class CNN:
    
    def __init__(self,numCats):
        self.layers=[]
        self.numClasses=numCats
        

    def addlayer(self,layer):
        self.layers.append(layer)

    def forward(self,input):
        nxtInput=input
        for layer in self.layers:
            nxtInput=layer.forward(nxtInput)
            print(nxtInput.shape)
            print(nxtInput)
       

    
        return nxtInput
    def sparseCategoricalCrossEntropyLoss(self,prediction,trueLabelIndex):

        epsilon=1e-12

        #prediction array has all the probabilities
        # true label as index will select the probability that is computed for true label
        # if its high loss is low
        # if it is low loss is high (very wrong)
        return -np.log(prediction[trueLabelIndex]+epsilon)