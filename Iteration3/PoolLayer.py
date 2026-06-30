import numpy as np
from Filter import Filter

class PoolLayer:
    def __init__(self,poolFilterSize,inputShape,poolStride=2,padding=0):

        #single / returns float
        #// for forced int
        height,width,depth=inputShape
        self.inputShape=inputShape
        self.outputShape=(((height-poolFilterSize+(2*padding))//poolStride)+1,((width-poolFilterSize+(2*padding))//poolStride)+1,depth)
        self.stride=poolStride
        self.filterSize=poolFilterSize

    def forward(self,input):
        outputHeight,outputWidth,outputDepth=self.outputShape
        output=np.zeros(self.outputShape)

        for  d in range(outputDepth):
            for r in range(outputHeight):
                for c in range(outputWidth):
                    patch=input[r*self.stride:r*self.stride+self.filterSize,c*self.stride:c*self.stride+self.filterSize,d]

                    output[r,c,d]=np.max(patch)


        return output
