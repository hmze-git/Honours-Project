import numpy as np
from Filter import Filter

class ConVLayer:
    def __init__(self,numFilters,filterSize,inputShape,stride=1,padding=0):

        #single / returns float
        #// for forced int
        height,width,depth=inputShape
        self.numKernels=numFilters
        self.inputShape=inputShape
        self.outputShape=(((height-filterSize+(2*padding))//stride)+1,((width-filterSize+(2*padding))//stride)+1,numFilters)
        self.stride=stride
        self.filters=[]
        for _ in range(numFilters):
            self.filters.append(Filter(filterSize,filterSize,depth))

    

    def forward(self,input):
        
        outputHeight,outputWidth,outputDepth=self.outputShape
        output=np.zeros(self.outputShape)

        for fd,filter in enumerate(self.filters):
            filterWidth=filter.width
            filterHeight=filter.height
            for r in range(outputHeight):
                for c in range(outputWidth):
                    patch=input[r:r*self.stride+filterHeight,c:c*self.stride+filterWidth,:]
                    output[r,c,fd]=filter.applyFilter(patch)

        output=self.ReLu(output)
        return output
    def backward(self,backwardsInput):

        workingArr=backwardsInput
        #layer ends with ReLU so start by undoing that 
        workingArr=self.derivative_ReLU(backwardsInput)
        
        
        pass

    

    def ReLu(self,arr):
        return np.maximum(arr,0)

    # when reul less than 0 deriv constant at 0 when not its 1
    def derivative_ReLU(self,Z):
        return (Z>0).astype(int)