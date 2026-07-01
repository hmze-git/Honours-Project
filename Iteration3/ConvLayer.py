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
        self.layerInput=None
        self.finalZ=None
        for _ in range(numFilters):
            self.filters.append(Filter(filterSize,filterSize,depth))
        self.update=True
    

    def forward(self,input):
        
        self.layerInput=input
        outputHeight,outputWidth,outputDepth=self.outputShape
        output=np.zeros(self.outputShape)

        for fd,filter in enumerate(self.filters):
            filterWidth=filter.width
            filterHeight=filter.height
            for r in range(outputHeight):
                for c in range(outputWidth):
                    patch=input[r*self.stride:r*self.stride+filterHeight,c*self.stride:c*self.stride+filterWidth,:]
                    output[r,c,fd]=filter.applyFilter(patch)

        self.finalZ=output
        output=self.ReLu(output)
        return output
    def backward(self,backwardsInput):

        
        #dl/dz chain rules to the backwards input * doutput/dz 
        Dz=backwardsInput * self.derivative_ReLU(self.finalZ)

        Dinput=np.zeros(self.inputShape)

        outputHeight,outputWidth,outputDepth=self.outputShape
        for fd,filter in enumerate(self.filters):
            
            #This is the updated weight matrix for each specefic filter
            filterWidth=filter.width
            filterHeight=filter.height
            dWeights=np.zeros_like(filter.weights)
            dBias=0.0
            for r in range(outputHeight):
                for c in range(outputWidth):
                    strideR=r*self.stride
                    strideC=c*self.stride
                    #cut out the exact patch that the filter was applied on
                    #filters go on all channels/depth at once so cut the full slice
                    patch=self.layerInput[strideR:strideR+filterHeight,strideC:strideC+filterWidth,:]

                    #change in loss due to changing the z value
                    # change in z affected by weight and bias so chain rule says that u apply this to those
                    gradient=Dz[r,c,fd]

                    #Dl/Dkmn=Input matrix cross correlated with Dl/DY
                    dWeights+=patch*gradient

                    dBias+=gradient
                    Dinput[strideR:strideR+filterHeight,strideC:strideC+filterWidth,:]+=filter.weights*gradient

            filter.dWeights=dWeights
            filter.dBias=dBias

        
        return Dinput

    def ReLu(self,arr):
        return np.maximum(arr,0)

    # when reul less than 0 deriv constant at 0 when not its 1
    def derivative_ReLU(self,Z):
        return (Z>0).astype(int)

    def updateParameters(self,learnRate):

        for filt in self.filters:
            filt.weights-=learnRate*filt.dWeights
            filt.bias-=learnRate*filt.dBias




