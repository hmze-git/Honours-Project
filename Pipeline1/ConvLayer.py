import cupy as np
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

        self.filterSize=filterSize
        
        for _ in range(numFilters):
            self.filters.append(Filter(filterSize,filterSize,depth))
        self.update=True

        self.cachedInput=[]
        self.cachedIm2Col=[]
        self.cachedZVal=[]
    

    def forward(self,input):
        
        self.cachedInput.append(input)
        outputHeight,outputWidth,outputDepth=self.outputShape
        output=np.zeros(self.outputShape)
        #think of X as input just reshaped
        X=self.convertIm2Col(input)
        
        self.cachedIm2Col.append(X)

        #stack the weights as row vectors because the columns of X are patches
        WeightVector=np.stack([f.weights.flatten() for f in self.filters],axis=0)
        biases=np.array([f.bias for f in self.filters]).reshape(-1,1)


        outputF=WeightVector@X + biases
       
        output=outputF.T.reshape(outputHeight,outputWidth,outputDepth)

        self.cachedZVal.append(output)
        output=self.ReLu(output)
        return output
    def backward(self,backwardsInput,frameIndex):

        
        #dl/dz chain rules to the backwards input * doutput/dz 

        #Dz is of the form of the input passed from layer to layer
        Dz=backwardsInput * self.derivative_ReLU(self.cachedZVal[frameIndex])
        #undo reshape from end of forward so we can go in reverse
        #do so it can work with the im2col version of X later on 

        #REMEMBER AFTER THE @ STEP IN FORWARD YOU GET RHE ROWsXCOLs where rows are the filters and cols are the patches this gives that 
        DzFlattened=Dz.reshape(-1,self.numKernels).T

        X=self.cachedIm2Col[frameIndex]

        #Weights stacked along the rows 
        #R1=W1's weight parameters
        #C1-Cn=W1's adjustable parameters
        Weights=np.stack([f.weights.flatten() for f in self.filters],axis=0)

        #Dl/Dkmn=Input matrix cross correlated with Dl/DY
        dWeights=DzFlattened@X.T


        dBias=DzFlattened.sum(axis=1,keepdims=True)

        inputHeight, inputWidth, inputDepth = self.inputShape


        #accumulate filter gradient across all frames for an input
        for i, filt in enumerate(self.filters):
            filt.dWeights += dWeights[i].reshape(self.filterSize, self.filterSize, inputDepth)
            filt.dBias += dBias[i, 0]

       

        dX=Weights.T@DzFlattened

        
        Dinput=self.col2Im(dX)

        
        return Dinput


    
    def col2Im(self,DX):
        inputHeight,inputWidth,inputDepth=self.inputShape
        outputHeight,outputWidth,outputDepth=self.outputShape

        Dinput=np.zeros(self.inputShape)
        x=0

        for r in range(outputHeight):
            for c in range(outputWidth):
                strideR=r*self.stride
                strideC=c*self.stride
                

                gradientFilter=DX[:,x].reshape(self.filterSize,self.filterSize,inputDepth)
                

                Dinput[strideR:strideR+self.filterSize,strideC:strideC+self.filterSize,:]+=gradientFilter
                x+=1
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

    def convertIm2Col(self,input):
            outputHeight,outputWidth,outputDepth=self.outputShape
            inputHeight,inputWidth,inputDepth=self.inputShape
            filterWidth=self.filterSize
            filterHeight=self.filterSize
            #first part gives number of entries 
            #second gives the number fo patches
            matCol=np.zeros((filterHeight*filterWidth*inputDepth,outputWidth*outputHeight))
            x=0
            for r in range(outputHeight):
                for c in range(outputWidth):
                    patch=input[r*self.stride:r*self.stride+filterHeight,c*self.stride:c*self.stride+filterWidth,:]

                    #take each and every row and for each row 
                    matCol[:,x]=patch.flatten()
                    x+=1
            return matCol
                    
    def clearCache(self):
        self.cachedInput=[]
        self.cachedIm2Col=[]
        self.cachedZVal=[]

        for filt in self.filters:
            filt.clearGradient()
