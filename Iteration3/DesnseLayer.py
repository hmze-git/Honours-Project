import cupy as np
from Filter import Filter

class DenseLayer:
    def __init__(self,numOutputNeurons,inputSize,priorBias,activationFunction='ReLu',priorWeights=None):

        self.numOutNeurons=numOutputNeurons
        self.actFunction=activationFunction
        self.Weights=None
        if priorWeights is None:
            self.Weights=np.random.randn(inputSize,self.numOutNeurons)*0.01
            self.bias=0.0

        else:
            self.Weights=priorWeights
            self.bias=priorBias
        self.Activation=None
        self.ZVal=None
        self.flatInput=None
        self.orginalShape=None

        #gradients
        self.Dw=None
        self.Db=None
        self.FinalLayer=False
        

    def flattenInput(self,input):

        #converts to 1d array
        # store orginal shape for reversing flattening later
        self.orginalShape=input.shape
        return input.reshape(-1)

    def forward(self,input):
        
        flattenedInput=self.flattenInput(input)
        self.flatInput=flattenedInput
        Z=self.Weights.T@flattenedInput +self.bias

        self.ZVal=Z
        if self.actFunction=='ReLu':
            self.Activation= self.ReLu(Z)
            return self.Activation
        elif self.actFunction=='SoftMax':
            self.Activation= self.SoftMax(Z)
            self.FinalLayer=True
            return self.Activation
    
    #DA is obtained from layer in 'front' as that provides cha
    def backward(self,DA,trueLabelIndex):
        if self.actFunction=='ReLu':
            Dz=DA*self.derivative_ReLU(self.ZVal)
        if self.actFunction=='SoftMax':
            Dz=self.Activation.copy()
            Dz[trueLabelIndex]-=1
        
        #DZ is updated activation output which is a col vector

        # take 2 1d vectors multiply them together such that a new matrix is produced
        # that is the weight 
        Dw=np.outer(self.flatInput,Dz)
        
        # derving everything with bias in mind partial deriv so everything else 0s out and bias stays constant 
        Db=Dz


        self.Dw=Dw
        self.Db=Db
        #maybe need to transpose weights
        dInput=self.Weights@Dz
    
        return dInput.reshape(self.orginalShape)
    

    def requiresLabel(self):
        return True

    
    def ReLu(self,arr):
        return np.maximum(arr,0)

    # when reul less than 0 deriv constant at 0 when not its 1
    def derivative_ReLU(self,Z):
        return (Z>0).astype(int)

        
    def SoftMax(self,arr):

        shift=arr-np.max(arr)
        exponentArr=np.exp(shift)

        return exponentArr/exponentArr.sum()
        #compute enrties as e^zi then sum up all the values and divide by that 
        #return np.exp(arr)/np.sum(np.exp(arr))
    
    def updateParameters(self,learnrate):

        self.Weights-=self.Dw*learnrate
        self.bias-=self.Db*learnrate 