import numpy as np

class Filter:


    def __init__(self,fHeight,fWidth,fDepth,weights=None,bias=0.0):

        if weights is None:
            self.weights=np.random.randn(fHeight,fWidth,fDepth) *0.1
        else:
            self.weights=weights
        self.bias=bias
        self.width=fWidth
        self.height=fHeight
        self.dWeights=None
        self.dBias=None


    #take the patch sliced out
    #remember the filter is a 3D map contribute to  a single entry in the output 2d act map
    #compute using np.sum 
    #return the 2d output
    def applyFilter(self,patch):    
        # bias is not a matrix but rather a single value linked to every resulting entry/"neuron"
        return np.sum(self.weights*patch)+self.bias