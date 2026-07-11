import cupy as np
from Filter import Filter
from cupy.lib.stride_tricks import as_strided
class PoolLayer:
    def __init__(self,poolFilterSize,inputShape,poolStride=2,padding=0):

        #single / returns float
        #// for forced int
        height,width,depth=inputShape
        self.inputShape=inputShape
        self.outputShape=(((height-poolFilterSize+(2*padding))//poolStride)+1,((width-poolFilterSize+(2*padding))//poolStride)+1,depth)
        self.stride=poolStride
        self.filterSize=poolFilterSize
        self.maxIndices=None

        self.frameCachedIndices=[]

    def forward(self,input):
        outputHeight,outputWidth,outputDepth=self.outputShape
     

        patches=self.retrieveWindows(input)

        # axis 3,4 is the max within the smaller windows
        #return 1 value per H,W,C so shape should match 
        output=patches.max(axis=(3,4))

        flat=patches.reshape(outputHeight,outputWidth,outputDepth,-1)

        self.maxIndices=flat.argmax(axis=-1)
        

        #for  d in range(outputDepth):
         #   for r in range(outputHeight):
          #      for c in range(outputWidth):
           #         patch=input[r*self.stride:r*self.stride+self.filterSize,c*self.stride:c*self.stride+self.filterSize,d]

                    # flatten the patch 2d thing and turn it into a logn list
                    #look for the highest value in that list return its index
             #       maxIndex=np.argmax(patch)
            #        #use max index and the original shape of the patch to get coordinates of the patch
                    #max2DCoords=np.unravel_index(maxIndex,patch.shape)

                    #if backprop breaks in pooiling check this logic
              #      self.maxIndices[r,c,d]=int(maxIndex)
               #     output[r,c,d]=np.max(patch)

        self.frameCachedIndices.append(self.maxIndices)
        return output
    
    def backward(self,dOutput,frameIndex):

        self.maxIndices=self.frameCachedIndices[frameIndex]
        #copy the structure of the next layer and fill 0s all aroun
        #do cause relevant entries will be updated and have values the ones that didnt contribute wont get anything
        dInput=np.zeros(self.inputShape)
        outputHeight,outputWidth,outputDepth=self.outputShape
        maxValH,MaxValW=np.unravel_index(self.maxIndices,(self.filterSize,self.filterSize))
    

        heightVector=np.arange(outputHeight).reshape(outputHeight,1,1)
        WidthVector=np.arange(outputWidth).reshape(1,outputWidth,1)
        depthVector=np.arange(outputDepth).reshape(1,1,outputDepth)
        
        #needs to match the dimensions in future so that np add works
        depthVector=np.broadcast_to(depthVector,(outputHeight,outputWidth,outputDepth))
        
        windowXPos=heightVector*self.stride+maxValH
        windowYPos=WidthVector*self.stride+MaxValW


        np.add.at(dInput,(windowXPos,windowYPos,depthVector),dOutput)

        
                
       # for  d in range(outputDepth):
        #    for r in range(outputHeight):
         #       for c in range(outputWidth):

          #          index=int(self.maxIndices[r,c,d])
           #         maxValH,MaxValW=np.unravel_index(np.array(index),(self.filterSize,self.filterSize))

                    #use the row and stride with stored cordinates in the max indicies to update only the activation neuron that needs updating due to the layer after this one
            #        dInput[r*self.stride+maxValH,c*self.stride+MaxValW,d]+=dOutput[r,c,d]

        return dInput
    

    def retrieveWindows(self,input):
        outputHeight,outputWidth,outputDepth=self.outputShape
       
        reshapedInput=(outputHeight,outputWidth,outputDepth,self.filterSize,self.filterSize)
        #how to move around byte by byte
        #think assembly registers
        updatedStrides=(
            input.strides[0]*self.stride,
            input.strides[1]*self.stride,
            input.strides[2],
            input.strides[0],
            input.strides[1],
        )
        windows=as_strided(input,reshapedInput,updatedStrides)
        return windows

    def clearCache(self):
        self.frameCachedIndices=[]