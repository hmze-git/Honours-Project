import cupy as np
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

        return nxtInput
    
    def backward(self,frameIndex,truelabelIndex):
        dInput=None

        for layer in reversed(self.layers):
            if hasattr(layer,'requiresLabel')and layer.FinalLayer==True:
                dInput=layer.backward(dInput,frameIndex,truelabelIndex)
            else:
                dInput=layer.backward(dInput,frameIndex)
        return dInput
    
    def update(self,learnRate):
        for layer in self.layers:
            if hasattr(layer,'updateParameters'):
                layer.updateParameters(learnRate)

    def sparseCategoricalCrossEntropyLoss(self,prediction,trueLabelIndex):

        epsilon=1e-12

        #prediction array has all the probabilities
        # true label as index will select the probability that is computed for true label
        # if its high loss is low
        # if it is low loss is high (very wrong)
        return -np.log(prediction[trueLabelIndex]+epsilon)
    
    def train(self,xtrain,ytrain,epochs,numEntries,lr=0.01):
        n=numEntries or len(xtrain)

        for epoch in range(epochs):
            totalLoss=0
            correct=0
            for i in range(n):
                probs=self.forward(xtrain[i])
               # print(probs)
                loss=self.sparseCategoricalCrossEntropyLoss(probs,ytrain[i])

                totalLoss+=loss

                if np.argmax(probs)==ytrain[i]:
                    correct+=1


                
                print(f"Sample {i}:")
                print(f"  probs: {probs}")
                print(f"  argmax: {np.argmax(probs)}")
                print(f"  true label: {ytrain[i]}")
                print(f"  prob of true class: {probs[ytrain[i]]:.6f}")
                print(f"  loss: {-np.log(probs[ytrain[i]] + 1e-12):.4f}")


                self.backward(ytrain[i])
                self.update(lr)
                avgLoss=totalLoss/(i+1)
                acc=correct/(i+1)
                print(f"Epoch {epoch}: loss={avgLoss:.4f}, accuracy={acc:.4f}")

    def accuracy(self,predictions,labels):
        predictions=np.array(predictions)
        lables=np.array(labels)

        return np.mean(predictions==lables)