from LSTM import LSTMCell
import cupy as np
import math
from Iteration4.LSTMDesnseLayer import DenseLayer
import tensorflow.keras.datasets.mnist as mnist

(xtrain,ytrain),(xtest,ytest)=mnist.load_data()


xtrainSmall=xtrain[:200]
ytrainSmall=ytrain[:200]


xtrainSmall=np.asarray(xtrainSmall)
ytrainSmall=np.asarray(ytrainSmall)

xtrainSmall=xtrainSmall.astype(np.float64)/255.0

T=3


inputsize=28
hiddensize=64
numclasses=10




def sparseCategoricalCrossEntropyLoss(prediction,trueLabelIndex):

        epsilon=1e-12

        #prediction array has all the probabilities
        # true label as index will select the probability that is computed for true label
        # if its high loss is low
        # if it is low loss is high (very wrong)
        return -np.log(prediction[trueLabelIndex]+epsilon)


L=LSTMCell(inputsize,hiddensize)
D=DenseLayer(numclasses,hiddensize,None,'SoftMax',None)





for e in range(200):
    totalLoss=0
    predicions=[]
    for x in range(200):
        DHPrevT=np.zeros(hiddensize)
        DCellPrevT=np.zeros(hiddensize)

        H=np.zeros(hiddensize,)
        C=np.zeros(hiddensize,)
        cArray=[]

        for i in range(28):
           cObj=L.forward(xtrainSmall[x][i],C,H)
           H=cObj.get('hnew')
           C=cObj.get('cnew')
           cArray.append(cObj)
        prediction=D.forward(H)
        predicions.append(prediction)

        totalLoss+=sparseCategoricalCrossEntropyLoss(prediction,ytrainSmall[x])

        DHDense=D.backward(prediction,ytrainSmall[x])
     
        for i in reversed(range(28)):
        
            if i==27:
                DHTOT=DHDense+DHPrevT
            else:
                DHTOT=DHPrevT

            DCellPrevT,DHPrevT= L.backward(cArray[i],DHTOT,DCellPrevT)

       # print(f"Sample {x}:")
       # print(f"  probs: {prediction}")
       # print(f"  argmax: {np.argmax(prediction)}")
       # print(f"  true label: {ytrain[x]}")

        L.update(0.1)
        L.zeroDeriGrad()
    print(f"Epoch {e}: avg loss = {totalLoss/200:.4f}")
    correct = sum(np.argmax(pred) == true for pred, true in zip(predicions, ytrainSmall))
    print(f"Accuracy: {correct/200*100:.1f}%")



def checkMat(WeightMatrix,dwAnalatica,forwardAndLoss,H,C,epsilon=1e-5):

    dwNumerical=np.zeros_like(WeightMatrix)

    it=np.nditer(WeightMatrix,flags=['multi_index'])

    for _ in it:
        idx=it.multi_index

        ogValue=WeightMatrix[idx]


        WeightMatrix[idx]=ogValue+epsilon

        lossPlus=forwardAndLoss(H,C)



        WeightMatrix[idx]=ogValue-epsilon
        lossMinus=forwardAndLoss(H,C)


        #reset after changing
        WeightMatrix[idx]=ogValue

        dwNumerical[idx]=(lossPlus-lossMinus)/(2*epsilon)


        diff=np.abs(dwAnalatica-dwNumerical)
        denom=np.abs(dwAnalatica)+np.abs(dwNumerical)+1e-8

        relErr=diff/denom

    return dwNumerical,relErr


def forwardAndLoss(H,C):

    total_loss = 0

    for i in range(T):
        cached = L.forward(X[i], C, H)
        H = cached.get('hnew')
        C = cached.get('cnew')
        pred = D.forward(H)                
        total_loss += sparseCategoricalCrossEntropyLoss(pred, Y[i]) 

    return total_loss 

    

#
##np.random.seed(0)
#for name, (W, dW_analytical) in params_to_check.items():

 #   dW_numerical, relative_error = checkMat(W, dW_analytical,forwardAndLoss,originalH,originalC)
  #  print(f"{name}: max relative error = {relative_error.max():.10f}")

