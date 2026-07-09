from LSTM import LSTMCell
import numpy as np
import math
from DesnseLayer import DenseLayer
T=3


inputsize=3
hiddensize=3
numclasses=5



X=np.random.randn(T,inputsize,)
Y=np.random.randint(0,numclasses,T)

H=np.random.randn(hiddensize,)
C=np.random.randn(hiddensize,)

originalH=H
originalC=C

L=LSTMCell(inputsize,hiddensize)
D=DenseLayer(numclasses,hiddensize,None,'SoftMax',None)
params_to_check = {
    # forget gate
    'WeightFGInput': (L.WeightFGInput, L.DwXf),
    'WeightFGHide':  (L.WeightFGHide,  L.DwHf),
    'BiasFG':        (L.BiasFG,        L.Dbiasf),

    # input gate
    'WeightInGInput': (L.WeightInGInput, L.DwXi),
    'WeightInGHide':  (L.WeightInGHide,  L.DwHi),
    'BiasInG':        (L.BiasInG,        L.Dbiasi),

    # candidate gate
    'WeightCGInput': (L.WeightCGInput, L.DwXc),
    'WeightCGHide':  (L.WeightCGHide,  L.DwHc),
    'BiasCG':        (L.BiasCG,        L.Dbiasc),

    # output gate
    'WeightOutGInput': (L.WeightOutGInput, L.DwXo),
    'WeightOutGHide':  (L.WeightOutGHide,  L.DwHo),
    'BiasOutG':        (L.BiasOutG,        L.Dbiaso),
}
np.random.seed(0)
for name, (W, _) in params_to_check.items():
    W[:] = np.random.randn(*W.shape) * 0.1

cArray=[]
predicions=[]
for i in range(T):
    cached=L.forward(X[i],C,H)
    H=cached.get('hnew')
    C=cached.get('cnew')
    cArray.append(cached)
    Temp=D.forward(H)
    predicions.append(Temp)


DHPrevT=np.zeros(hiddensize)
DCellPrevT=np.zeros(hiddensize)
for i in reversed(range(T)):
    DHDense=D.backward(predicions[i],Y[i])
    DHTOT=DHDense+DHPrevT
    DCellPrevT,DHPrevT=L.backward(cArray[i],DHTOT,DCellPrevT)


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

def sparseCategoricalCrossEntropyLoss(prediction,trueLabelIndex):

        epsilon=1e-12

        #prediction array has all the probabilities
        # true label as index will select the probability that is computed for true label
        # if its high loss is low
        # if it is low loss is high (very wrong)
        return -np.log(prediction[trueLabelIndex]+epsilon)
    


np.random.seed(0)
for name, (W, dW_analytical) in params_to_check.items():

    dW_numerical, relative_error = checkMat(W, dW_analytical,forwardAndLoss,originalH,originalC)
    print(f"{name}: max relative error = {relative_error.max():.10f}")

