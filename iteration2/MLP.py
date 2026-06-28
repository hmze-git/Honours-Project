from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt 



def retrieveDataset():
    minst=fetch_openml('mnist_784', version=1,as_frame=False)

    X,Y=minst['data'],minst['target']
    # Display the shape of data and labels
    print(f"Data shape: {X.shape}")
    print(f"Labels shape: {Y.shape}")

    # Display the first few labels
    print(f"First 10 labels: {Y[:10]}")

    Y=Y.astype(int)
    X=X/255.0

    return X,Y



def initMLP():
    # 5 neurons in H1 784 in Input layer so we need 5 rows with 784 cols
    W1=np.random.randn(784,64)*np.sqrt(2/764)
    W2=np.random.randn(64,32)*np.sqrt(2/64)
    W3=np.random.randn(32,10)*np.sqrt(2/32)

    #biases
    BH1=np.zeros((1,64))
    BH2=np.zeros((1,32))
    BH3=np.zeros((1,10))

    return W1,W2,W3,BH1,BH2,BH3


def ReLu(arr):
    return np.maximum(arr,0)


def SoftMax(arr):

    shift=arr-np.max(arr,axis=1,keepdims=True)
    exponentArr=np.exp(shift)

    return exponentArr/exponentArr.sum(axis=1,keepdims=True)
    #compute enrties as e^zi then sum up all the values and divide by that 
    #return np.exp(arr)/np.sum(np.exp(arr))

def forwardPass(W1,W2,W3,BH1,BH2,BH3,X):
    #score for the first layer
    #W1 784x5
    #X 70000x784
    Z1= X@W1+BH1

    A1=ReLu(Z1)

    Z2= A1@W2 +BH2

    A2=ReLu(Z2)

    Z3=A2@W3+BH3

    A3=SoftMax(Z3)

    return Z1,A1,Z2,A2,Z3,A3


def oneHotEncode(LabelSet):

    # max is 9 and we have 10 categories so we end up with 70 000 rows and 10 columns
    encodedOutputLabels=np.zeros((LabelSet.size,LabelSet.max()+1),dtype=int)
    encodedOutputLabels[np.arange(LabelSet.size),LabelSet]=1
    
    return encodedOutputLabels


# when reul less than 0 deriv constant at 0 when not its 1
def derivative_ReLU(Z):
    return (Z>0).astype(int)

def backPropagation(Y,Z1,A1,W1,Z2,A2,W2,Z3,A3,W3,X):


    m=Y.size
    # one hot encode first because working with actual digits/categories caues issues in math
    encodedY=oneHotEncode(Y)
    Loss=- np.sum(encodedY*np.log(A3))

    # derivation is confusing so use formula and dagestan forget
    DZ3=A3-encodedY

    #because of the fact were going in reverse we need to use reverse/transpose of prev matrix to compute weight
    #change 

    #DZ3: &70,000x10
    #A2: 70,000x10
    DW3=1/m*A2.T@DZ3

    DB3=1/m *np.sum(DZ3,axis=0,keepdims=True)

    # hidden layer 2
    DZ2= (DZ3@W3.T) * derivative_ReLU(Z2)

    DW2=1/m*A1.T@DZ2

    DB2=1/m *np.sum(DZ2,axis=0,keepdims=True)


    #hidden layer 1

    DZ1=(DZ2@W2.T)*derivative_ReLU(Z1)
    
  
    DW1=1/m*X.T@DZ1

    DB1=1/m *np.sum(DZ1,axis=0,keepdims=True)



    return DW1,DB1,DW2,DB2,DW3,DB3,Loss


def updateLoop(W1,B1,DW1,DB1,W2,B2,DW2,DB2,W3,B3,DW3,DB3,learnrate):
    W1=W1-learnrate*DW1
    B1=B1-learnrate*DB1

    W2=W2-learnrate*DW2
    B2=B2-learnrate*DB2

    W3=W3-learnrate*DW3
    B3=B3-learnrate*DB3

    return W1,B1,W2,B2,W3,B3

def predictions(OL):
    return np.argmax(OL,1) #get single highest value

def genAccuracy(prediction,Y):
    print(f"Preiction is {prediction}")
    print(f"Actual is {Y}")

    return np.sum(prediction==Y)/Y.size

def training(epochs,learnRate):

    X,Y= retrieveDataset()

    W1,W2,W3,BH1,BH2,BH3=initMLP()


    for e in range(epochs):
        Z1,A1,Z2,A2,Z3,A3=forwardPass(W1,W2,W3,BH1,BH2,BH3,X)

        DW1,DB1,DW2,DB2,DW3,DB3,Loss=backPropagation(Y,Z1,A1,W1,Z2,A2,W2,Z3,A3,W3,X)

        W1,BH1,W2,BH2,W3,BH3=updateLoop(W1,BH1,DW1,DB1,W2,BH2,DW2,DB2,W3,BH3,DW3,DB3,learnRate)

   

        if e%50==0:
             print(f"Loss:{Loss}")
             print("Iteration",e)
             print("Accuracy:",genAccuracy(predictions(A3),Y))


training(1000,0.4)


        

    



