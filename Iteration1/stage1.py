import numpy as np
import matplotlib.pyplot as plt 


np.random.seed(0)


# clamps values between 0 and 1
# very negative and youre close to 0 
# very postive and youre close to 1

def sigmoid (z):
    return 1/(1+np.exp(-z))

# ytrue and YHat need to be collections of values not just one value or its poitnless

def lossBinCrossEntropy(yTrue,yHat):
    BCE=-np.mean((yTrue*np.log(yHat))+(1-yTrue)*np.log(1-yHat))

    return BCE

#2 classifiable clusters of information 
#see cs231 notes on the image thing 
# below want to mathematically classify into one of the 2 clusters
class1= np.random.randn(100,2)+np.array([0,0])
class2= np.random.randn(100,2)+np.array([4,4])

X=np.vstack([class1,class2])
Y=np.hstack([np.zeros(100),np.ones(100)])

w= np.zeros(2)
b=0.0

learningRate=0.1
epochs =1000

losses=[]

for epoch in range(epochs):
    
        #computed score value
        #should give 200 z values 

        #NB be careful of matrix shape 
        # X is 200 by 2 
        # w is 2 by 1 only works if X@w cuase then 200x2 X 2x1
        # giving a 200x1 matrix
        z=X@w +b


        # Sigmoid works natively because of how numpy works under the hood
        # takes the full 200x1 matrix outputs a 1x200 yhat calc
        yHat=sigmoid(z)

        #compute loss now
        #how wel youve done on single training iteration
        Loss=lossBinCrossEntropy(Y,yHat) 
        losses.append(Loss)

        #gradient descent
        # train/learn
        # 1x200 X 200x2  
        averageSize=X.shape[0]

        #dL/db=Yhat-Y
        #dL/dw=Y-Yhat*X 
        # above are derivatives computed by hand using the formulas used
        dw=(yHat-Y)@X /averageSize
        db=sum(yHat-Y)/averageSize


        #use computed gradients to update weight and bias
        w=w-learningRate*dw
        b=b-learningRate*db

plt.figure(1)
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.plot(losses)
plt.savefig("TainLoss.png")
      

