from ConvLayer import ConVLayer
from PoolLayer import PoolLayer
from DesnseLayer import DenseLayer
from CNN import CNN 
import cupy as np
import math
"""
#l=ConVLayer(2,3,(24,24,3))
#p=PoolLayer(2,(22,22,2),2)
testInput= np.random.randn(24,24,3)



c1=ConVLayer(5,3,testInput.shape)
model.addlayer(c1)
c2=ConVLayer(10,2,c1.outputShape)
model.addlayer(c2)
p1=PoolLayer(2,c2.outputShape)
model.addlayer(p1)
d1=DenseLayer(5,math.prod(p1.outputShape),None,'SoftMax')

model.addlayer(d1)
model.forward(testInput)

#out=l.forward(testInput)


#maxPoolOut=p.forward(out)


#print(out.shape)
#print(out)

#print(maxPoolOut.shape)
#print(maxPoolOut)

"""
import tensorflow.keras.datasets.mnist as mnist



(xtrain,ytrain),(xtest,ytest)=mnist.load_data()

print(xtrain.shape,ytrain.shape)
print(xtest.shape,ytest.shape)

xtrain=np.asarray(xtrain)
xtest=np.asarray(xtest)

ytrain=np.asarray(ytrain)
ytest=np.asarray(ytest)


xtrain=xtrain[...,np.newaxis].astype(np.float64)
xtest=xtest[...,np.newaxis].astype(np.float64)


xtrain/=255
xtest/=255

ytrain=ytrain.astype(int)
ytest=ytest.astype(int)
model=CNN(10)

c1=ConVLayer(8,3,(28,28,1),1,0)
model.addlayer(c1)


c2=ConVLayer(12,2,c1.outputShape,1,0)
model.addlayer(c2)

p1=PoolLayer(2,c2.outputShape)
model.addlayer(p1)
d1=DenseLayer(10,math.prod(p1.outputShape),None,'SoftMax',None)
model.addlayer(d1)


model.train(xtrain,ytrain,100,1000,0.01)