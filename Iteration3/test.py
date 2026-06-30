from ConvLayer import ConVLayer
from PoolLayer import PoolLayer
from DesnseLayer import DenseLayer
from CNN import CNN 
import numpy as np
import math

#l=ConVLayer(2,3,(24,24,3))
#p=PoolLayer(2,(22,22,2),2)
testInput= np.random.randn(24,24,3)

model=CNN(5)

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