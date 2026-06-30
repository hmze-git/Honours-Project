from ConvLayer import ConVLayer
from PoolLayer import PoolLayer

import numpy as np

l=ConVLayer(2,3,(24,24,3))
p=PoolLayer(2,(22,22,2),2)
testInput= np.random.randn(24,24,3)



out=l.forward(testInput)


maxPoolOut=p.forward(out)


print(out.shape)
#print(out)

print(maxPoolOut.shape)
print(maxPoolOut)