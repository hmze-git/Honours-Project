from LSTM import LSTMCell
import numpy as np
import math

L=LSTMCell(2,2)

L.WeightFGHide= np.array([[0.1, 0.2],
                           [0.1, 0.2]])


L.WeightFGInput = np.array([[0.1, 0.2],
                           [0.1, 0.2]])      # shape (inputSize, hiddenSize)


L.BiasFG = np.array([0.0, 0.0])

L.WeightInGInput=np.array([[0.05, 0.10],
                           [0.05, 0.10]])

L.WeightInGHide= np.array([[0.05, 0.10],
                           [0.05, 0.10]])



L.BiasInG = np.array([0.0, 0.0])


L.WeightCGInput = np.array([[-0.05, -0.10],
                           [-0.05, -0.10]])


L.WeightCGHide = np.array([[-0.05, -0.10],
                           [-0.05, -0.10]])


L.BiasCG  = np.array([0.0, 0.0])


L.WeightOutGInput = np.array([[0.20, 0.25],
                           [0.20, 0.25]])


L.WeightOutGHide = np.array([[0.20, 0.25],
                           [0.20, 0.25]])


L.BiasOutG= np.array([0.0, 0.0])

x_t     = np.array([1.0, 2.0])
h_prev  = np.array([0.5, -0.5])
c_prev  = np.array([0.1, 0.2])



outH,outc=L.forward(x_t,c_prev,h_prev)


print(f"hidden {outH}")
print(f"cell {outc}")

