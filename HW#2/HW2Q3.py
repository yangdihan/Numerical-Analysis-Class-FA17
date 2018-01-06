import numpy as np
import numpy.linalg as la

A = np.zeros((10,4))
A[:4,:4] = np.identity(4)
A[4:7,0] = 1
A[7:9,1] = 1
A[9,2] = 1
A[4,1] = -1
A[5,2] = -1
A[6,3] = -1
A[7,2] = -1
A[8:,3] = -1

f = np.array([2.95,1.74,-1.45,1.32,1.23,4.45,1.61,3.21,0.45,-2.75])

x = (la.lstsq(A,f))[0]
x_m = np.array([2.95,1.74,-1.45,1.32])
rel_errors = []

for i in range(4):
	rel_errors.append(np.abs((x_m[i] - x[i])/x[i]))

rel_errors = np.array(rel_errors)
print(x)