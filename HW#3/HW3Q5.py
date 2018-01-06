import scipy.linalg as la
import numpy as np
A = np.array([[6,2,1],[2,3,1],[1,1,1]])
M = A - 2*np.identity(3)
LU = la.lu_factor(M)
x0 = np.array([0,0,1])
for i in range(15):
	x0 = la.lu_solve(LU, x0)
eigvec = x0
eigval = (x0.T@A@x0)/(x0.T@x0)
e,v = np.linalg.eig(A)
diffval = min(np.abs(e-eigval))

import numpy.linalg as la
import numpy as np
import matplotlib.pyplot as plt

size = []
ratio = []
for i in range(1,16):
	A = (-1*np.triu(np.ones(i), 0)+2*np.identity(i))
	U, s, V = la.svd(A)
	s_max = max(s)
	s_min = min(s)
	k = s_max/s_min
	size.append(i)
	ratio.append(k)
print(A)
plt.plot(size,ratio)
plt.xlabel('size of matrix')
plt.ylabel('condition number of matrix')
plt.title('condition vs. size of nearly singular matrix')
plt.show()

print('From the plot it is evident that the condition number for a nearly singular matrix grows exponentially (with base a little bit more than 2).\nHowever, eigenvalues of all of these matrices are straight "1"s.\nThis indicates that a nearly singular matrix does not necessarily have a "small" eigenvalue.')
