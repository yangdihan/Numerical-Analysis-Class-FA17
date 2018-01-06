import numpy as np
import numpy.linalg as la
import math
import matplotlib.pyplot as plt
from copy import deepcopy
def HilbertMat(n):
	# n = 2~12
	H = np.zeros((n,n))
	for i in range(n):
		for j in range(n):
			H[i,j] = 1/(i+j+1)
	return H

def CGS(A):
	Q = np.zeros(A.shape)
	n = A.shape[1]
	R = np.zeros((n,n))
	for k in range(n):
		Q[:,k] = A[:,k]
		for j in range(k):
			R[j,k] = Q[:,j].T@A[:,k]
			Q[:,k]-= R[j,k]*Q[:,j]
		R[k,k] = la.norm(Q[:,k],2)
		if (R[k,k] == 0):
			break
		Q[:,k] /= R[k,k]
	return Q

def MGS(A):
	Q = np.zeros(A.shape)
	n = A.shape[1]
	R = np.zeros((n,n))
	for k in range(n):
		R[k,k] = la.norm(A[:,k],2)
		if (R[k,k] == 0):
			break
		Q[:,k] = A[:,k]/R[k,k]
		for j in range(k+1,n):
			R[k,j] = Q[:,k].T@A[:,j]
			A[:,j]-= R[k,j]*Q[:,k]
	return Q

def HH(A):
	m = A.shape[0]
	n = A.shape[1]
	# a = A
	I = np.eye(n)
	Q = I
	for k in range(n):
		v = A[:,k]
		v[:k] = 0
		al = -np.sign(A[k,k])*la.norm(v,2)
		v[k]-= al
		bt = v.T@v
		for j in range(k,n):
			gm = v.T@A[:,j]
			A[:,j]-= 2*gm/bt*v
		# print(np.kron(v,v.T))
		# print(v.reshape(n,1)@v.T.reshape(1,n))
		# H = I - 2/bt*(v*v.T)
		H = I - 2/bt*(v.reshape(n,1)@v.T.reshape(1,n))
		Q = Q@H.T
	# Qh = Qh.T
	# return Qh
	return Q

def DoA(Q):
	dim = len(Q)
	I = np.identity(dim)
	o = -math.log10(la.norm((I - Q.T@Q),2))
	# print(la.norm((I - Q.T@Q),2))
	return o

hilbert = []
size = []
DoA_CGS = []
DoA_MGS = []
DoA_dCGS = []
DoA_HH = []

for n in range(2,13):#13
	H = HilbertMat(n)
	originH = deepcopy(H)
	hilbert.append(originH)
	size.append(n)

	Q_CGS = CGS(H)
	Q_MGS = MGS(H)
	QQ_CGS = CGS(Q_CGS)
	Q_HH = HH(H)

	DoA_CGS.append(DoA(Q_CGS))
	DoA_MGS.append(DoA(Q_MGS))
	DoA_dCGS.append(DoA(QQ_CGS))
	DoA_HH.append(DoA(Q_HH))


plt.rcParams.update({'font.size': 7})
plt.plot(size,DoA_CGS,'r',label='Classical Gram-Schmidt')
plt.plot(size,DoA_MGS,'g',label='Modified Gram-Schmidt')
plt.plot(size,DoA_dCGS,'darkred',label='double Classical Gram-Schmidt')
plt.plot(size,DoA_HH,'b',label='Householder')
plt.xlabel('size of Hilbert Matrix')
plt.ylabel('digits of accuracy')
plt.title('digits of accuracy of four orthogonalization method')
plt.legend(loc='lower left')
plt.show()
	# Q = np.zeros(A.shape)
	# m = A.shape[0]
	# n = A.shape[1]
	# R = np.zeros((n,n))
	# e = np.zeros(A.shape)
	# e+= np.identity(n)
	# v = np.zeros(A.shape)
	# alpha = np.zeros(n)
	# beta = np.zeros(n)
	# gamma = np.zeros(n)
	# for k in range(n):
	# 	if (A[k,k] > 0):
	# 		sign = -1
	# 	else:
	# 		sign = 1
	# 	alpha[k] = sign*np.norm(A[k:,k],2)
	# 	temp = np.zeros(m)
	# 	temp[k:] = A[k:,k]
	# 	v[:,k] = temp - alpha[k]*e[:,k]
	# 	beta[k] = v[:,k]@v[:,k]
	# 	if (beta[k] == 0):
	# 		continue
	# 	for j in range(k,n):
	# 		gamma[j] = v[:,k]@A[:,j]
	# 		A[:,j]-= 2*gamma[j]/beta[k]*v[:,k]
# 	import numpy as np

# def HilbertMat(n):
# 	# n = 2~12
# 	H = np.zeros((n,n))
# 	for i in range(n):
# 		for j in range(n):
# 			H[i,j] = 1/(i+j+1)
# 	return H

# # construct hilbert matrixes
# hilbert = []
# for n in range(2,13):
#     hilbert.append(HilbertMat(n))

