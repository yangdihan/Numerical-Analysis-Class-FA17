
import scipy.linalg as la
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

def f(x):
	x1 = x[0]
	x2 = x[1]
	f1 = (x1+3)*(x2**3-7)+18
	f2 = np.sin(x2*np.exp(x1)-1)
	return np.array([f1,f2])

def J(x):
	x1 = x[0]
	x2 = x[1]
	J11 = x2**3-7
	J12 = 3*(x1+3)*x2**2
	J21 = x2*np.exp(x1)*np.cos(x2*np.exp(x1)-1)
	J22 = np.exp(x1)*np.cos(x2*np.exp(x1)-1)
	return np.array([[J11,J12],[J21,J22]])

x0 = np.array([-0.5,1.4])
x_acc = np.array([0,1])
eps = np.finfo(float).eps

e_newton = []
x_n = deepcopy(x0)
error_n = la.norm(x_n - x_acc,2)

e_broyden = []
x_b = deepcopy(x0)
B = J(x_b)
error_b = la.norm(x_b - x_acc,2)

ct = 0
iter_ = []

while (error_b >= eps or error_n >= eps):
	x_n += la.solve(J(x_n),-f(x_n))
	error_n = la.norm(x_n - x_acc,2)
	e_newton.append(error_n)

	x_last = deepcopy(x_b)
	s = la.solve(B,-f(x_b))
	x_b += s
	y = f(x_b) - f(x_last)
	B = B + ((y.reshape(2,1) - B@s.reshape(2,1))@s.reshape(1,2))/(s.reshape(1,2)@s.reshape(2,1))
	error_b = la.norm(x_b - x_acc,2)
	e_broyden.append(error_b)

	ct += 1
	iter_.append(ct)


plt.plot(iter_,e_newton,'r',label="error by Newton's Method")
plt.plot(iter_,e_broyden,'b',label="error by Broyden's Method")
plt.yscale('log')
plt.xlabel('iteration steps')
plt.ylabel('error for root approximation')
plt.title("compare for convergence rate between Newton's Method and Broyden's Method")
plt.legend()
plt.show()











