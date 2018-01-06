import numpy as np
import matplotlib.pyplot as plt

def ydata(x):
	t = 20*np.random.rand(200,1) #Generate random points on interval
	t = np.sort(t, axis=0) #Sort the points
	#Evaluate function at points
	y = x[0,0]*np.exp(-x[1,0]*t)*np.sin(x[2,0]*t-x[3,0])+x[4,0]
	return y, t

a = 0.3 + 2*(np.random.rand() - 0.5)/2
b = 0.1 + 2*(np.random.rand() - 0.5)/25
omega = 4 + 2*(np.random.rand() - 0.5)/2
phase = -1.0 + (2*(np.random.rand() - 0.5))/2
c = 1.0 + (2*(np.random.rand() - 0.5))/2
coeffs = np.array([a, b, omega, phase, c])
#coeffs = np.array([0.3, 0.1, 4.0, -1.0, 1.0])

coeffs = coeffs.reshape((5,1))
[y,t] = ydata(coeffs)


import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
# print(y) #Your code can access these provided  n x 1 numpy arrays
# print(t)

#You can use this as your initial guess.
def R(x,y,t):
	x1 = x[0]
	x2 = x[1]
	x3 = x[2]
	x4 = x[3]
	x5 = x[4]
	r = y - (x1*np.exp(-x2*t)*np.sin(x3*t+x4)+x5)
	return r


x = np.array([0.3, 0.1, 4.0, -1.0, 1.0])
r = R(x,y,t)
norm_r = la.norm(r)
k = 0
while (not(norm_r < 10**-6)):
	k += 1
	m = len(t)
	n = len(x)
	r = R(x,y,t)
	mu = r.T@r
	x1 = x[0]
	x2 = x[1]
	x3 = x[2]
	x4 = x[3]
	x5 = x[4]
	J = np.zeros((m,5))
	for i in range(m):
		tcurr = t[i]
		J[i,0] = -np.exp(-x2*tcurr)*np.sin(x3*tcurr+x4)
		J[i,1] = tcurr*np.exp(-x2*tcurr)*x1*np.sin(x3*tcurr+x4)
		J[i,2] = -tcurr*x1*np.exp(-tcurr*x2)*np.cos(x4 + tcurr*x3)
		J[i,3] = -x1*np.exp(-tcurr*x2)*np.cos(x4 + tcurr*x3)
		J[i,4] = -1
	M_mu = np.sqrt(mu)*np.identity(n)
	lhs = np.vstack((J,M_mu))
	rhs = np.vstack((-r.reshape(m,1),np.zeros((n,1))))
	s = la.lstsq(lhs,rhs)[0].reshape(1,5)[0]
	x = x+s
	r = R(x,y,t)
	norm_r = la.norm(r)

x = x.reshape((5,1))

x1 = x[0]
x2 = x[1]
x3 = x[2]
x4 = x[3]
x5 = x[4]
plt.plot(t,y,label = 'Function Curve')
f = x1*np.exp(-x2*t)*np.sin(x3*t+x4)+x5
plt.plot(t,f,'ro',label = 'Fitting Curve')
plt.xlabel('abscissas')
plt.ylabel('function value')
plt.legend()
plt.grid()
plt.title('Data fitting: Decaying sinusoid')
plt.show()


