import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import copy

def r(t,x):
	m = len(t)
	n = len(x)
	# residual = np.zeros(m)
	# for i in range(m):
	residual = 1
	for j in range(n):
		residual *= (t-x[j])
	# residual_vector[i] = residual

	return residual

def t(m):
	t_vector = np.zeros(m)
	h = 2/m
	for i in range(1,m+1):
		t_vector[i-1] = -1-h/2+i*h

	return t_vector

def x(n):
	return np.linspace(-1, 1, num=n, endpoint=True)


def J(t,x):
	m = len(t)
	n = len(x)
	Jacobian = np.zeros((m,n))
	for j in range(n):
			# for i in range(n):
		h = copy.deepcopy(x)
		h[j]+=0.01
		Jacobian[:,j] = (r(t,h)-r(t,x))/0.01
		# for j in range(n):
		# 	dr = -1
		# 	for k in range(n):
		# 		if (k!=j):
		# 			dr *= (t[i]-x[k])
		# 	Jacobian[i,j] = dr

	return Jacobian

def LM(m,n):
	t_ = t(m)
	x_ = x(n)
	r_ = r(t_,x_)
	norm_r = la.norm(r_)
	for k in range(50):
		if (norm_r < 10**(-15)):
			break
		J_ = J(t_,x_)
		Mu_ = np.sqrt(np.exp(-k))*np.identity(n)
		lhs = np.vstack((J_,Mu_))
		rhs = np.vstack((-r_.reshape(m,1),np.zeros((n,1))))
		s = la.lstsq(lhs,rhs)[0].reshape(1,n)[0]
		x_ += s
		r_ = r(t_,x_)
		norm_r = la.norm(r_)
   
	return [x_,norm_r]

def GN(m,n):
	t_ = t(m)
	x_ = x(n)
	r_ = r(t_,x_)
	norm_r = la.norm(r_)
	Mu_ = np.zeros((n,n))
	for k in range(50):
		if (norm_r < 10**(-15)):
			break
		J_ = J(t_,x_)
		lhs = np.vstack((J_,Mu_))
		rhs = np.vstack((-r_.reshape(m,1),np.zeros((n,1))))
		s = la.lstsq(lhs,rhs)[0].reshape(1,n)[0]
		x_ += s
		r_ = r(t_,x_)
		norm_r = la.norm(r_)
   
	return [x_,norm_r]

m = 300
t1 = t(m)
x_uni = x(40)
q_uni = r(t1,x_uni)
x_opt = LM(m,40)[0]
q_opt = r(t1,x_opt)

plt.figure(0)
plt.plot(t1,q_uni,'r',label = 'Q interpolation by initial guessed X')
plt.plot(t1,q_opt,'b',label = 'Q interpolation by LM-optimized X')
plt.title('q interpolation vs X')
plt.legend()
plt.grid()
plt.show()


norm_uni = []
norm_LM = []
norm_GN = []
idx = []
for n in range(1,41):
	norm_LM.append(LM(m,n)[1])
	norm_GN.append(GN(m,n)[1])
	norm_uni.append(la.norm(r(t1,x(n))))
	idx.append(n)

plt.figure(1)
plt.semilogy(idx,norm_LM,label = 'Levenberg-Marquardt Method optimized')
plt.semilogy(idx,norm_GN,label = 'Gauss-Newton Method optimized')
plt.semilogy(idx,norm_uni,label = 'Uniform Point')
plt.legend()
plt.xlabel('n value')
plt.ylabel('2-norm of q')
plt.title('2-norm of q interpolation between three point sample stratagies')
plt.show()













