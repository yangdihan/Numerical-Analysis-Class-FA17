import numpy as np 
import numpy.linalg as la
import matplotlib.pyplot as plt
import math
from scipy.integrate import odeint
import scipy.sparse as sparse

def f(x):
  return 4/(1+x**2)

def mid_comp(n):
  x = np.linspace(0, 1, num=n+1, endpoint=True)
  s = 0
  for i in range(n):
	  s += f((x[i]+x[i+1])/2)
  s /= n
  return s

def tra_comp(n):
  x = np.linspace(0, 1, num=n+1, endpoint=True)
  s = 0
  for i in range (n):
	  s += 0.5*f(x[i])
	  s += 0.5*f(x[i+1])
  s /= n
  return s

def sim_comp(n):
  x = np.linspace(0, 1, num=n+1, endpoint=True)
  s = 0
  for i in range (n):
	  s += 1/6*f(x[i])
	  s += 2/3*f((x[i]+x[i+1])/2)
	  s += 1/6*f(x[i+1])
  s /= n
  return s

# def Romberg(f,rng,n):
#    size = int(np.log2(n)+1)
#    T = np.zeros((size,size))
#    # Compute T k,0
#    for k in range(size):
#        n_curr = 2**k
#        T[k,0] = Trapezoid(f,rng,n_curr)
#    # Compute T j,k
#    for k in range(1,size):
#        for j in range(size-1,k-1,-1):
#            T[j,k] = ((4**k)*T[j,k-1]-T[j-1,k-1])/(4**k-1)
   
#    return T[-1,-1]

def romb(n):
    # n = 1,2,3,4,5
    # size = 2,4,8,16,32
    size = 2**n
    T = np.zeros((n,n))
    T[0,0] = tra_comp(2**0)
    for k in range(1,n):
        # T[k,0] = tra_comp(int(n/2**k))
        T[k,0] = tra_comp(2**k)
        for j in range(1,k+1):
            T[k,j] = (4**j*T[k,j-1]-T[k-1,j-1])/(4**j-1)
    return T[-1,-1]

# def romb(size):
# 	# n = 1,2,3,4,5
# 	# size = 2,4,8,16,32
# 	# size = 2**n
# 	n = int(np.log2(size)+1)
# 	T = np.zeros((n,n))
# 	T[0,0] = tra_comp(2**0)
# 	for k in range(1,n):
# 		# T[k,0] = tra_comp(int(n/2**k))
# 		T[k,0] = tra_comp(2**k)
# 		for j in range(1,k+1):
# 			T[k,j] = (4**j*T[k,j-1]-T[k-1,j-1])/(4**j-1)
# 	return T[-1,-1]

def GL(n):
  x,w = np.polynomial.legendre.leggauss(n)
  s = 0
  for i in range(n):
	  s += w[i]*f((x[i]+1)/2)
  return s/2

num = []
# num_rom = []

mid_err = []
tra_err = []
sim_err = []
romb_err = []
GL_err = []
for i in range(1,21):
	num.append(i)
	mid_err.append(abs(mid_comp(i)-np.pi))
	tra_err.append(abs(tra_comp(i)-np.pi))
	sim_err.append(abs(sim_comp(i)-np.pi))
	romb_err.append(abs(romb(i)-np.pi))
	GL_err.append(abs(GL(i)-np.pi))

# err = 99
# while (err )
plt.figure()

# plt.plot(2**np.array(num),mid_err,label='midpoint composite quadrature')
# plt.plot(2**np.array(num),tra_err,label='trapezoid composite quadrature')
# plt.plot(2**np.array(num),sim_err,label='Simpson composite quadrature')
# plt.plot(num,romb_err,label='Romberg integration')
# plt.plot(2**np.array(num),GL_err,label='Gauss-Legendre quadrature')

plt.plot(num,mid_err,label='midpoint composite quadrature')
plt.plot(num,tra_err,label='trapezoid composite quadrature')
plt.plot(num,sim_err,label='Simpson composite quadrature')
plt.plot(2**np.array(num)[:5],romb_err[:5],label='Romberg integration')
plt.plot(num,GL_err,label='Gauss-Legendre quadrature')

plt.xscale('log')
plt.yscale('log')
plt.ylabel('absolute error')
plt.xlabel('number of function evaluations')
plt.xlim(0,20)
plt.legend()
plt.title('Numerical Integration Errors of 5 methods at increasing number of functions evaluation')
plt.show()