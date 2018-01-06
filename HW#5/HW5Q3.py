import numpy as np 
import numpy.linalg as la
import matplotlib.pyplot as plt
import math
from scipy.integrate import odeint
import scipy.sparse as sparse

def D_m(x):
  n = len(x)
  c = np.ones((n,1))
  D = np.zeros((n,n))

  for i in range(n):
      for j in range(n):
          if (i != j):
              c[i] *= x[i]-x[j]
  c = 1./c

  for i in range(n):
      for j in range(n):
          if (i != j):
              D[i,j] = c[j]/(c[i]*(x[i]-x[j]))
          else:
              for k in range(n):
                  if (i != k):
                      D[i,j] += 1/(x[i]-x[k])
  return D

def f(x):
  return np.cos(5*x)
def f_d(x):
  return -5*np.sin(5*x)
def uni_x(n):
  return np.linspace(-1, 1, num=n, endpoint=True)
def GL_x(n):
  return np.polynomial.legendre.leggauss(n)[0]

num = []
err_uni_pt = []
err_GL_pt = []
for n in range(2,51):
  num.append(n)

  x_uni = uni_x(n)
  f_uni = f(x_uni)
  g_uni = D_m(x_uni)@f_uni
  fd_uni = f_d(x_uni)
  err_uni = max(abs(fd_uni-g_uni))
  err_uni_pt.append((err_uni))

  x_GL = GL_x(n)
  f_GL = f(x_GL)
  g_GL = D_m(x_GL)@f_GL
  fd_GL = f_d(x_GL)
  err_GL = max(abs(fd_GL-g_GL))
  err_GL_pt.append(err_GL)

plt.figure()
plt.plot(num,err_uni_pt,label='uniform points')
plt.plot(num,err_GL_pt,label='Gauss-Legendre points')
plt.yscale('log')
plt.title('Derivatives of Lagrange interpolants by two point sampling methods')
plt.xlabel('number of abscissas')
plt.ylabel('maximum pointwise error')
plt.legend()
plt.show()

print("both uniform point and Gauss_Legendre point methods have similar convergence rate, however, GL approach can reach and maintain a higher accuracy than uniform points when interpolation points are more than 20 for this problem")

