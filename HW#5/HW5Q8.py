import numpy as np 
import numpy.linalg as la
import matplotlib.pyplot as plt
import math
from scipy.integrate import odeint
import scipy.sparse as sparse

#You may find it convenient to use scipy.sparse.diags() 
#(and other sparse functions like sparse.eye()) to construct and work with the operator L
print('When n=40, u grows unboundedly to infinity.')
print('n=30 is the largest n can run for the parameters [ν,dt]=[.05,.01] and still have Euler forward be stable')

def dudt(v,n,u):
  L = sparse.diags([-1,2,-1], [-1, 0, 1],shape=(n,n))
  return -v*(n+1)**2*L@u + np.ones(n)

def solve(n):
  v = 0.05
  t = 0
  dt = 0.01
  dx = 1/(n+1)
  u0 = np.zeros(n)
  u = u0
  # x = 0
  u_ = [u]

  x_ = np.zeros(n)
  for j in range(n):
    x_[j] = j*dx

  t_ = [t]
  while (True):
      t += dt
      if (t>10):
          break
      # x += dx
      u = u + dt*dudt(v,n,u)
      u_10 = la.norm(u,np.inf)
      if (u_10 == np.inf):
          break
      u_.append(u)
      t_.append(t)
      # x_.append(x)
  print("||u(t=10)||∞ is ",u_10," when n=",n)
  return t_,u_,x_

t_20,u_20,x_20 = solve(20)
plt.figure()
for i in range(len(t_20)):
    if (i%100 == 0):
        plt.plot(x_20,u_20[i],label='t='+str(round(t_20[i])))
plt.xlabel('x')
plt.ylabel('u components')
plt.legend()
plt.title('Stability of unsteady diffusion @ n=20')
plt.grid()
plt.show()

t_40,u_40,x_40 = solve(40)
plt.figure()
for i in range(len(t_40)):
    if (i%100 == 0):
        plt.plot(x_40,u_40[i],label='t='+str(round(t_40[i])))
plt.xlabel('x')
plt.ylabel('u components')
plt.title('Stability of unsteady diffusion @ n=40')
plt.legend()
plt.grid()
plt.show()

t_30,u_30,x_30 = solve(30)
t_31,u_31,x_31 = solve(31)
plt.figure()
for i in range(len(t_30)):
    if (i%100 == 0):
        plt.plot(x_30,u_30[i],label='t='+str(round(t_30[i])))
for i in range(len(t_31)):
    if (i%100 == 0):
        plt.plot(x_31,u_31[i],label='t='+str(round(t_31[i])))
plt.xlabel('x')
plt.ylabel('u components')
plt.title('Stability of unsteady diffusion @ n=30 vs n=31')
plt.legend()
plt.grid()
plt.show()

x_20,u_20 = solve(20)
plt.figure()
plt.plot(x_20,u_20,label='n=20')
plt.xlabel('x')
plt.ylabel('u components')
plt.title('Stability of unsteady diffusion @ n=20')
plt.grid()
plt.show()

x_40,u_40 = solve(40)
plt.figure()
plt.plot(x_40,u_40,label='n=40')
plt.xlabel('x')
plt.ylabel('u components')
plt.title('Stability of unsteady diffusion @ n=40')
plt.grid()
plt.show()

x_30,u_30 = solve(30)
x_31,u_31 = solve(31)
plt.figure()
plt.plot(x_30,u_30,'r')
plt.plot(x_31,u_31,'b')
plt.xlabel('x')
plt.ylabel('u components')
plt.title('Stability of unsteady diffusion @ n=30(red line) vs n=31(blue line)')
plt.grid()
plt.show()




