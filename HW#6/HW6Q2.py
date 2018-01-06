import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.optimize import fsolve
from scipy.sparse import diags


print("when going from the Dirichlet case to the Neumann case, the bottom-right entries of both A and B matrices have to take half")
def A_(n):
	k = np.array([-1*np.ones(n-1),2*np.ones(n),-1*np.ones(n-1)])
	offset = [-1,0,1]
	return diags(k,offset).toarray()

def B_(n):
	k = np.array([1/6*np.ones(n-1),2/3*np.ones(n),1/6*np.ones(n-1)])
	offset = [-1,0,1]
	return diags(k,offset).toarray()

def Dirichlet(n):
	A = (n+1)*A_(n)
	B = 1/(n+1)*B_(n)
	l = sp.linalg.inv(B)@A
	lamb, v = sp.linalg.eig(l)
	l_min = min(lamb)
	return l_min

def Neumann(n):	
	A = (n+1)*A_(n+1)
	B = 1/(n+1)*B_(n+1)
	A[-1,-1] *= 0.5
	B[-1,-1] *= 0.5
	l = sp.linalg.inv(B)@A
	lamb, v = sp.linalg.eig(l)
	l_min = min(lamb)
	return l_min

Dirichlet_a = np.pi**2
Neumann_a = 0.25*np.pi**2

n_ = [2,4,8,16,32,64,128,256,512]
n2 = []
D_e = []
N_e = []
for n in n_:
	D_e.append(np.abs(Dirichlet(n)-Dirichlet_a)/Dirichlet_a)
	N_e.append(np.abs(Neumann(n)-Neumann_a)/Neumann_a)
	n2.append(n**-2)
plt.plot(n_,D_e,label='Dirichlet')
plt.plot(n_,N_e,label='Neumann')
plt.plot(n_,n2,label='reference: n^-2')

plt.legend()
plt.xlabel('n')
plt.ylabel('error')
plt.xscale('log')
plt.yscale('log')
plt.title('BVP solve smallest eigenvalue error convergence')
plt.show()






