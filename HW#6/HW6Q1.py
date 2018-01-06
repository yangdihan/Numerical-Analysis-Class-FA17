import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.optimize import fsolve
from scipy.sparse import diags

def eqn_sys(u):
    u = list(u)
    n = len(u)
    h = 1/(n+1)
    u_ = [0] + u + [1]
    eqns = []
    for i in range(n):
        ti = h*(i+1)
        eqn = ((u_[i]-2*u_[i+1]+u_[i+2]))/h**2 - (10*u_[i+1]**3+3*u_[i+1]+ti**2)
        eqns.append(eqn)
    return eqns

for k in [1,3,7,15]:
	time = np.linspace(0,1,k+2,endpoint=True)
	u_ini = time[1:-1]
	u = fsolve(eqn_sys,u_ini)
	u = [0] + list(u) + [1]
	plt.plot(time,u,label=('n='+str(k)))
plt.legend()
plt.xlabel('t')
plt.ylabel('u')
plt.title('Solving a Boundary Value Problem')
plt.show()