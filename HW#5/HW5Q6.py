import numpy as np 
import numpy.linalg as la
import matplotlib.pyplot as plt
import math
from scipy.integrate import odeint
import scipy.sparse as sparse

V,D = la.eig(A)
print(V)
print(D)
def func(y,t):
    y1=y[0]
    y2=y[1]
    y3=y[2]
    d1=-y1*y2
    d2=y1*y2-5*y2
    d3=5*y2
    return [d1,d2,d3]

t = np.linspace(0, 1, num=50, endpoint=True)
y0 = [95,5,0]
y = odeint(func, y0, t)
y1=y[:,-1]

plt.figure()
plt.plot(t,y[:,0],label='susceptibles' )
plt.plot(t,y[:,1],label='infectives in circulation' )
plt.plot(t,y[:,2],label='infectives removed' )
plt.xlabel('time')
plt.ylabel('number')
plt.title('Modeling Epidemics as ODEs')
plt.legend()
plt.show()


