import numpy as np 
import numpy.linalg as la
import matplotlib.pyplot as plt
import math
from scipy.integrate import odeint
import scipy.sparse as sparse

print("The maximum dt allowable for the system if using Euler forward method is 0.5")
A = np.array([[0,0,-5],[1,0,-9.25],[0,1,-6]])
y0 = np.array([3,3,3])
t0 = 0
t_l = t0
t_s = t0
# dt_l = 0.5*(1+np.finfo(float).eps)
# dt_s = 0.5*(1-np.finfo(float).eps)
dt_l = 0.5+0.01
dt_s = 0.5-0.01
y_l = y0
y_s = y0
t_l_ = [t0]
t_s_ = [t0]
y_l_1 = [y_l[0]]
y_l_2 = [y_l[1]]
y_l_3 = [y_l[2]]
y_s_1 = [y_s[0]]
y_s_2 = [y_s[1]]
y_s_3 = [y_s[2]]
while(True):
  t_l += dt_l
  if (t_l>20):
      break
  t_l_.append(t_l)
  y_l = y_l + dt_l*A@y_l
  y_l_1.append(y_l[0])
  y_l_2.append(y_l[1])
  y_l_3.append(y_l[2])

while(True):
  t_s += dt_s
  if (t_s>20):
      break
  t_s_.append(t_s)
  y_s = y_s + dt_s*A@y_s
  y_s_1.append(y_s[0])
  y_s_2.append(y_s[1])
  y_s_3.append(y_s[2])


plt.figure()
plt.plot(t_l_,y_l_1,'g--',label='y1 @dt>0.5 (unstable)')
plt.plot(t_l_,y_l_2,'b--',label='y2 @dt>0.5 (unstable)')
plt.plot(t_l_,y_l_3,'c--',label='y3 @dt>0.5 (unstable)')
plt.plot(t_s_,y_s_1,'r-',label='y1 @dt<0.5 (stable)')
plt.plot(t_s_,y_s_2,'m-',label='y2 @dt<0.5 (stable)')
plt.plot(t_s_,y_s_3,'y-',label='y3 @dt<0.5 (stable)')
plt.xlabel('t')
plt.ylabel('y')
plt.title('Stability of Euler forward method: dt above vs. below critical value')
plt.legend()
plt.grid()
plt.show()