import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.optimize import fsolve
from scipy.sparse import diags

def u_final(dx,CFL):
	c = 1
	dt = CFL*dx/c
	c1 = 23/12
	c2 = -16/12
	c3 = 5/12
	nx = int(3/dx)+1
	nt = int(1/dt)+1
	x_ = np.linspace(-1,2,nx,endpoint=True)
	t_ = np.linspace(0,1,nt,endpoint=True)
	k = np.array([-1*np.ones(nx-1),1*np.ones(nx-1)])
	offset = [-1,1]
	C = -1/(2*dx)*diags(k,offset)

	# first 3 guesses
	u_a = np.zeros(nx)
	u_b = np.zeros(nx)
	u_c = np.zeros(nx)
	u_ex = np.zeros(nx)
	for j in range(1,nx):
		u_a[j] = np.exp(-400*(x_[j]+2*dt)**2)
		u_b[j] = np.exp(-400*(x_[j]+dt)**2)
		u_c[j] = np.exp(-400*(x_[j])**2)
		u_ex[j] = np.exp(-400*(x_[j]-1)**2)
	u_0 = u_c

	for t in range(1,nt):
		f_a = C @ u_a
		f_b = C @ u_b
		f_c = C @ u_c
		u_d = u_c + dt*(c1*f_c + c2*f_b + c3*f_a)
		f_a = f_b
		f_b = f_c
		f_c = C @ u_d
		u_a = u_b
		u_b = u_c
		u_c = u_d

	err = max(abs(u_d - u_ex))
	return u_0, u_d, err

CFL = 0.5
err = []
for dx in [0.02,0.01,0.005,0.002]:
	nx = int(3/dx)+1
	x_ = np.linspace(-1,2,nx,endpoint=True)
	ui,uu,e = u_final(dx, CFL)
	err.append(e)
	plt.figure()
	plt.plot(x_,ui,label=('u=0'))
	plt.plot(x_,uu,label=('u=1'))
	plt.xlabel('x')
	plt.ylabel('u')
	plt.title('Advection Equation @ dx='+str(dx)+' ,CFL='+str(CFL))
	plt.legend()
	# plt.show()
plt.figure()
plt.plot([0.02,0.01,0.005,0.002],err)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('dx')
plt.ylabel('error')
plt.title('error at different dx')
plt.grid()
# plt.show()

dx = 0.002
nx = int(3/dx)+1
x_ = np.linspace(-1,2,nx,endpoint=True)
for CFL in [0.7, 0.75]:
	ui,uu,e = u_final(dx, CFL)
	plt.figure()
	plt.plot(x_,ui,label=('u=0'))
	plt.plot(x_,uu,label=('u=1'))
	plt.xlabel('x')
	plt.ylabel('u')
	plt.title('Advection Equation @ dx='+str(dx)+' ,CFL='+str(CFL))
	plt.legend()
	# plt.show()
# u_t2 = u_t1 + dt*(c1*f_t1 + c2*f_t0 + c3*f_tf1)


