import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as la
import numpy.linalg as nla
import matplotlib.pyplot as plt
import time

def kron_eval(A, B, C, u):
    ma, na = A.shape
    mb, nb = B.shape
    mc, nc = C.shape

    v = np.zeros((nc, mb, na))
    u = u.reshape((nc, nb, na))

    Bt = B.transpose()

    for k in range(na):
        v[:, :, k] = u[:, :, k] @ Bt

    v = v.reshape((nc, nb*na))
    v = C @ (v)
    
    v = v.reshape((mc*nb, nc))
    v = v @ (A.transpose())
    
    v = v.reshape((ma*mb*mc, 1))
    return v

def generate_Ax(N):
	Ax = (N+1)**2*sp.diags([-1, 2, -1], [-1, 0, 1], shape=(N, N)).toarray()
	return Ax

def generate_A(Ax):
	# N is the size of Ax
	# n = N**3
	# n is the size of A
	# h = 1/(N+1)
	N = len(Ax)
	I = np.identity(N)
	B1 = sp.kron(sp.kron(I,I),Ax)
	B2 = sp.kron(sp.kron(I,Ax),I)
	B3 = sp.kron(sp.kron(Ax,I),I)
	B = (B1+B2+B3).toarray()
	return B


def fast_solve(Ax, f):
    #Implement this as part of step 3
    N = len(Ax)
    h = 1/(N+1)
    Sx = np.zeros((N,N))
    Vx = np.zeros((N,N))
    Ix = np.identity(N)
    for i in range(N):
    	for j in range(N):
    		Sx[i,j] = np.sqrt(2*h)*np.sin(np.pi*h*(i+1)*(j+1))
    		if (i == j):
    			Vx[i,j] = 2/(h**2)*(1-np.cos(np.pi*h*(j+1)))

    # print(Sx)
    Sx_inv = np.transpose(Sx)
    # check_Ax = Sx@Vx@Sx_inv
    # print(check_Ax)
    mid = generate_A(Vx)
    # print(mid.shape)
    mid_inv = np.zeros((N**3,N**3))
    for i in range(N**3):
    	mid_inv[i,i] = 1/mid[i,i]
    c = kron_eval(Sx_inv,Sx_inv,Sx_inv,f)
    b = mid_inv@c
    f = kron_eval(Sx,Sx,Sx,b)
    return f

N = 10
n = 10**3
Ax = generate_Ax(N)
A = generate_A(Ax)
f = np.random.random((n,1))
u_direct = nla.solve(A,f)
u_fast = fast_solve(Ax,f)

print(u_direct-u_fast)
# print()

# compare run
# time_direct = []
# time_fast = []
# size = []
# for i in range(3,10):
# 	N = i+1
# 	n = N**3
# 	Ax = generate_Ax(N)
# 	A = generate_A(Ax)
# 	f = np.random.random((n,1))

# 	start = time.time()
# 	u = nla.solve(A,f)
# 	mid = time.time()
# 	u = fast_solve(Ax,f)
# 	end = time.time()
# 	t_direct = mid-start
# 	t_fast = end-mid
# 	size.append(n)
# 	time_direct.append(t_direct)
# 	time_fast.append(t_fast)

#plot
# plt.figure(0)
# plt.subplot(211)
plt.plot(size,time_direct,'b')
plt.plot(size,time_fast,'r')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('n')
plt.ylabel('time(s)')
plt.title('blue direct solver vs. red fast solver')
plt.show()





