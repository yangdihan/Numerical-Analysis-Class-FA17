import matplotlib.pyplot as plt
import numpy as np

# Tridiagonal solver
def tridiag(alist, blist, clist, flist):
    a = np.array(alist, copy=True)
    b = np.array(blist, copy=True)
    c = np.array(clist, copy=True)
    f = np.array(flist, copy=True)

    n = b.size

    for i in range(n - 1):
        a[i]   = a[i]/b[i]
        b[i+1] = b[i+1] - a[i]*c[i]
        f[i+1] = f[i+1] - a[i]*f[i]

    f[n - 1] = f[n - 1]/b[n - 1]
    for i in range(n - 1, 0, -1):
        f[i-1] = f[i-1] - f[i]*c[i-1]
        f[i-1] = f[i-1]/b[i-1]

    return f

def periodic_tridiag(alist, blist, clist, alpha, beta, flist):
    a = np.array(alist, copy=True)
    b = np.array(blist, copy=True)
    c = np.array(clist, copy=True)
    f = np.array(flist, copy=True)
    n = b.size
    #create last column
    lastCol = np.zeros(n)
    lastCol[0] = alpha
    lastCol[-1] = b[-1]
    lastCol[-2] = c[-1]
    #eliminate a
    for i in range(n - 2):
        mult   = a[i]/b[i]
        b[i+1] -= mult*c[i]
        lastCol[i+1] -= mult*lastCol[i]
        f[i+1] -= mult*f[i]
    #create last row
    lastRow = np.zeros(n)
    lastRow[0] = beta
    lastRow[-1] = b[-1]
    lastRow[-2] = a[-1]
    #last row:
    for i in range(n - 2):
        mult = lastRow[i]/b[i]
        lastRow[i] -= mult*b[i]
        lastRow[i+1] -= mult*c[i]
        lastRow[-1] -= mult*lastCol[i]
        f[-1] -= mult*f[i]
    #last of last row:
    mult = lastRow[-2]/b[-2]
    lastRow[-2] -= mult*b[-2]
    lastRow[-1] -= mult*lastCol[-2]
    f[-1] -= mult*f[-2]
    #sync everything:
    b[-1] = lastRow[-1]
    lastCol[-1] = lastRow[-1]
    c[-1] = lastCol[-2]
    #back substitute:
    f[-1] = f[-1]/b[-1]
    f[-2] = f[-2] - f[-1]*c[-1]
    f[-2] = f[-2]/b[-2]

    for i in range(n - 2, 0, -1):
        f[i-1] = f[i-1] - f[i]*c[i-1] - f[-1]*lastCol[i-1]
        f[i-1] = f[i-1]/b[i-1]

    return f

# A simple routine to test `tridiag`. You can modify this
# to test `periodic_tridiag`. Use with caution
# as this is in no way a complete test.
def test():
    n = 10
    b = 2*np.ones(n)
    a = c = -1* np.ones(n-1)
    f = np.random.rand(n)
    A = np.diag(a, k=-1) + np.diag(b, k=0) + np.diag(c, k = 1)
    my_x = tridiag(a, b, c, f)
    numpy_x = np.linalg.solve(A, f)
    error = np.linalg.norm(my_x - numpy_x, np.inf)

    if error < 10**(-10):
        print("Test non-periodic passed.")
    else:
        print("Test non-periodic failed.")

    al = 0
    bt = 0
    A[0,-1] = al
    A[-1,0] = bt
    my_x = periodic_tridiag(a, b, c, al, bt, f)
    numpy_x = np.linalg.solve(A, f)
    error = np.linalg.norm(my_x - numpy_x, np.inf)
    if error < 10**(-10):
        print("Test pseudo-periodic passed.")
    else:
        print("Test pseudo-periodic failed.")

    al = 50**2
    bt = 500**2
    A[0,-1] = al
    A[-1,0] = bt
    my_x = periodic_tridiag(a, b, c, al, bt, f)
    numpy_x = np.linalg.solve(A, f)
    error = np.linalg.norm(my_x - numpy_x, np.inf)
    print("A=")
    print(A)
    print("f=")
    print(f)
    print("my solution is:")
    print(my_x)
    print("exact solution is:")
    print(numpy_x)
    if error < 10**(-10):
        print("Test periodic passed.")
    else:
        print("Test periodic failed.")

test()

# Lists to keep track of error and h values
hlist = []
errorlist = []

# Driver code to create the plots
for n in [32, 64, 128, 256, 512, 1024, 2048, 4096]:
    h = 1./n
    x = np.linspace(1, n, num=n)*h
    f = np.sin(2*np.pi*x)
    gamma = 0.01

    alpha = beta = -1./h**2
    b = -alpha*(2 + gamma*(h**2))*np.ones(n)
    a =  c = alpha*np.ones(n-1)

    A = np.diag(a, k = -1) + np.diag(b, k = 0) + np.diag(c, k = 1)
    A[0][n-1] = alpha
    A[n-1][0] = beta

    approx_solution = periodic_tridiag(a, b, c, alpha, beta, f)
    exact_solution  = np.sin(2*np.pi*x)/(4*np.pi**2 + gamma)

    hlist.append(h)
    errorlist.append(np.linalg.norm(approx_solution - exact_solution, np.inf))

plt.figure()
plt.loglog(hlist, errorlist, 'o-', label='error')
plt.title('Error variation with $h$')
plt.loglog(hlist, [x**2 for x in hlist], '--', label='h^2')
plt.xlabel('$h$')
plt.ylabel('error')
plt.legend(frameon=True)