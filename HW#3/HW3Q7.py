import numpy as np
import scipy.sparse as ss
import matplotlib.pyplot as plt

def open(filename, mode="r"):
    """
    Only used in the autograder. No need to use this locally.
    """
    try:
        data = data_files[filename]
    except KeyError:
        raise IOError("file not found")

    # 'data' is a 'bytes' object at this point.

    from io import StringIO
    return StringIO(data.decode("utf-8"))


def readmesh(fname):
    """
    Read a mesh file and return vertics as a (npts, 2)
    numpy array and triangles as (ntriangles, 3) numpy
    array. `npts` is the number of vertices of the mesh
    and `ntriangles` is the number of triangles of the
    mesh.
    """

    with open(fname, "r") as f:
        npoints = int(next(f))
        points = np.zeros((npoints, 2))
        for i in range(npoints):
            points[i, :] = [float(x) for x in next(f).split()]

        ntriangles = int(next(f))
        triangles = np.zeros((ntriangles, 3), dtype=int)
        for i in range(ntriangles):
            triangles[i, :] = [int(x)-1 for x in next(f).split()]

    return points, triangles


def plotmesh(points, triangles, tricolors = None):
    """
    Given a list of points (shape: (npts, 2)) and triangles
    (shape: (ntriangles, 3)), plot the mesh.
    """

    plt.figure()
    plt.gca().set_aspect('equal')
    if tricolors is None:
        plt.triplot(points[:, 0], points[:, 1], triangles, 'bo-', lw=1.0)
        plt.show()
    else:
        plt.tripcolor(points[:, 0], points[:, 1], triangles, facecolors=tricolors, edgecolors='k')
        plt.show()
    return


def mesh2dualgraph(triangles):
    """
    Calculate the graph laplacian of the dual graph associated
    with the mesh given by numpy array traingles.
    """

    n, m = triangles.shape

    assert(m == 3), "Triangle should have exactly three points !!"

    G = np.zeros((n, n))

    for i, ti in enumerate(triangles):
        for j, tj in enumerate(triangles):
            ## If there is a common edge
            if len( set(ti) - set(tj) ) == 1:
                G[i, j] = G[j, i] = -1

    for i in range(n):
        G[i, i] = -np.sum(G[i, :])

    return ss.csr_matrix(G)




import scipy.linalg as la
def lanczos(A, x0, iterations):
    # import numpy.linalg as la
    #Implement the body of the function
	n = A.shape[0]
	k = iterations
	Q = np.zeros((n,k))
	T = np.zeros((k,k))
	a = []
	b = []
	# q0 = Q[:,0]
	beta_prev = 0
	Q[:,0] = x0/la.norm(x0,2)
	for i in range(iterations):
	    if (i == 0):
	    	q_prev = np.zeros(n)
	    else:
	    	q_prev = Q[:,i-1]
	    q = Q[:,i]
	    u = A@q
	    alpha = np.conjugate(q)@u
	    a.append(alpha)
	    u = u - beta_prev*q_prev - alpha*q
	    beta = la.norm(u,2)
	    beta_prev = beta
	    if (beta == 0):
	    	break
	    if (i == iterations-1):
	    	break
	    Q[:,i+1] = u/beta
	    b.append(beta_prev)

	T = np.diag(a, 0) + np.diag(b, -1) + np.diag(b, 1)
	if(iterations > 145):
		print(T)
	return Q, T
# def lanczos(A, x0, iterations):
#     #Implement the body of the function
#     k = iterations
#     n = np.shape(A)[0]
#     Q = np.zeros((n,k))
#     T = np.zeros((k,k))
    
#     beta_last = 0
#     Q[:,0]=x0/la.norm(x0)
#     for i in range(k):
#         qk = Q[:,i]
#         try:
#             qklast = Q[:,i-1]
#         except:
#             qklast = np.zeros(n)
            
#         uk = A@qk
#         ak = np.conj(qk).T@uk
#         T[i,i] = ak
#         uk = uk - beta_last*qklast-ak*qk
#         betak = la.norm(uk)
#         beta_last = betak
#         if betak == 0:
#             break
#         try:
#             Q[:,i+1] = uk/betak
#             T[i,i+1] = T[i+1,i] = betak
#         except:
#             break       
#     return Q, T
# def lanczos(A, x0, iterations):
#     #Implement the body of the function
#     n=np.shape(A)[0]
#     m=iterations
#     T=np.zeros((m, m))
#     Q=np.zeros((n, m))
#     A=A.toarray()
#     beta=0
#     x0=x0/np.linalg.norm(x0)
#     q0=0*x0
#     for k in range(m):
#         Q[:,k]=x0
#         u=np.dot(A,x0)
#         alpha=np.dot(np.transpose(x0),u)
#         u=u-beta*q0-alpha*x0
#         beta=np.linalg.norm(u)
#         q0=x0
#         x0=u/beta
#         T[k,k]=alpha
#         if k<m-1:
#             T[k,k+1]=beta
#             T[k+1,k]=beta
#     return Q, T

def fiedler(G, k):
    """
    Calculate the fiedler vector of the graph Laplacian matrix
    `G` using `k` iterations of Lanczos algorithm.
    """
    n, m = G.shape

    assert (n == m), "Matrix should be square !!"

    x0 = np.linspace(1, n, num=n)

    ## You should complete this Lanczos function
    Q, T = lanczos(G, x0, k)
    # print(Q)
    # print(T)
    eVal, eVec = np.linalg.eig(T)
    idx = eVal.argsort()
    eVal = eVal[idx]
    eVec = eVec[:,idx]
    fiedlerVec = np.dot(Q, eVec[:, 1])

    partitionVec = np.zeros_like(fiedlerVec)
    mfeidler = np.ma.median(fiedlerVec)

    for i in range(n):
        if fiedlerVec[i] >= mfeidler:
            partitionVec[i] = 1
        else:
            partitionVec[i] = -1

    return partitionVec

points, triangles = readmesh("mesh.1")
# plotmesh(points, triangles)
G = mesh2dualgraph(triangles)
partitionVec = fiedler(G, 150)
plotmesh(points, triangles, partitionVec)





