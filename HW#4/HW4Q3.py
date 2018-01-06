import numpy as np
import numpy.linalg as la
import scipy.optimize as opt
######## EXAMPLE FOR USING MINIMIZE_SCALAR ##############
# Define function
# def f(x,a):
#     return rosenbrock(x-a*gradient(x))
# Call routine - min now contains the minimum x for the function
# min = opt.minimize_scalar(f,args=(x)).a
def f(a,x):
    return rosenbrock(x-a*gradient(x))
#########################################################

def rosenbrock(x):
    x1 = x[0]
    x2 = x[1]
    return 100*(x2 - x1**2)**2 + (1-x1)**2

# FINISH THIS
def gradient(x):
    # Returns gradient of rosenbrock function at x as numpy array
    x1 = x[0]
    x2 = x[1]
    grad = np.array([200*(x2-x1**2)*(-2*x1)-2*(1-x1),200*(x2-x1**2)])
    return grad

# FINISH THIS
def hessian(x):
    # Returns hessian of rosenbrock function at x as numpy array
    x1 = x[0]
    x2 = x[1]
    hess = [[1200*x1**2-400*x2+2,-400*x1],[-400*x1,200]]
    return hess

# INSERT NEWTON FUNCTION DEFINITION
def NM(x):
    for i in range(10):
        s = la.solve(hessian(x),-gradient(x))
        x = x+s
    return x

# INSERT STEEPEST DESCENT FUNCTION DEFINITION
def SD(x):
    for i in range(10):
        # a = opt.minimize_scalar(f,args=x).x
        a = opt.minimize_scalar(f, args=x).x
        x = x - a*gradient(x)
    return x

# DEFINE STARTING POINTS AND RETURN SOLUTIONS
start1 = np.array([-1.,1.])
start2 = np.array([0.,1.])
start3 = np.array([2.,1.])

nm1 = NM(start1)
nm2 = NM(start2)
nm3 = NM(start3)
sd1 = SD(start1)
sd2 = SD(start2)
sd3 = SD(start3)

# print(nm1)
# print(nm2)
# print(nm3)
# print(sd1)
# print(sd2)
# print(sd3)
