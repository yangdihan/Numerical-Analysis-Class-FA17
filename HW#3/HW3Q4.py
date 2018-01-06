# import numpy as np
# import matplotlib.pyplot as plt

# def f(x):
# 	return x**2 -3*x + 2
# def g1(x):
# 	return (x**2 + 2)/3
# def g2(x):
# 	return np.sqrt(3*x - 2)
# def g3(x):
# 	return 3 - 2/x
# def g4(x):
# 	return (x**2 - 2)/(2*x - 3)

# x_guess = 5
# x_acc = 2
# x1 = x_guess
# x2 = x_guess
# x3 = x_guess
# x4 = x_guess
# e1 = []
# e2 = []
# e3 = []
# e4 = []
# for i in range(10):
	
# 	e1.append(np.abs((x1-x_acc)/x_acc))
# 	x1 = g1(x1)
	
# 	e2.append(np.abs((x2-x_acc)/x_acc))
# 	x2 = g2(x2)
	
# 	e3.append(np.abs((x3-x_acc)/x_acc))
# 	x3 = g3(x3)
	
# 	e4.append(np.abs((x4-x_acc)/x_acc))
# 	x4 = g4(x4)

# print('e1')
# print(e1)
# print('e2')
# print(e2)
# print('e3')
# print(e3)
# print('e4')
# print(e4)
# plt.plot(e1,'r')
# plt.rc('text', usetex=True)
# plt.semilogy(e2,'g',label=r'g_{2}(x) & =\sqrt{3x-2}\protect\\')
# plt.semilogy(e3,'b',label=r'g_{3}(x) & =3-\frac{2}{x}\\')
# plt.semilogy(e4,'m',label=r'g_{4}(x) & =\frac{x^{2}-2}{2x-3}\protect\\')
# plt.title('convergence rate of different g-function choice for Fixed Point Problems')
# plt.xlabel('iteration steps')
# plt.ylabel('relative error in x approximation (log scale)')
# plt.legend()
# plt.show()









