import scipy.interpolate as itp
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

def f1(x):
    return 1/(1+25*x**2)

def f2(x):
    return np.exp(np.cos(x))

def uni(start,end,number):
    return np.linspace(start, end, num=number, endpoint=True)

def chb(start,end,number):
    i = np.arange(1,number+1)
    theta = np.pi*(2*i-1)/(2*number)
    t = np.cos(theta)
    out = start+(end-start)*(t+1)/2
    return out


x_uni_itp = uni(-1,1,12)
y_uni_itp = f1(x_uni_itp)
x_chb_itp = chb(-1,1,12)
y_chb_itp = f1(x_chb_itp)
#lagrange uniform
l_uni = itp.lagrange(x_uni_itp, y_uni_itp)
#cubic spine uniform
c_uni = itp.CubicSpline(x_uni_itp, y_uni_itp)
#lagrange chebyshev
l_chb = itp.lagrange(x_chb_itp, y_chb_itp)

#plot
x_uni_plt = uni(-1,1,120)
l_uni_plt = l_uni(x_uni_plt)
c_uni_plt = c_uni(x_uni_plt)
x_chb_plt = chb(-1,1,120)
l_chb_plt = l_chb(x_chb_plt)

plt.figure(0)
plt.plot(x_uni_plt, l_uni_plt, label='uniform Lagrange')
plt.plot(x_uni_plt, c_uni_plt, label='uniform Cubic Spine')
plt.plot(x_chb_plt, l_chb_plt, label='Chebyshev Lagrange')
plt.xlabel('x')
plt.ylabel('interpolation value')
plt.legend()
plt.title('three interpolation stretagies for Runge funtion')
plt.show()

# for f1
e1_lagrange_uni = []
e1_cubicSpline_uni = []
e1_lagrange_chb = []
number1 = []
for num in range(4,51):
    x1_uni_itp = uni(-1,1,num)
    y1_uni_itp = f1(x1_uni_itp)
    x1_chb_itp = chb(-1,1,num)
    y1_chb_itp = f1(x1_chb_itp)
    #lagrange uniform
    l1_uni = itp.lagrange(x1_uni_itp, y1_uni_itp)
    #cubic spine uniform
    c1_uni = itp.CubicSpline(x1_uni_itp, y1_uni_itp)
    #lagrange chebyshev
    l1_chb = itp.lagrange(x1_chb_itp, y1_chb_itp)

    x1_uni_plt = uni(-1,1,num*10)
    f1_uni_plt = f1(x1_uni_plt)
    l1_uni_plt = l1_uni(x1_uni_plt)
    c1_uni_plt = c1_uni(x1_uni_plt)
    e1_l_uni = la.norm(l1_uni_plt-f1_uni_plt)
    e1_lagrange_uni.append(e1_l_uni)
    e1_cs_uni = la.norm(c1_uni_plt-f1_uni_plt)
    e1_cubicSpline_uni.append(e1_cs_uni)
    x1_chb_plt = chb(-1,1,num*10)
    f1_chb_plt = f1(x1_chb_plt)
    l1_chb_plt = l1_chb(x1_chb_plt)
    e1_l_chb = la.norm(l1_chb_plt-f1_chb_plt)
    e1_lagrange_chb.append(e1_l_chb)
    number1.append(num)

plt.figure(1)
plt.plot(number1,e1_lagrange_uni,label='error of uniform Lagrange interpolation')
plt.plot(number1,e1_cubicSpline_uni,label='error of uniform Cubic Spline')
plt.plot(number1,e1_lagrange_chb,label='error of Chebyshev Lagrange interpolation')
plt.yscale('log')
plt.xlabel('number of interpolation points')
plt.ylabel('interpolation error')
plt.legend()
plt.title('Error of three interpolation stretagies for Runge funtion 1')
plt.show()




# for f2
e2_lagrange_uni = []
e2_cubicSpline_uni = []
e2_lagrange_chb = []
number2 = []
for num in range(4,51):
    x2_uni_itp = uni(0,2*np.pi,num)
    y2_uni_itp = f2(x2_uni_itp)
    x2_chb_itp = chb(0,2*np.pi,num)
    y2_chb_itp = f2(x2_chb_itp)
    #lagrange uniform
    l2_uni = itp.lagrange(x2_uni_itp, y2_uni_itp)
    #2ubic spine uniform
    c2_uni = itp.CubicSpline(x2_uni_itp, y2_uni_itp)
    #2agrange chebyshev
    l2_chb = itp.lagrange(x2_chb_itp, y2_chb_itp)

    x2_uni_plt = uni(0,2*np.pi,num*10)
    f2_uni_plt = f2(x2_uni_plt)
    l2_uni_plt = l2_uni(x2_uni_plt)
    c2_uni_plt = c2_uni(x2_uni_plt)
    e2_l_uni = la.norm(l2_uni_plt-f2_uni_plt)
    e2_lagrange_uni.append(e2_l_uni)
    e2_cs_uni = la.norm(c2_uni_plt-f2_uni_plt)
    e2_cubicSpline_uni.append(e2_cs_uni)
    x2_chb_plt = chb(0,2*np.pi,num*10)
    f2_chb_plt = f2(x2_chb_plt)
    l2_chb_plt = l2_chb(x2_chb_plt)
    e2_l_chb = la.norm(l2_chb_plt-f2_chb_plt)
    e2_lagrange_chb.append(e2_l_chb)
    number2.append(num)


plt.figure(2)
plt.plot(number2,e2_lagrange_uni, label='error of uniform Lagrange interpolation')
plt.plot(number2,e2_cubicSpline_uni, label='error of uniform Cubic Spline')
plt.plot(number2,e2_lagrange_chb, label='error of Chebyshev Lagrange interpolation')
plt.yscale('log')
plt.xlabel('number of interpolation points')
plt.ylabel('interpolation error')
plt.legend()
plt.title('Error of three interpolation stretagies for Runge funtion 2')
plt.show()