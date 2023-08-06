import sympy
from sympy import print_latex
import numpy as np
from sympy.plotting import plot
import matplotlib.pyplot as plt
import colorsys

# Obtaining derivative for y = x^4
x = sympy.symbols('x', real=True)
y = x ** 4
dydx = sympy.diff(y, x)
print(y, dydx)

# Calculating and printing some values
y = sympy.lambdify(x, y)
dydx = sympy.lambdify(x, dydx)
est = ((x + 0.01) ** 4 - x ** 4) / 0.01
est = sympy.lambdify(x, est)
i = np.array([1,2,3])
print(y(i),dydx(i), est(i))

# Plotting estimates
deltas = [1,0.5,0.001] #[0.01]
plt.rcParams['font.size'] = 14
fig,ax=plt.subplots(2)
xx = np.arange(-1, 1.01, 0.01)
for i,delta in enumerate(deltas):
    est = ((x + delta) ** 4 - x ** 4) / delta
    est = sympy.lambdify(x, est)
    colour = colorsys.hls_to_rgb(*(0.9 - i * 0.4,0.5,1))
    ax[0].plot(xx, est(xx), color=colour,label = (r'$\delta = $' +str(delta)))
    ax[1].plot(xx, est(xx)-dydx(xx),color = colour)
ax[0].plot(xx, dydx(xx), color='black',linestyle = '--',label = r'$\dfrac{dy}{dx}$')
ax[0].set_xlabel('x')
ax[0].set_ylabel('Derivative')
ax[1].set_xlabel('x')
ax[1].set_ylabel('Error')
ax[0].legend()
plt.show()

# Gradient descent
def gradDescent(f, df, x0, alpha, iters=50):
    x = x0
    X = [x]
    for k in range(iters):
        step = alpha * df(x)
        x = x - step
        X.append(x)
    return X

#Plotting gradient descent
plt.rcParams['font.size'] = 12
fig,ax = plt.subplots(3,2)
for i, x in enumerate([-1.15,1,-0.8]):
    xx = np.arange(-abs(x), abs(x)+0.01, 0.01)
    ax[i,0].plot(xx, y(xx), color='silver',label=(r'$x^4$'))
    for j, c in enumerate([250,25,2.5,1.01,1]):
        alpha = 0.5/x**2/c
        X = gradDescent(y, dydx, x, alpha)
        colour=colorsys.hls_to_rgb(*(0.95-i*0.4,(0.1+0.2*j) ,1))
        ax[i,0].step(X,y(np.array(X)),color=colour,label=(r'$\alpha = %.3f$'%alpha))
        ax[i,0].set_xlabel('x')
        ax[i,0].set_ylabel('f(x)')
        ax[i,1].plot(y(np.array(X)), label=(r'$\alpha = %.2f$'%alpha), color=colour)
        ax[i,1].set_xlabel('x')
        ax[i,1].set_ylabel('Error')
        ax[i,0].legend(bbox_to_anchor=(0., 1.02, 2.2, .102), loc=3,ncol=6, mode="expand", borderaxespad=0)
plt.show()

#Plotting gradient descent for scaled functions
plt.rcParams['font.size'] = 15
x = sympy.symbols('x', real=True)
for i, gamma in enumerate([0.5,1,2]):
    y = gamma*abs(x) #x**2
    dydx = sympy.lambdify(x,sympy.diff(y,x))
    y = sympy.lambdify(x, y)
    xx = np.arange(-1, 1.01, 0.01)
    colour = colorsys.hls_to_rgb(*(0.9 - i * 0.4, 0.8, 1))
    plt.plot(xx, y(xx), color=colour,label = ('gamma = '+str(gamma)))
    X = gradDescent(y, dydx, 1, 0.1)
    colourstep = colorsys.hls_to_rgb(*(0.9 - i * 0.4, 0.3, 1))
    plt.step(X, y(np.array(X)),color=colourstep)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
plt.show()