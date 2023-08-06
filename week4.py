# function: 9*(x-8)^4+8*(y-9)^2
# function: Max(x-8,0)+8*|y-9|
import time

import sympy
from sympy import Max, Abs
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from math import *

# Obtaining derivatives
x0, x1 = sympy.symbols('x0, x1', real=True)
x = sympy.Array([x0, x1])

f1 = 9 * (x[0] - 8) ** 4 + 8 * (x[1] - 9) ** 2
df1dx = sympy.diff(f1, x)
f1 = sympy.lambdify(x, f1)
df1dx = sympy.lambdify(x, df1dx)

f2 = Max(x[0] - 8, 0) + 8 * Abs(x[1] - 9)
df2dx = sympy.diff(f2, x)
f2 = sympy.lambdify(x, f2)
df2dx = sympy.lambdify(x, df2dx)

x = sympy.symbols('x', real=True)
f3 = Max(x, 0)
df3dx = sympy.diff(f3, x)
f3 = sympy.lambdify(x, f3)
df3dx = sympy.lambdify(x, df3dx)


def gradDescentReLu(f, df, x0, alpha0, beta1, beta2, epsilon=0.00001, iters=5000, update='none'):
    x = x0
    X = [x]
    alpha = alpha0
    sum, step, m, v = [0, 0, 0, 0]
    fPrev = f(x)

    for k in range(iters):
        if update == 'Adam':
            m = beta1 * m + (1 - beta1) * np.array(df(x)) / (1 - beta1 ** (k + 1))  # Adam
            v = beta2 * v + (1 - beta2) * np.dot(np.array(df(x)), np.array(df(x))) / (1 - beta2 ** (k + 1))  # Adam
            step = alpha * m / (sqrt(v) + epsilon)  # Adam
        elif update == 'HeavyBall':
            print(step)
            step = beta1 * step + alpha * np.array(df(x))  # HeavyBall
        else:
            step = alpha * np.array(df(x))  # Default
        if update == 'Polyak':
            alpha = np.array(f(x)) / (epsilon + np.dot(np.array(df(x)), np.array(df(x))))  # Polyak step
        elif update == 'RMSProp':
            sum = beta1 * sum + (1 - beta1) * np.dot(np.array(df(x)), np.array(df(x)))  # RMSProp
            alpha = alpha0 / sqrt(sum + epsilon)  # RMSProp
        fPrev = f(x)
        x = x - step
        X.append(x)
    return X


def gradDescent(f, df, x0, alpha0, beta1, beta2, epsilon=0, iters=5000, update='none'):
    x = np.array(x0)
    X0, X1 = [x[0]], [x[1]]
    alpha = alpha0
    sum, step, m, v = [0, 0, 0, 0]
    fPrev = f(*x)
    print(update)
    for k in range(iters):
        if update == 'Adam':
            m = beta1 * m + (1 - beta1) * np.array(df(*x)) / (1 - beta1 ** (k + 1))  # Adam
            v = beta2 * v + (1 - beta2) * np.dot(np.array(df(*x)), np.array(df(*x))) / (1 - beta2 ** (k + 1))  # Adam
            step = alpha * m / (sqrt(v) + epsilon)  # Adam
        elif update == 'HeavyBall':
            step = beta1 * step + alpha * np.array(df(*x))  # HeavyBall
        else:
            step = alpha * np.array(df(*x))  # Default
        if update == 'Polyak':
            alpha = np.array(f(*x)) / np.dot(np.array(df(*x)), np.array(df(*x)))  # Polyak step
        elif update == 'RMSProp':
            sum = beta1 * sum + (1 - beta1) * np.dot(np.array(df(*x)), np.array(df(*x)))  # RMSProp
            alpha = alpha0 / sqrt(sum + epsilon)  # RMSProp
        fPrev = f(*x)
        x = x - step
        X0.append(x[0])
        X1.append(x[1])
        if abs(fPrev - f(*x)) < 0.00001:
            break
    return [X0, X1]


def plot3D(X, Y, Z):
    norm = plt.Normalize(Z.min(), Z.max())
    colors = cm.PuRd_r(norm(Z))
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, facecolors=colors, shade=False).set_facecolor((0, 0, 0, 0))
    ax.set_xlabel('$x_{0}$')
    ax.set_ylabel('$x_{1}$')
    ax.set_zlabel('f')
    plt.show()


def plotSteps(f, df, x0, linestyle, colour, update='none'):
    if f == f1:
        if update == 'RMSProp':
            beta1List = [0.25, 0.9]
            beta2List = [0]
            alphaList = [0.00025, 0.025, 0.25, 0.5]
        elif update == 'HeavyBall':
            beta1List = [0.25, 0.9]
            beta2List = [0]
            alphaList = [0.002, 0.0035, 0.1, 0.3]
        elif update == 'Adam':
            beta1List = [0.25, 0.9]
            beta2List = [0.25, 0.9]
            alphaList = [0.01, 0.025]
        else:
            beta1List = [0]
            beta2List = [0]
            alphaList = [0.01]
    else:
        if update == 'RMSProp':
            beta1List = [0.25, 0.9]
            beta2List = [0]
            alphaList = [0.00025, 0.08, 0.25, 0.8]
        elif update == 'HeavyBall':
            beta1List = [0.25, 0.9]
            beta2List = [0]
            alphaList = [0.00001, 0.0035, 0.1, 0.3]
        elif update == 'Adam':
            beta1List = [0.25, 0.9]
            beta2List = [0.25, 0.9]
            alphaList = [0.025, 0.8]
        else:
            beta1List = [0]
            beta2List = [0]
            alphaList = [0.01]
    for b1, beta1 in enumerate(beta1List):
        for a, alpha in enumerate(alphaList):
            for b2, beta2 in enumerate(beta2List):
                start = time.time() * 1000
                steps = gradDescent(f, df, x0, alpha, beta1, beta2, update=update)
                stop = time.time() * 1000
                ax.step(steps[0], steps[1], linestyle=line[b1], color=colour[2 * a + b2][b1], alpha=0.75, label=(
                            r'$\beta_{1} = $' + str(beta1) + r' $\beta_{2} = $' + str(beta2) + r' $\alpha = $' + str(
                        alpha) + r' $time = %.2f ms$' % (stop - start)))  #


def plotReLu(x0, f, df, colours):
    plt.rcParams['font.size'] = 12
    fig, ax = plt.subplots()
    x = np.linspace(-x0, x0, 3)
    y = np.maximum(x, 0)
    # print(xx)
    # ax.plot(x, y, 'r')
    ax.plot(x, np.maximum(x, 0), color='silver', label=(r'$Max(x,0)$'))
    alpha = [0.02, 0.007, 0.035, 0.25, 0.025]
    beta1 = [0.01, 0.01, 0.9, 0.9, 0.9]
    beta2 = [0.01, 0.01, 0.01, 0.01, 0.9]
    ax.set_xlim(-10, 1.1)
    ax.set_ylim(-0.1, 1.1)
    for u, update in enumerate(['none', 'Polyak', 'RMSProp', 'HeavyBall', 'Adam']):
        X = gradDescentReLu(f, df, x0, alpha[u], beta1[u], beta2[u], update=update)
        ax.step(X, np.maximum(X, 0), label=update)

    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.legend()
    plt.show()


#fig, ax = plt.subplots()
colour = [['salmon', 'firebrick'], ['gold', 'darkgoldenrod'], ['limegreen', 'green'], ['lightskyblue', 'steelblue']]
line = ['-', '--']

x0, df = [8.75, 9.75], df1dx  # f1
x = np.linspace(7.5, 9, 30)
y = np.linspace(8.5, 10, 30)
X, Y = np.meshgrid(x, y)

f = f3
df = df3dx

# Z = f(X, Y) #f1
# Z = np.maximum(X - 8, 0) + 8 * np.abs(Y - 9) #f1

'''ax.contour(X, Y, Z, levels=30, cmap='PuRd_r')
plotSteps(f,df,x0,line,colour,update ='Adam') #Select the update

ax.legend(ncol=2)
ax.set_xlabel('$x_{0}$')
ax.set_ylabel('$x_{1}$')
ax.set_xlim(7.5, 9)
ax.set_ylim(8.5, 10)
plt.show()'''

# plot3D(X,Y,Z)

plotReLu(100, f, df, colour)
