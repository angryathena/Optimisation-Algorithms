import numpy as np
from math import *
import matplotlib.pyplot as plt
from matplotlib import cm


def generate_trainingdata(m=25):
    return np.array([0, 0]) + 0.25 * np.random.randn(m, 2)


def f(x, minibatch):
    # loss function sum_{w in training data} f(x,w)
    y = 0;
    count = 0
    for w in minibatch:
        z = x - w - 1
        y = y + min(42 * (z[0] ** 2 + z[1] ** 2), (z[0] + 7) ** 2 + (z[1] + 7) ** 2)
        count = count + 1
    return y / count


# Estimating the derivative of f(x,N)
def finiteDifference(x, N, delta=10 ** (-5)):
    g0 = (f([x[0] + delta, x[1]], N) - f([x[0] - delta, x[1]], N)) / (2 * delta)
    g1 = (f([x[0], x[1] + delta], N) - f([x[0], x[1] - delta], N)) / (2 * delta)
    return np.array([g0, g1])


# Computing the values to plot
def makeZ(X, Y, T):
    Z = np.zeros_like(X)
    for i in range(len(X)):
        for j in range(len(X[0])):
            Z[i, j] = f([X[i, j], Y[i, j]], T)
    return Z


# Plotting the wireframe (using a surface plot as it allows colourmaps)
def plot3D(X, Y, Z):
    plt.rcParams['font.size'] = 12
    norm = plt.Normalize(Z.min(), Z.max())
    colors = cm.PuRd_r(norm(Z))

    ax.plot_surface(X, Y, Z, facecolors=colors, shade=False).set_facecolor((0, 0, 0, 0))
    ax.set_xlabel('$x_{0}$')
    ax.set_ylabel('$x_{1}$')
    ax.set_zlabel('f')
    plt.show()


# Creating the contour plot; Generating and plotting the steps
def plotContour(X, Y, Z, colours, lines=1, alphaRange=[0.01, 0.1], batchRange=[5], update='None'):
    plt.rcParams['font.size'] = 12
    fig, ax = plt.subplots(1, 1)
    ax.set(xlim=(-10, 4), ylim=(-10, 4))
    ax.contour(X, Y, Z, levels=30, cmap='PuRd_r')
    for i in range(lines):
        for a, alpha in enumerate(alphaRange):
            for b, batch in enumerate(batchRange):
                steps = SGDminibatch(x0=[3, 3], T=T, alpha0=alpha, batchSize=batch, update=update)
                ax.step(steps[0], steps[1], alpha=0.75, color=colours[a][b], label='batch = ' + str(batch))
    # Uncomment to show the GD behaviour
    '''for a, alpha in enumerate(alphaRange):
        steps = SGDminibatch(x0=[3, 3], T=T, alpha0=alpha, batchSize=25, update='None')
        ax.step(steps[0], steps[1], alpha=0.5, linestyle='--',color='black')'''
    # ax.legend(labels = [r'$\alpha = $'+str(0.01),r'$\alpha = $'+str(0.1)]) (b i,ii)
    ax.legend()
    ax.set_xlabel('$x_{0}$')
    ax.set_ylabel('$x_{1}$')
    plt.show()


def plotLine(T, colours, lines=1, alphaRange=[0.01, 0.1], batchRange=[5], update='None'):
    plt.rcParams['font.size'] = 12
    fig, ax = plt.subplots(1, 1)
    ax.set(xlim=(0, 100), ylim=(0, 300))
    for i in range(lines):
        for a, alpha in enumerate(alphaRange):
            for b, batch in enumerate(batchRange):
                steps = SGDminibatch(x0=[3, 3], T=T, alpha0=alpha, batchSize=batch, update=update)
                steps = zip(steps[0], steps[1])
                Z = [f(s, T) for s in steps]
                ax.plot(Z, c=colours[a][b], label='batch = ' + str(batch))
    # Uncomment to show the GD behaviour
    '''for a, alpha in enumerate(alphaRange):
        steps = SGDminibatch(x0=[3, 3], T=T, alpha0=alpha, batchSize=25, update='None')
        steps = zip(steps[0], steps[1])
        Z = [f(s, T) for s in steps]
        ax.plot(Z, c='black',alpha =0.5,linestyle='--',label = 'GD')'''
    # ax.legend(labels = [r'$\alpha = $'+str(0.01),r'$\alpha = $'+str(0.1)]) #(b i,ii)
    ax.legend()  # b iii
    ax.set_xlabel('Iteration')
    ax.set_ylabel('f')
    plt.show()


# SGD minibatch finds the minimum and returns the steps taken
def SGDminibatch(x0, T, alpha0=0.01, beta1=0.6, beta2=0.1, epsilon=10 ** -3, iters=10000, batchSize=5, update='None'):
    x = np.array(x0)
    X0, X1 = [x[0]], [x[1]]
    n = len(T)
    alpha = alpha0
    sum, step, m, v = [0, 0, 0, 0]
    for k in range(iters):
        np.random.shuffle(T)
        for i in np.arange(0, n, batchSize):
            sample = T[i:i + batchSize]
            gradient = finiteDifference(x, sample)
            if update == 'Adam':
                m = beta1 * m + (1 - beta1) * np.array(gradient) / (1 - beta1 ** (k + 1))  # Adam
                v = beta2 * v + (1 - beta2) * np.dot(np.array(gradient), np.array(gradient)) / (
                        1 - beta2 ** (k + 1))  # Adam
                step = alpha * m / (sqrt(v) + epsilon)  # Adam
            elif update == 'HeavyBall':
                step = beta1 * step + alpha * gradient  # HeavyBall
            else:
                step = alpha * gradient  # Default
            if update == 'Polyak':
                alpha = f(x, sample) / (epsilon + np.dot(gradient, gradient))  # Polyak step
                step = alpha * gradient
            elif update == 'RMSProp':
                sum = beta2 * sum + (1 - beta2) * np.dot(gradient, gradient)  # RMSProp
                alpha = alpha0 / sqrt(sum + epsilon)  # RMSProp
                step = alpha * gradient
            fPrev = f(x, T)
            x = x - step
            print(gradient, np.dot(gradient, gradient), x, fPrev)
            X0.append(x[0])
            X1.append(x[1])
        if abs(fPrev - f(x, T)) < 10 ** (-5):
            break
    return [X0, X1]


T = generate_trainingdata(m=25)
x = np.linspace(-10, 4, 100)
y = np.linspace(-10, 4, 100)
X, Y = np.meshgrid(x, y)
Z = makeZ(X, Y, T)

colours = [['yellowgreen', 'forestgreen', 'mediumaquamarine'], ['gold', 'orange', 'tomato']]  # (b)(i, ii)
# colours = [['gold', 'yellowgreen','forestgreen',]] #(b iii) (c )
# colours = [['tomato'],['orange'],['gold'],['yellowgreen'],['forestgreen'],['mediumaquamarine']] #(b)(i, iv)
plot3D(X, Y, Z)
plotContour(X, Y, Z, colours, lines=1, alphaRange=[0.1], batchRange=[5])
plotLine(T, colours, lines=1, alphaRange=[0.1], batchRange=[5])
