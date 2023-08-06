# function: 9*(x-8)^4+8*(y-9)^2
# function: Max(x-8,0)+8*|y-9|

import time
from math import *
import random
import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras import layers, regularizers
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, LeakyReLU
from sklearn.metrics import  accuracy_score
from sklearn.utils import shuffle

import matplotlib.pyplot as plt
from matplotlib import cm

rng = random.Random()
rng.seed(42)


def convolutions(x):
    batch_size, epochs, alpha, beta1, beta2 = x
    batch_size, epochs = int(batch_size), int(epochs)
    '''plt.rc('font', size=18)
    plt.rcParams['figure.constrained_layout.use'] = True'''
    import sys

    # Model / data parameters
    num_classes = 10
    input_shape = (32, 32, 3)

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    n = 5000
    x_train = x_train[1:n];
    y_train = y_train[1:n]

    with tf.device('/device:GPU:0'):
        # Scale images to the [0, 1] range
        x_train = x_train.astype("float32") / 255
        x_test = x_test.astype("float32") / 255
        #print("orig x_train shape:", x_train.shape)

        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)

        use_saved_model = False
        if use_saved_model:
            model = keras.models.load_model("cifar.model")
        else:
            model = keras.Sequential()
            model.add(Conv2D(16, (3, 3), padding='same', input_shape=x_train.shape[1:], activation='relu'))
            model.add(Conv2D(16, (3, 3), strides=(2, 2), padding='same', activation='relu'))
            model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
            model.add(Conv2D(32, (3, 3), strides=(2, 2), padding='same', activation='relu'))
            model.add(Dropout(0.5))
            model.add(Flatten())
            model.add(Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l1(0.0001)))
            model.compile(loss="categorical_crossentropy",
                          optimizer=Adam(learning_rate=alpha, beta_1=beta1, beta_2=beta2), metrics=["accuracy"])
            model.summary()
            history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
            model.save("cifar.model")
            '''plt.subplot(211)
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper left')
            plt.subplot(212)
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss');
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper left')
            plt.show()'''
        preds = model.predict(x_test)
        y_pred = np.argmax(preds, axis=1)
        y_test1 = np.argmax(y_test, axis=1)
        accuracy = accuracy_score(y_test1, y_pred)
    return -accuracy

# (Un)comment to select function
def f(x):
    # y = max(x[0] - 8, 0) + 8 * abs(x[1] - 9)
    y = 9 * (x[0] - 8) ** 4 + 8 * (x[1] - 9) ** 2
    return y


def finite_difference(x, delta=10 ** (-5)):
    g0 = (f([x[0] + delta, x[1]]) - f([x[0] - delta, x[1]])) / (2 * delta)
    g1 = (f([x[0], x[1] + delta]) - f([x[0], x[1] - delta])) / (2 * delta)
    return np.array([g0, g1])


def global_random_search(l, u, N=4200, function=f):
    n = len(l)
    best_x = [rng.uniform(l[i], u[i]) for i in range(n)]
    best_f = function(best_x)
    # X0, X1 = [best_x[0]], [best_x[1]]
    X = [best_x]
    Y = [best_f]
    for k in range(N):
        x = [rng.uniform(l[i], u[i]) for i in range(n)]
        f_x = function(x)
        if f_x < best_f:
            best_f = f_x
            best_x = x
            # X0.append(x[0])
            # X1.append(x[1])
        X.append(best_x)
        Y.append(best_f)
        print(k, Y)
    return X, Y  # [X0, X1], Y


def population_search(l, u, iters=4200, M=10, N=2, neighbourhood=0.5, condition=50, function=f):
    n = len(l)
    x_sample = [[rng.uniform(l[i], u[i]) for i in range(n)] for j in range(N)]
    #X0, X1 = [], []
    X = []
    Y = []
    f_prev = 100
    change = 0
    for k in range(iters):
        x_neighbourhood = x_sample.copy()
        for x in x_sample:
            for m in range(N):
                neighbour = [x[i] + rng.uniform(-(u[i]-l[i])/10, (u[i]-l[i])/10) for i in range(n)]
                if l[0] < neighbour[0] < u[0] and l[1] < neighbour[1] < u[1]:
                    x_neighbourhood.append(neighbour)
        x_sample = sorted(x_neighbourhood, key=lambda x: function(x))[:M]
        f_x = function(x_sample[0])
        #X0.append(x_sample[0][0])
        #X1.append(x_sample[0][1])
        X.append(x_sample[0])
        Y.append(f_x)
        if f_prev == f_x:
            change = change + 1
        else:
            f_prev = f_x
            change = 0
        if change > condition:
            break
        print(k, Y)
    return X, Y#[X0, X1], Y


def gradient_descent(x0, alpha, condition=-5, iters=4200):
    x = np.array(x0)
    f_x = f(x)
    X0, X1 = [x[0]], [x[1]]
    Y = [f_x]
    for k in range(iters):
        gradient = finite_difference(x)
        step = alpha * gradient  # Default
        f_prev = f_x
        x = x - step
        f_x = f(x)
        X0.append(x[0])
        X1.append(x[1])
        Y.append(f_x)
        if abs(f_prev - f_x) < 10 ** (condition):
            break
    return [X0, X1], Y


def makeZ(X, Y):
    Z = np.zeros_like(X)
    for i in range(len(X)):
        for j in range(len(X[0])):
            Z[i, j] = f([X[i, j], Y[i, j]])
    return Z


def plot3D(X, Y, Z):
    plt.rcParams['font.size'] = 12
    norm = plt.Normalize(Z.min(), Z.max())
    colors = cm.PuRd_r(norm(Z))
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, facecolors=colors, shade=False).set_facecolor((0, 0, 0, 0))
    ax.set_xlabel('$x_{0}$')
    ax.set_ylabel('$x_{1}$')
    ax.set_zlabel('f')
    plt.show()


def plotContour(X, Y, Z):
    plt.rcParams['font.size'] = 9
    fig, ax = plt.subplots(1, 1)
    ax.contour(X, Y, Z, levels=30, alpha=0.5, cmap='PuRd_r')
    tic = time.time_ns()
    steps, line = global_random_search([6.5, 7], [9.5, 11])
    toc = time.time_ns()
    ax.step(steps[0], steps[1], color='gold', label='GRS ' + str((toc - tic) // 1000 / 100) + ' ms')
    tic = time.time_ns()
    steps, line = gradient_descent([9, 10], 0.01, condition=-5)
    toc = time.time_ns()
    ax.step(steps[0], steps[1], color='yellowgreen', label='GD ' + str((toc - tic) // 10000 / 100) + ' ms')
    colours = ['hotpink', 'mediumorchid']
    lines = ['-', '--']
    for n, N in enumerate([2, 20]):
        for m, M in enumerate([10, 100]):
            tic = time.time_ns()
            steps, line = population_search([6.5, 7], [9.5, 11], N=N, M=M)
            toc = time.time_ns()
            ax.step(steps[0], steps[1], color=colours[n], linestyle=lines[m],
                    label=str(M) + ' x ' + str(N) + ' ' + str((toc - tic) // 10000 / 100) + ' ms')
    ax.legend(ncol=3)
    ax.set_xlabel('$x_{0}$')
    ax.set_ylabel('$x_{1}$')
    plt.show()


def compare_gd_grs():
    clock = []
    function = []
    for N in range(10000, -1000, -1000):
        clock_row = []
        function_row = []
        for p in range(-10, 1, 1):
            tic = time.time_ns() / 1000000
            x_grs, y_grs = global_random_search([6.5, 7], [9.5, 11], N=N)
            toc = time.time_ns() / 1000000
            grs = (toc - tic)
            tic = time.time_ns() / 1000000
            x_gd, y_gd = gradient_descent([9, 10], 0.01, condition=p, iters=N)  # 0.01 f1, 0.02 f2
            toc = time.time_ns() / 1000000
            gd = (toc - tic)
            function_row.append(log10(y_grs[-1]) - log10(y_gd[-1]))
            # function_row.append((y_grs[-1] - y_gd[-1]))
            clock_row.append(grs - gd)
        clock.append(clock_row)
        function.append(function_row)
    for grid in [clock]:
        plt.rcParams['font.size'] = 9
        plt.imshow(grid, cmap='PiYG', vmin=-4, vmax=4)
        plt.xlabel('Gradient Descent stopping condition')
        tick_vals = range(0, 11)
        tick_labels = [r'$10^{%s}$' % i for i in range(-10, 1)]
        plt.xticks(ticks=tick_vals, labels=tick_labels)

        plt.ylabel('Iterations')
        tick_vals = range(0, 11)
        tick_labels = [i for i in range(10000, -1000, -1000)]
        plt.yticks(ticks=tick_vals, labels=tick_labels)

        plt.colorbar()
        plt.show()


x = np.linspace(6.5, 9.5, 30)
y = np.linspace(7, 11, 30)
X, Y = np.meshgrid(x, y)
Z = makeZ(X, Y)
# plot3D(X, Y, Z)
#plotContour(X, Y, Z)
# compare_gd_grs()

tic = time.time()
print(global_random_search([32, 10, 0.0001, 0, 0], [256, 40, 0.5, 1, 1], N=40, function=convolutions))
#print(population_search([32, 10, 0.0001, 0, 0], [256, 40, 0.5, 1, 1], iters=10, M=2, N=2, condition=5, function=convolutions))
toc = time.time()
print(toc-tic)
