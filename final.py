import time
from math import *
import random
import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras import layers, regularizers
from keras.optimizers import Adam, SGD
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, LeakyReLU
from sklearn.metrics import accuracy_score, log_loss
from sklearn.utils import shuffle
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt
from matplotlib import cm

import warnings

warnings.filterwarnings("ignore")

rng = random.Random()
# rng.seed(42)

accuracy_test_temp = []
accuracy_train_temp = []
loss_test_temp = []
loss_train_temp = []

A_test = []
A_train = []
L_test = []
L_train = []

accuracy_test = 0
loss_test = 0
accuracy_train = 0
loss_train = 0


def convolutions(x):
    global accuracy_test_temp
    global accuracy_train_temp
    global loss_test_temp
    global loss_train_temp
    if len(x) == 4:
        batch_size, alpha, beta1, beta2 = x
    if len(x) == 2:
        batch_size, alpha = x
    batch_size = int(batch_size)

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
        # print("orig x_train shape:", x_train.shape)

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
            # model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
            if len(x) == 4:
                model.compile(loss="categorical_crossentropy",
                              optimizer=Adam(learning_rate=alpha, beta_1=beta1, beta_2=beta2), metrics=["accuracy"])
            if len(x) == 2:
                model.compile(loss="categorical_crossentropy", optimizer=SGD(learning_rate=alpha), metrics=["accuracy"])
            # model.summary()
            epochs = 20
            history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, verbose=0)
            model.save("cifar.model")
        preds = model.predict(x_test)
        y_pred = np.argmax(preds, axis=1)
        y_test1 = np.argmax(y_test, axis=1)
        accuracy_test = accuracy_score(y_test1, y_pred)
        loss_test = log_loss(y_test, preds)
        accuracy_train = history.history['accuracy'][-1]
        loss_train = history.history['loss'][-1]
        accuracy_test_temp.append(accuracy_test)
        accuracy_train_temp.append(accuracy_train)
        loss_test_temp.append(loss_test)
        loss_train_temp.append(loss_train)
    return accuracy_test


def population_search(l, u, iters=20, M=2, N=3, neighbourhood=10, condition=50, function=convolutions):
    global accuracy_test_temp
    global accuracy_train_temp
    global loss_test_temp
    global loss_train_temp
    global A_test
    global A_train
    global L_test
    global L_train
    A_test = []
    A_train = []
    L_test = []
    L_train = []
    n = len(l)
    x_sample = [[rng.uniform(l[i], u[i]) for i in range(n)] for j in range(N)]
    X = []
    Y = []
    f_prev = 100
    change = 0
    for k in range(iters):
        accuracy_test_temp = []
        accuracy_train_temp = []
        loss_test_temp = []
        loss_train_temp = []
        x_neighbourhood = x_sample.copy()
        for x in x_sample:
            for m in range(N):
                neighbour = [x[i] + rng.uniform(-(u[i] - l[i]) / neighbourhood, (u[i] - l[i]) / neighbourhood) for i in
                             range(n)]
                if l[0] < neighbour[0] < u[0] and l[1] < neighbour[1] < u[1]:
                    x_neighbourhood.append(neighbour)
        x_dict = {}
        for x in x_neighbourhood:
            x_dict[tuple(x)] = function(x)
        positions = {}
        for k, key in enumerate(list(x_dict.keys())):
            positions[key] = k
        x_dict = {k: v for k, v in sorted(x_dict.items(), key=lambda item: item[1])}
        x_sample = list(x_dict.keys())[-M:]
        A_test.append(accuracy_test_temp[positions[x_sample[-1]]])
        A_train.append(accuracy_train_temp[positions[x_sample[-1]]])
        L_test.append(loss_test_temp[positions[x_sample[-1]]])
        L_train.append(loss_train_temp[positions[x_sample[-1]]])
        f_x = list(x_dict.values())[-1]
        X.append(x_sample[-1])
        # Y.append(f_x)
        if f_prev == f_x:
            change = change + 1
        else:
            f_prev = f_x
            change = 0
        if change > condition:
            break
    return X


def population_search_tuning():
    global A_test
    global A_train
    global L_test
    global L_train

    plt.rcParams['font.size'] = 12
    fig, ax = plt.subplots(2)
    colours = ['hotpink', 'darkviolet', 'deepskyblue', 'mediumseagreen', 'goldenrod']
    i = 0
    for M in [2, 3]:
        print('M ' + str(M))
        for N in [1, 3]:
            print('N ' + str(N))
            population_search([32, 0.001], [256, 0.05], iters=5, M=M, N=N, condition=5, function=convolutions)
            print(A_test)
            print(A_train)
            print(L_test)
            print(L_train)
            ax[0].plot(A_test, c=colours[i], linestyle='-', label=('test M = ' + str(M) + ' N = ' + str(N)))
            ax[0].plot(A_train, c=colours[i], linestyle='--', label=('train M = ' + str(M) + ' N = ' + str(N)))
            ax[1].plot(L_test, c=colours[i], linestyle='-', label=('test M = ' + str(M) + ' N = ' + str(N)))
            ax[1].plot(L_train, c=colours[i], linestyle='--', label=('train M = ' + str(M) + ' N = ' + str(N)))
            i += 1
    # ax.set(ylim=(0, 1))
    ax[0].legend(ncol=2)
    ax[1].legend(ncol=2)
    ax[0].set_xlabel('Iteration')
    ax[0].set_ylabel('Accuracy')
    ax[1].set_xlabel('Iteration')
    ax[1].set_ylabel('Loss')
    plt.show()


def bayesian_optimisation(l, u, iters=20, random=10, function=convolutions):
    global accuracy_test_temp
    global accuracy_train_temp
    global loss_test_temp
    global loss_train_temp
    global A_test
    global A_train
    global L_test
    global L_train
    accuracy_test_temp = []
    accuracy_train_temp = []
    loss_test_temp = []
    loss_train_temp = []

    def convolutions_bayes(batch_size, alpha, beta1=0, beta2=0):
        x = [batch_size, alpha, beta1, beta2] if beta1 > 0 else [batch_size, alpha]
        return function(x)

    bounds = {'batch_size': (l[0], u[0]), 'alpha': (l[1], u[1]), 'beta1': (l[2], u[2]), 'beta2': (l[3], u[3])} if len(
        l) == 4 else {'batch_size': (l[0], u[0]), 'alpha': (l[1], u[1])}

    bayes = BayesianOptimization(f=convolutions_bayes, pbounds=bounds, verbose=1)
    bayes.maximize(n_iter=iters, init_points=random)
    best_params = bayes.max['params']
    best_accuracy = bayes.max['target']

    best_acc = 0
    best_a = 0
    for a, acc in enumerate(accuracy_test_temp):
        if acc > best_acc:
            best_acc = acc
            best_a = a
        A_test.append(best_acc)
        A_train.append(accuracy_train_temp[best_a])
        L_test.append(loss_test_temp[best_a])
        L_train.append(loss_train_temp[best_a])
    return best_accuracy, best_params


def bayesian_optimisation_tuning():
    global A_test
    global A_train
    global L_test
    global L_train

    plt.rcParams['font.size'] = 12
    fig, ax = plt.subplots(2)
    colours = ['hotpink', 'darkviolet', 'deepskyblue', 'mediumseagreen', 'goldenrod']
    i = 0
    for B in [20]:
        print('B ' + str(B))
        for R in [5, 10]:
            print('R ' + str(R))
            bayesian_optimisation([32, 0.001, 0.8, 0.8], [256, 0.05, 1, 1], B, R, function=convolutions)
            print(accuracy_test_temp)
            print(accuracy_train_temp)
            print(loss_test_temp)
            print(loss_train_temp)
            print(A_test)
            print(A_train)
            print(L_test)
            print(L_train)
            ax[0].plot(A_test, c=colours[i], linestyle='-', label=('test B = ' + str(B) + ' R = ' + str(R)))
            ax[0].plot(A_train, c=colours[i], linestyle='--', label=('train B = ' + str(B) + ' R = ' + str(R)))
            ax[1].plot(L_test, c=colours[i], linestyle='-', label=('test B = ' + str(B) + ' R = ' + str(R)))
            ax[1].plot(L_train, c=colours[i], linestyle='--', label=('train B = ' + str(B) + ' R = ' + str(R)))
            i += 1
    # ax.set(ylim=(0, 1))
    ax[0].legend(ncol=2)
    ax[1].legend(ncol=2)
    ax[0].set_xlabel('Iteration')
    ax[0].set_ylabel('Accuracy')
    ax[1].set_xlabel('Iteration')
    ax[1].set_ylabel('Loss')
    plt.show()


def baseline_Adam():
    global accuracy_test_temp
    global accuracy_train_temp
    global loss_test_temp
    global loss_train_temp
    accuracy_test_temp = []
    accuracy_train_temp = []
    loss_test_temp = []
    loss_train_temp = []
    best_params = []
    best_accuracy = 0
    accuracy_history = []
    best_history = []
    for batch_size in [64, 128]:
        for alpha in [0.001, 0.01]:
            for beta1 in [0.8, 0.9]:
                for beta2 in [0.9, 0.999]:
                    params = [batch_size, alpha, beta1, beta2]
                    accuracy = convolutions(params)
                    accuracy_history.append(accuracy)
                    if accuracy > best_accuracy:
                        best_history.append(accuracy)
                        best_accuracy = accuracy
                        best_params = params
    best_acc = 0
    best_a = 0
    for a, acc in enumerate(accuracy_test_temp):
        if acc > best_acc:
            best_acc = acc
            best_a = a
        A_test.append(best_acc)
        A_train.append(accuracy_train_temp[best_a])
        L_test.append(loss_test_temp[best_a])
        L_train.append(loss_train_temp[best_a])
    return best_accuracy, best_params, accuracy_history, best_history


def baseline_constant():
    global accuracy_test_temp
    global accuracy_train_temp
    global loss_test_temp
    global loss_train_temp
    accuracy_test_temp = []
    accuracy_train_temp = []
    loss_test_temp = []
    loss_train_temp = []
    best_params = []
    best_accuracy = 0
    accuracy_history = []
    best_history = []
    for batch_size in [64, 128]:
        for alpha in [0.001, 0.01]:
            params = [batch_size, alpha]
            accuracy = convolutions(params)
            accuracy_history.append(accuracy)
            if accuracy > best_accuracy:
                best_history.append(accuracy)
                best_accuracy = accuracy
                best_params = params
    best_acc = 0
    best_a = 0
    for a, acc in enumerate(accuracy_test_temp):
        if acc > best_acc:
            best_acc = acc
            best_a = a
        A_test.append(best_acc)
        A_train.append(accuracy_train_temp[best_a])
        L_test.append(loss_test_temp[best_a])
        L_train.append(loss_train_temp[best_a])
    return best_accuracy, best_params, accuracy_history, best_history


def baseline(l, u):
    if len(l) == 2:
        return baseline_constant()
    if len(l) == 4:
        return baseline_Adam()


def compare(l, u):
    global A_test
    global A_train
    global L_test
    global L_train
    plt.rcParams['font.size'] = 12
    fig, ax = plt.subplots(2)
    colours = ['hotpink', 'darkviolet', 'deepskyblue', 'mediumseagreen', 'goldenrod']
    labels = ['Population Search, ', 'Bayesian Optimisation, ', 'Baseline, ']
    for f, function in enumerate([population_search, bayesian_optimisation, baseline]):
        A_test = []
        A_train = []
        L_test = []
        L_train = []
        print(function(l, u))
        print(A_test)
        print(A_train)
        print(L_test)
        print(L_train)
        ax[0].plot(A_test, c=colours[f], linestyle='-', label=labels[f] + 'test')
        ax[0].plot(A_train, c=colours[f], linestyle='--', label=labels[f] + 'train')
        ax[1].plot(L_test, c=colours[f], linestyle='-', label=labels[f] + 'test')
        ax[1].plot(L_train, c=colours[f], linestyle='--', label=labels[f] + 'train')

    #ax[1].set(ylim=(1, 10))
    ax[0].legend(ncol=3)
    ax[1].legend(ncol=3)
    ax[0].set_xlabel('Iteration')
    ax[0].set_ylabel('Accuracy')
    ax[1].set_xlabel('Iteration')
    ax[1].set_ylabel('Loss')
    plt.show()


#bayesian_optimisation_tuning()
# population_search_tuning()
#compare([32, 0.001],[256, 0.05])
#compare([32, 0.001,0.7,0.8],[256, 0.05,1,1])
