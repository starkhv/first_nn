from math import exp
import numpy as np
import random

def ftn(inp):
    var2 = 33
    var6 = 54
    temp = inp*var2%var6
    if temp>10:
        val =  (temp%23+20)
    else:
        val = (temp)
    return val
    # return (val+19)/rang

def generate_dataset():
    binary = lambda x: np.matrix([int(d) for d in format(x, '021b')])
    while(True):
        inp = random.randint(0, 1000000)
        out = ftn(inp)
        yield (binary(inp), binary(out))
        # yield (np.array((inp,)), np.array((inp,)))

def v_generate_dataset():
    X = [random.random() for i in range(100000)]
    y = [i for i in range(10000)]
    # for i in range(1000000):
        # inp = random.randint(0, 100)
        # X.append(inp)
        # y.append(inp)
    return (X,y)

# def sigmoid(inp):
    # if inp<0:
        # return 1 - sigmoid(-inp)
    # else:
        # return 1/(1+exp(-inp))

# def init_nn_layer(num_inputs, num_neurons):
    # """ Initialize layer weights to small random values """
    # layer1_weights = np.random.rand(num_inputs+1, num_neurons)
    # return layer1_weights

# def compute_activations(inputs, weights):
    # fn_vec = np.vectorize(sigmoid)
    # return fn_vec(np.dot(inputs, weights))

# def compute_cost(outputs, predictions):
    # return np.sum((outputs-predictions)**2, axis=(0,1))

# def compute_predictions(inputs, weights_list):
    # num_layers = len(weights_list)
    # for i in range(num_layers):
        # if i == 0:
            # activations = inputs
        # else:
            # activations = np.hstack((np.ones(activations.shape[0],1), activations))
        # activations = compute_activations(activations, weights_list[i])
    # return activations
