import numpy as np

# sigmoid
def sigmoid(x):
    return 1 / (1 + np.e**(-x))

def sigmoid_prime(x):
    s = sigmoid(x)
    return s * (1 - s)

# tanh
def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    1 - np.tanh(x) ** 2

# ReLU
def relu(x):
    return max(0, x)

def relu_prime(x):
    return 0 if x <= 0 else 1

# softrelu
def softrelu(x):
    return np.log10(1 + np.e**x)

def softrelu_prime(x):
    return sigmoid(x) * 0.4342944819033
