import numpy as np
from layer import Layer

class FCLayer(Layer):
    # input_size: number of input neurons
    # output_size: number of output neurons
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.weights: np.ndarray = 2 * np.random.rand(input_size, output_size) - 1
        self.bias: np.ndarray = 2 * np.random.rand(1, output_size)

    def forward_propagation(self, input_data: np.array) -> None:
        self.input = input_data
        self.output = np.dot(self.input, self.weights)

    def back_propagation(self, output_error, learning_rate) -> None:
        pass
