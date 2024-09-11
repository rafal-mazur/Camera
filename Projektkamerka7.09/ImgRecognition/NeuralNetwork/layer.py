#  Base class
class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    # Compute output (Y) of the layer given input (X)
    def forward_propagation(self, input_data):
        raise NotImplementedError

    # Update parameters given value of the cost function
    def back_propagation(self, output_error, learning_rate):
        raise NotImplementedError
