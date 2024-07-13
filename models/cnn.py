import numpy as np


# activation classes have two functions: fn(z) and prime(z) which return the activation function and its derivative, respectively.
class activation(object):
    def fn(z):
        return z

    def prime(z):
        return z


class relu(activation):
    def fn(z):
        return np.maximum(0, z)

    def prime(z):
        return (fn(z) + 0.00000001)/(fn(z) + 0.00000001)


class sigmoid(activation):
    def fn(z):
        return 1/(1+np.exp(-z))

    def prime(z):
        return sigmoid.fn(z)*(1.0-sigmoid.fn(z))


class Cost(object):
    def fn(a, y):
        return y-a

    def prime(a, y):
        return -1


class Quadratic(Cost):
    def fn(a, y):
        return 0.5*np.linalg.norm(a-y)**2

    def delta(a, y):
        return a-y


class CrossEntropy(Cost):
    def fn(a, y):
        pass

    def delta(a, y):
        return a-y


def default_weight_initializer(input_shape, output_shape):
    return np.random.randn(output_shape, input_shape)


def he_weight_initializer(input_shape, output_shape):
    return default_weight_initializer(input_shape, output_shape) * np.sqrt(2/input_shape)


# used for tanh activation.
def xavier_weight_initializer(input_shape, output_shape):
    return default_weight_initializer(input_shape, output_shape) * np.sqrt(1/input_shape)


class CNN(object):
    # input shape for a single training example, a **tuple**.
    def __init__(self, input_shape):
        self.model = []
        self.input_shape = input_shape

    def add(self, layer):
        self.model.append(layer)

    def initializer(self, input_shape):  # input_shape for a single input (no batch dim)
        # for layer in self.model:
        # input_shape = layer.initializer(input_shape)
        for i in range(len(self.model)):
            input_shape = self.model[i].initializer(input_shape)

    def SGD(self, train_data, mini_batch_size, epochs, lr, validation_data=None):
        if validation_data != None:
            test_input, test_label = validation_data
        self.initialize()
        for i in range(epochs):
            random.shuffle(train_data)
            mini_batches = [zip(*train_data[i:i+mini_batch_size])
                            for i in range(0, len(train_data), mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, lr)
            print(f"Epoch {i+1} complete")
            if validation_data != None:
                print("Accuracy (%):", self.evaluate(
                    np.array(test_input), np.array(test_label)))

    def update_mini_batch(self, mini_batch, lr):
        train_input, train_label = mini_batch
        a = np.array(train_input)
        for layer in self.model:
            a = layer.feedforward(a)
        self.model[-1].set_labels(np.array(train_label))
        for layer in self.model[::-1]:
            a = layer.backprop(a, lr)

    def initialize(self):
        self.initializer(self.input_shape)

    def feedforward(self, a):
        for layer in self.model:
            a = layer.feedforward(a)
        return a

    def backprop(self, dCda, lr):
        for layer in self.model[::-1]:
            dCda = layer.backprop(dCda, lr)
        return dCda

    def evaluate(self, test_input, test_label):  # test_data is batched.
        return sum([int(np.argmax(x) == y) for x, y in zip(self.feedforward(test_input), test_label)])/len(test_input)*100


class Layer(object):
    activations = {
        "default": activation,
        "relu": relu,
        "sigmoid": sigmoid
    }

    def __init__(self):
        pass

    def feedforward(self, a):
        self.set_input(a)
        return a

    def backprop(self, dCda, lr):
        return dCda

    # input_shape and output_shape are for single inputs, ignoring batches.
    def initializer(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = self.get_output_shape(input_shape)
        return self.output_shape

    def set_input(self, a):
        self.a = a

    def get_output_shape(self, input_shape):
        return input_shape


class ConvLayer(Layer):
    # still need to implement zero padding
    def __init__(self, kernel_size, activation, stride=1, zero_padding='valid'):
        """kernel_size - a list of length 2 containing integers representing the x and y sizes of the filter"""
        super().__init__()
        self.activation = super().activations[activation]
        self.kernel_size = kernel_size
        self.stride = stride
        self.zero_padding = zero_padding

    # a - a numpy array of dimensionality 4: (batch size, image x, image y, color channel)
    def feedforward(self, a):
        self.set_input(a)
        output = np.zeros([a.shape[0]] + [int((i-k)/self.stride) + 1 for i,
                          k in zip(a.shape[1:3], self.kernel_size)] + [a.shape[3]])
        a = np.transpose(a, (1, 2, 0, 3))
        # transpose output to (image x, image y, batch size, color channel) in order to set values in batches
        output = np.transpose(output, (1, 2, 0, 3))
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                output[i][j] = np.einsum("ijkl, ij -> kl", a[i*self.stride:i*self.stride +
                                         self.kernel_size[0], j*self.stride:j*self.stride+self.kernel_size[1]], self.filter)
        self.z = np.transpose(output, (2, 0, 1, 3)) + self.biases
        return self.activation.fn(self.z)

    # dconv is the matrix of partial derivatives of shape (batch size, conv x, conv y, color channel), conv x and conv y are the sizes of the convoluted output from the layer's feedforward.
    def backprop(self, dconv, lr):
        # delta is the derivative of the cost with respect to z.
        delta = dconv * self.activation.prime(self.z)
        dCdb = delta
        # initialize shape of filter gradient with shape (filter x, filter y, batch size)
        dCdF = np.zeros(self.filter.shape + (delta.shape[0],))
        delta, a = (np.transpose(delta, (1, 2, 0, 3)), np.transpose(
            self.a, (1, 2, 0, 3)))  # reshape to dimensions (x, y, batch, color channel)
        dCda = np.zeros(a.shape)
        for i in range(delta.shape[0]):
            for j in range(delta.shape[1]):
                x, y = (i*self.stride, j*self.stride)
                # a is the orginal input to the layer.
                dCdF += np.sum(delta[i][j] * a[x:x+self.kernel_size[0],
                               y:y+self.kernel_size[1]], axis=3)
                dCda[x:x+self.kernel_size[0], y:y+self.kernel_size[1]
                     ] += np.einsum("ij, kl -> ijkl", self.filter, delta[i][j])
        self.filter = self.filter - \
            (lr/dCdF.shape[2])*np.sum(np.transpose(dCdF, (2, 0, 1)), axis=0)
        self.biases = self.biases - (lr/dCdb.shape[0])*np.sum(dCdb, axis=0)
        return np.transpose(dCda, (2, 0, 1, 3))

    """backprop todo:
  remove intermediate values dCdb and dCdF. Instead, update them directly"""

    def initializer(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = self.get_output_shape(input_shape)
        self.filter = np.random.standard_normal(
            self.kernel_size) * np.sqrt(2/(input_shape[0]*input_shape[1]))
        self.biases = np.random.standard_normal(self.output_shape)
        # self.biases = np.zeros(self.output_shape)
        return self.output_shape

    def get_output_shape(self, input_shape):
        return tuple([int((i-k)/self.stride)+1 for i, k in zip(input_shape[:2], self.kernel_size)]) + (self.input_shape[-1],)


class Flatten(Layer):
    def __init__(self):
        super().__init__()

    def feedforward(self, a):
        self.set_input(a)
        return np.reshape(a.sum(axis=3), (a.shape[0], -1, 1))

    def backprop(self, dreshape, lr):
        return np.repeat(dreshape.reshape((dreshape.shape[0],) + self.input_shape[:-1] + (1,)), self.input_shape[-1], axis=3)

    def get_output_shape(self, input_shape):
        return (np.prod(input_shape[:-1]).item(), 1)


class Dense(Layer):
    w_inits = {
        "default": default_weight_initializer,
        "he": he_weight_initializer,
        "xavier": xavier_weight_initializer
    }

    # shape is an integer specifying the # of output neurons.
    def __init__(self, shape, activation="sigmoid", weight_initialization="he"):
        super().__init__()
        self.shape = shape
        self.w_init = Dense.w_inits[weight_initialization]
        self.activation = Layer.activations[activation]

    def initializer(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = self.get_output_shape(input_shape)
        self.weights = self.w_init(input_shape[0], self.output_shape[0])
        self.biases = np.random.standard_normal(self.output_shape)
        return self.output_shape

    def feedforward(self, a):
        self.set_input(a)
        self.z = np.matmul(self.weights, a) + self.biases
        return self.activation.fn(self.z)

    def backprop(self, dCda, lr):
        delta = dCda * self.activation.prime(self.z)
        dCda = np.matmul(self.weights.transpose(), delta)
        self.biases = self.biases - (lr/delta.shape[0])*np.sum(delta, axis=0)
        self.weights = self.weights - \
            (lr/self.a.shape[0])*np.sum(np.matmul(delta,
                                                  self.a.transpose([0, 2, 1])), axis=0)
        return dCda

    def get_output_shape(self, input_shape):
        return (self.shape, 1)


class Output(Dense):
    costs = {
        "cross entropy": CrossEntropy,
        "quadratic": Quadratic
    }

    def __init__(self, shape, activation="softmax", weight_initialization="he", cost="cross entropy"):
        super().__init__(shape=shape, activation=activation,
                         weight_initialization=weight_initialization)
        self.cost = Output.costs[cost]

    # dCda is actually just the output a of the feedforward.
    def backprop(self, dCda, lr):
        delta = self.cost.delta(dCda, self.train_label)
        dCda = np.matmul(self.weights.transpose(), delta)
        self.biases = self.biases - (lr/delta.shape[0])*np.sum(delta, axis=0)
        self.weights = self.weights - \
            (lr/self.a.shape[0])*np.sum(np.matmul(delta,
                                                  self.a.transpose([0, 2, 1])), axis=0)
        return dCda

    def set_labels(self, train_label):
        self.train_label = train_label
