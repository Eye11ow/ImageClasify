import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    grad = np.zeros_like(x)
    grad[x > 0] = 1
    return grad

def softmax(x):
    shiftx = x - np.max(x, axis=1, keepdims=True)
    exps = np.exp(shiftx)
    return exps / np.sum(exps, axis=1, keepdims=True)

def cross_entropy_loss(probs, y):
    N = probs.shape[0]
    correct_logprobs = -np.log(probs[np.arange(N), np.argmax(y, axis=1)] + 1e-8)
    loss = np.sum(correct_logprobs) / N
    return loss

class ThreeLayerNN:
    def __init__(self, input_size, hidden_size, output_size, activation='relu', reg=0.0):
        self.params = {}
        self.params['W1'] = 0.001 * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = 0.001 * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        
        self.activation = activation
        self.reg = reg  # L2 正则化强度

    def forward(self, X):
        self.z1 = X.dot(self.params['W1']) + self.params['b1']
        if self.activation == 'relu':
            self.a1 = relu(self.z1)
        elif self.activation == 'tanh':
            self.a1 = np.tanh(self.z1)
        else:
            raise ValueError('Unsupported activation function')
        self.scores = self.a1.dot(self.params['W2']) + self.params['b2']
        return self.scores

    def loss(self, X, y=None):
        scores = self.forward(X)
        if y is None:
            return scores

        probs = softmax(scores)
        data_loss = cross_entropy_loss(probs, y)
        reg_loss = 0.5 * self.reg * (np.sum(self.params['W1']**2) + np.sum(self.params['W2']**2))
        loss = data_loss + reg_loss

        grads = {}
        N = X.shape[0]
        dscores = probs.copy()
        dscores[np.arange(N), np.argmax(y, axis=1)] -= 1
        dscores /= N

        grads['W2'] = self.a1.T.dot(dscores) + self.reg * self.params['W2']
        grads['b2'] = np.sum(dscores, axis=0)

        da1 = dscores.dot(self.params['W2'].T)
        if self.activation == 'relu':
            dz1 = da1 * relu_grad(self.z1)
        elif self.activation == 'tanh':
            dz1 = da1 * (1 - np.tanh(self.z1)**2)
        grads['W1'] = X.T.dot(dz1) + self.reg * self.params['W1']
        grads['b1'] = np.sum(dz1, axis=0)

        return loss, grads

    def predict(self, X):
        scores = self.forward(X)
        return np.argmax(scores, axis=1)