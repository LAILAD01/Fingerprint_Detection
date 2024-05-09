import numpy as np

class NeuralNetwork:
    def __init__(self, input_size=128*128, num_classes=10, hidden_units=100, learning_rate=0.01):
        self.learning_rate = learning_rate
        # Initialize weights and biases for the first hidden layer
        self.weights1 = np.random.randn(input_size, hidden_units) * 0.01
        self.bias1 = np.zeros((1, hidden_units))
        # Initialize weights and biases for the output layer
        self.weights2 = np.random.randn(hidden_units, num_classes) * 0.01
        self.bias2 = np.zeros((1, num_classes))
        # Cache for backward pass
        self.cache = {}

    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(x.dtype)

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def forward(self, x):
        # Forward pass through the first hidden layer
        z1 = np.dot(x, self.weights1) + self.bias1
        a1 = self.relu(z1)
        # Forward pass through the output layer
        z2 = np.dot(a1, self.weights2) + self.bias2
        a2 = self.softmax(z2)
        # Store forward pass results for use in backpropagation
        self.cache = {'x': x, 'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2}
        return a2

    def compute_loss(self, y_pred, y_true):
        # Cross-entropy loss
        m = y_true.shape[0]
        log_likelihood = -np.log(y_pred[range(m), y_true.argmax(axis=1)])
        loss = np.sum(log_likelihood) / m
        return loss
    
    def backward(self, y_true):
        # Unpack cache
        x, z1, a1, z2, a2 = self.cache['x'], self.cache['z1'], self.cache['a1'], self.cache['z2'], self.cache['a2']
        # Calculate derivatives
        m = y_true.shape[0]
        dz2 = a2 - y_true  # Derivative of loss with respect to z2
        dw2 = np.dot(a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        da1 = np.dot(dz2, self.weights2.T)
        dz1 = da1 * self.relu_derivative(z1)
        dw1 = np.dot(x.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        # Update weights and biases
        self.weights1 -= self.learning_rate * dw1
        self.bias1 -= self.learning_rate * db1
        self.weights2 -= self.learning_rate * dw2
        self.bias2 -= self.learning_rate * db2
    
    def train(self, x_train, y_train, epochs):
        for epoch in range(epochs):
            y_pred = self.forward(x_train)
            loss = self.compute_loss(y_pred, y_train)
            self.backward(y_train)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, x):
        # Predict class labels for samples in x
        output = self.forward(x)
        return np.argmax(output, axis=1)

    def save_model(self, file_path):
        # Save model parameters
        np.savez(file_path, weights1=self.weights1, bias1=self.bias1, weights2=self.weights2, bias2=self.bias2)

    def load_model(self, file_path):
        # Load model parameters
        data = np.load(file_path)
        self.weights1 = data['weights1']
        self.bias1 = data['bias1']
        self.weights2 = data['weights2']
        self.bias2 = data['bias2']
