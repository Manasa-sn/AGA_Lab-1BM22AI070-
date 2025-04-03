import numpy as np
from sklearn.preprocessing import binarize

class Stack:
    def __init__(self, capacity):
        self.capacity = capacity
        self.stack = []

    def push(self, item):
        if len(self.stack) < self.capacity:
            self.stack.append(item)
            return True
        return False

    def pop(self):
        if not self.is_empty():
            return self.stack.pop()
        return None

    def is_empty(self):
        return len(self.stack) == 0

class RBM:
    def __init__(self, n_visible, n_hidden, learning_rate=0.1):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.lr = learning_rate

        self.weights = np.random.normal(0, 0.1, (n_visible, n_hidden))
        self.v_bias = np.zeros(n_visible)
        self.h_bias = np.zeros(n_hidden)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sample_hidden(self, visible):
        if visible.ndim == 1:
            visible = visible.reshape(1, -1)
        activation = np.dot(visible, self.weights) + self.h_bias
        prob_h = self.sigmoid(activation)
        sampled_h = binarize(prob_h, threshold=0.5)
        return prob_h, sampled_h

    def sample_visible(self, hidden):
        if hidden.ndim == 1:
            hidden = hidden.reshape(1, -1)
        activation = np.dot(hidden, self.weights.T) + self.v_bias
        prob_v = self.sigmoid(activation)
        sampled_v = binarize(prob_v, threshold=0.5)
        return prob_v, sampled_v

    def train(self, data, epochs=100, batch_size=10):
        n_samples = data.shape[0]

        for epoch in range(epochs):
            idx = np.random.permutation(n_samples)
            data_shuffled = data[idx]

            for i in range(0, n_samples, batch_size):
                batch = data_shuffled[i:i+batch_size]

                pos_hidden_probs, pos_hidden = self.sample_hidden(batch)
                pos_associations = np.dot(batch.T, pos_hidden_probs)

                neg_visible_probs, neg_visible = self.sample_visible(pos_hidden)
                neg_hidden_probs, neg_hidden = self.sample_hidden(neg_visible)
                neg_associations = np.dot(neg_visible.T, neg_hidden_probs)

                self.weights += self.lr * (pos_associations - neg_associations) / batch_size
                self.v_bias += self.lr * np.mean(batch - neg_visible, axis=0)
                self.h_bias += self.lr * np.mean(pos_hidden_probs - neg_hidden_probs, axis=0)

class StackRBM:
    def __init__(self, capacity, n_visible, n_hidden):
        self.stack = Stack(capacity)
        self.rbm = RBM(n_visible, n_hidden)

    def push_and_train(self, data):
        if self.stack.push(data):
            data_array = np.array(data).reshape(1, -1)
            self.rbm.train(data_array)
            return True
        return False

    def pop_and_reconstruct(self):
        data = self.stack.pop()
        if data is not None:
            data_array = np.array(data).reshape(1, -1)
            hidden_probs, hidden = self.rbm.sample_hidden(data_array)
            visible_probs, reconstructed = self.rbm.sample_visible(hidden)
            return reconstructed[0]
        return None

    def get_stack_size(self):
        return len(self.stack.stack)

if __name__ == "__main__":
    stack_rbm = StackRBM(capacity=5, n_visible=4, n_hidden=2)
    sample_data = [[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]]
    for data in sample_data:
        success = stack_rbm.push_and_train(data)
        print(f"Pushed data {data}: {success}")
    reconstructed = stack_rbm.pop_and_reconstruct()
    if reconstructed is not None:
        print(f"Reconstructed data: {reconstructed}")
    print(f"Current stack size: {stack_rbm.get_stack_size()}")
