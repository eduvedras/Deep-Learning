#%%
#!/usr/bin/env python

# Deep Learning Homework 1

import argparse
import random
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, savefig

import utils

def configure_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))

    def update_weight(self, x_i, y_i, **kwargs):
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features):
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        print(y.shape)
        print(y_hat.shape)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


class Perceptron(LinearModel):
    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """
        # Q1.1a
        #print(self.W.shape)
        y_hat = (self.W.dot(x_i)).argmax(axis=0)
        if y_hat != y_i:
            self.W += kwargs['learning_rate'] * y_i * x_i


class LogisticRegression(LinearModel):
    def update_weight(self, x_i, y_i, learning_rate=0.001):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """
        # Q1.1b
        y_hat = (self.W.dot(x_i)).argmax(axis=0)
        self.W += learning_rate * (y_i - y_hat) * x_i


class MLP(object):
    # Q3.2b. This MLP skeleton code allows the MLP to be used in place of the
    # linear models with no changes to the training loop or evaluation code
    # in main().
    def __init__(self, n_classes, n_features, hidden_size):
        units=[n_features,hidden_size,n_classes]
        # Initialize an MLP with a single hidden layer.
        self.W1 = np.random.normal(0.1, 0.1**2, size=(units[1],units[0]))
        self.W2 = np.random.normal(0.1, 0.1**2, size=(units[2],units[1]))
        self.b1 = np.zeros(units[1])
        self.b2 = np.zeros(units[2])
        '''self.h0 = []
        self.rand = -1

        self.z1 = []
        self.h1 = []

        self.z2 = []
        self.h2 = []'''

    def predict(self, X):
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes, whereas this is required
        # at training time.
        rand = np.random.randint(len(X))
        #self.h0 = X[self.rand]
        h0 = X[rand]

        z1 = self.W1.dot(h0) + self.b1
        h1 = np.maximum(0,z1)

        z2 = self.W2.dot(h1) + self.b2
        h2 = z2
        
        return h2.argmax(axis=0)


    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible

    def train_epoch(self, X, y, learning_rate=0.001):
        #self.predict(X)
        rand = np.random.randint(len(X))
        h0 = X[rand]

        z1 = self.W1.dot(h0) + self.b1
        h1 = np.maximum(0,z1)

        z2 = self.W2.dot(h1) + self.b2
        h2 = z2
        # Gradient of hidden layer below before activation.
        grad_z2 = h2 - y[rand]   # Grad of loss wrt z3.
        print(grad_z2)

        # Gradient of hidden parameters.
        grad_W2 = grad_z2[:, None].dot(h1[:, None].T)
        grad_b2 = grad_z2
        print(grad_W2)
        print(grad_b2)

        # Gradient of hidden layer below.
        grad_h1 = self.W2.T.dot(grad_z2)
        print(grad_h1)
        
        # Gradient of hidden layer below before activation.
        h1_aux = h1
        h1_aux[h1_aux<=0] = 0
        h1_aux[h1_aux>0] = 1
        
        grad_z1 = h1_aux
        
        grad_z1 = grad_z1 * grad_h1
        
        
        
        '''grad_z1 = []
        i=0
        print(len(self.h1))
        while i < len(self.h1):
            if self.h1[i] >=0:
                grad_z1.append(grad_h1[i] * 1)
            else:
                grad_z1.append(0)
            i = i + 1'''
        
            
        print(grad_z1)

        # Gradient of hidden parameters.
        grad_W1 = grad_z1[:, None].dot(h0[:, None].T)
        grad_b1 = grad_z1
        print(grad_W1)
        print(grad_b1)
        
        self.W1 -= learning_rate*grad_W1
        self.b1 -= learning_rate*grad_b1
        self.W2 -= learning_rate*grad_W2
        self.b2 -= learning_rate*grad_b2


def plot(epochs, valid_accs, test_accs):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xticks(epochs)
    plt.plot(epochs, valid_accs, label='validation')
    plt.plot(epochs, test_accs, label='test')
    plt.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=200,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-layers', type=int, default=1,
                        help="""Number of hidden layers (needed only for MLP,
                        not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp"
    data = utils.load_classification_data(bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]

    n_classes = np.unique(train_y).size  # 10
    n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size)#, opt.layers)
    epochs = np.arange(1, opt.epochs + 1)
    valid_accs = []
    test_accs = []
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        model.train_epoch(
            train_X,
            train_y,
            learning_rate=opt.learning_rate
        )
        valid_accs.append(model.evaluate(dev_X, dev_y))
        test_accs.append(model.evaluate(test_X, test_y))

    # plot
    plot(epochs, valid_accs, test_accs)
    savefig(f'images/{opt.model}.png')


if __name__ == '__main__':
    main()

# %%
