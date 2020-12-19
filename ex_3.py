import numpy as np
import sys


def reLU(x):
    return np.maximum(x, 0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


def make_y_vector(y, size):
    vector_y = np.zeros((size, 1))
    vector_y[int(y)] = 1
    return vector_y


def forward_propagation(x, y, params):
    w1, b1, w2, b2 = [params[key] for key in ('w1', 'b1', 'w2', 'b2')]
    z1 = np.dot(w1, x) + b1
    h1 = sigmoid(z1)
    z2 = np.dot(w2, h1) + b2
    # calculate the loss to the multi-class
    h2 = softmax(z2)
    loss = np.sum(-make_y_vector(y, 10) * np.log(h2))
    ret = {'x': x, 'y': y, 'z1': z1, 'h1': h1, 'z2': z2, 'h2': h2, 'loss': loss}
    for key in params:
        ret[key] = params[key]
    return ret


def back_propagation(params):
    x, y, z1, h1, z2, h2, loss = [params[key] for key in ('x', 'y', 'z1', 'h1', 'z2', 'h2', 'loss')]
    vector_y = make_y_vector(y, 10)
    dz2 = (h2 - vector_y)
    dw2 = np.dot(dz2, h1.T)
    db2 = dz2
    active = sigmoid(z1)
    dz1 = np.dot(params['w2'].T, (h2 - vector_y)) * active * (1 - active)
    dw1 = np.dot(dz1, x.T)
    db1 = dz1
    return {'db1': db1, 'dw1': dw1, 'db2': db2, 'dw2': dw2}


def update(bprop_cache, params, eta):
    w1, b1, w2, b2 = [params[key] for key in ('w1', 'b1', 'w2', 'b2')]
    db1, dw1, db2, dw2 = [bprop_cache[key] for key in ('db1', 'dw1', 'db2', 'dw2')]
    w2 = w2 - eta * dw2
    w1 = w1 - eta * dw1
    b2 = b2 - eta * db2
    b1 = b1 - eta * db1
    return {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}


def write_to_file(predict_y):
    with open('test_y', 'w') as file:
        for y in predict_y:
            file.write("%s\n" % y)
    file.close()


def main():
    train_x, train_y, test_x = sys.argv[1], sys.argv[2], sys.argv[3]
    #train_x, train_y = sys.argv[1], sys.argv[2]
    train_x = np.loadtxt(train_x) / 255.0  # normalized
    train_y = np.loadtxt(train_y)

    # initialize random parameters w and bias (784 fixels on 10 labels)
    w1 = np.random.randn(200, 784) * np.sqrt(2 / 784)
    b1 = np.full((200, 1), 0)
    w2 = np.random.randn(10, 200) * np.sqrt(2 / 200)
    b2 = np.full((10, 1), 0)
    params = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}

    # now we train the NN model on training set
    epochs = 20
    eta = 0.01
    # train
    for epoch in range(epochs):
        # shuffle the training set
        zip_info = list(zip(train_x, train_y))
        np.random.shuffle(zip_info)
        # run on each train sample
        for x, y in zip(train_x, train_y):
            x_arr = np.ndarray(shape=(784, 1), buffer=x)
            # forward the input
            fprop = forward_propagation(x_arr, y, params)

            # compute the gradients
            bprop = back_propagation(fprop)

            # update the parameters
            updated_params = update(bprop, params, eta)
            params = updated_params

    # run on the test
    test_set = np.loadtxt(test_x) / 255.0  # normalized
    predict_y = []
    for test in test_set:
        test_x = np.ndarray(shape=(784, 1), buffer=test)
        fprop = forward_propagation(test_x, 0, params)
        y_hat = np.argmax(fprop['h2'])
        predict_y.append(int(y_hat))
    write_to_file(predict_y)


if __name__ == "__main__":
    main()