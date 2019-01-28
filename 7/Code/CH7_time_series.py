import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

from tensorflow.python.framework import dtypes
from tensorflow.contrib import learn
from tensorflow.contrib import rnn
from tensorflow.contrib.learn.python import SKCompat
from tensorflow.contrib import learn as tflearn
from tensorflow.contrib import layers as tflayers

import logging

logging.basicConfig(level=logging.INFO)

from sklearn.metrics import mean_squared_error

LOG_DIR = './ops_logs'
TIMESTEPS = 5
RNN_LAYERS = [{'steps': TIMESTEPS, 'num_units': 5}]
DENSE_LAYERS = None
TRAINING_STEPS = 10000
BATCH_SIZE = 100
PRINT_STEPS = TRAINING_STEPS / 100


def lstm_model2(time_steps, rnn_layers, dense_layers=None):
    def lstm_cells(layers):
        return [rnn.BasicLSTMCell(layer['steps'])
                for layer in layers]

    def dnn_layers(input_layers, layers):
        return input_layers

    def _lstm_model(X, y):
        stacked_lstm = rnn.MultiRNNCell(lstm_cells(rnn_layers))
        x_ = tf.unstack(X, num=time_steps, axis=1)
        output, layers = rnn.static_rnn(stacked_lstm, x_, dtype=dtypes.float64)
        output = dnn_layers(output[-1], dense_layers)
        return learn.models.linear_regression(output, y)

    return _lstm_model


def lstm_model(time_steps, rnn_layers, dense_layers=None, learning_rate=0.1, optimizer='Adagrad'):
    print(time_steps)
    # exit(0)
    """
        Creates a deep model based on:
            * stacked lstm cells
            * an optional dense layers
        :param num_units: the size of the cells.
        :param rnn_layers: list of int or dict
                             * list of int: the steps used to instantiate the `BasicLSTMCell` cell
                             * list of dict: [{steps: int, keep_prob: int}, ...]
        :param dense_layers: list of nodes for each layer
        :return: the model definition
        """

    def lstm_cells(layers):
        if isinstance(layers[0], dict):
            return [rnn.DropoutWrapper(rnn.BasicLSTMCell(layer['num_units'], state_is_tuple=True), layer['keep_prob'])
                    if layer.get('keep_prob')
                    else rnn.BasicLSTMCell(layer['num_units'], state_is_tuple=True)
                    for layer in layers]

        return [rnn.BasicLSTMCell(steps, state_is_tuple=True) for steps in layers]

    def dnn_layers(input_layers, layers):
        if layers and isinstance(layers, dict):
            return tflayers.stack(input_layers, tflayers.fully_connected,
                                  layers['layers'],
                                  activation=layers.get('activation'),
                                  dropout=layers.get('dropout'))
        elif layers:
            return tflayers.stack(input_layers, tflayers.fully_connected, layers)
        else:
            return input_layers

    def _lstm_model(X, y):
        stacked_lstm = rnn.MultiRNNCell(lstm_cells(rnn_layers), state_is_tuple=True)
        x_ = tf.unstack(X, num=time_steps, axis=1)

        output, layers = rnn.static_rnn(stacked_lstm, x_, dtype=dtypes.float64)
        output = dnn_layers(output[-1], dense_layers)
        prediction, loss = tflearn.models.linear_regression(output, y)
        train_op = tf.contrib.layers.optimize_loss(
            loss, tf.contrib.framework.get_global_step(), optimizer=optimizer,
            learning_rate=learning_rate)
        return prediction, loss, train_op

    return _lstm_model


# regressor = learn.Estimator(model_fn=lstm_model(TIMESTEPS, RNN_LAYERS, DENSE_LAYERS))

regressor = SKCompat(learn.Estimator(model_fn=lstm_model(TIMESTEPS, RNN_LAYERS, DENSE_LAYERS), ))

df = pd.read_csv("data/elec_load.csv", error_bad_lines=False)
plt.subplot()
plot_test, = plt.plot(df.values[:1500], label='Load')
plt.legend(handles=[plot_test])

print(df.describe())
array = (df.values - 147.0) / 339.0
plt.subplot()
plot_test, = plt.plot(array[:1500], label='Normalized Load')
plt.legend(handles=[plot_test])

listX = []
listy = []
X = {}
y = {}

for i in range(0, len(array) - 6):
    listX.append(array[i:i + 5].reshape([5, 1]))
    listy.append(array[i + 6])

arrayX = np.array(listX)
arrayy = np.array(listy)

X['train'] = arrayX[0:12000]
X['test'] = arrayX[12000:13000]
X['val'] = arrayX[13000:14000]

y['train'] = arrayy[0:12000]
y['test'] = arrayy[12000:13000]
y['val'] = arrayy[13000:14000]

# print y['test'][0]
# print y2['test'][0]


# X1, y2 = generate_data(np.sin, np.linspace(0, 100, 10000), TIMESTEPS, seperate=False)
# create a lstm instance and validation monitor
validation_monitor = learn.monitors.ValidationMonitor(X['val'], y['val'],
                                                      every_n_steps=PRINT_STEPS,
                                                      early_stopping_rounds=1000)

regressor.fit(X['train'], y['train'], monitors=[validation_monitor])

predicted = regressor.predict(X['test'])
rmse = np.sqrt(((predicted - y['test']) ** 2).mean(axis=0))
score = mean_squared_error(predicted, y['test'])
print(("MSE: %f" % score))

# plot_predicted, = plt.plot(array[:1000], label='predicted')

plt.subplot()
plot_predicted, = plt.plot(predicted, label='predicted')

plot_test, = plt.plot(y['test'], label='test')
plt.legend(handles=[plot_predicted, plot_test])
