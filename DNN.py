import keras
import numpy as np
from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout, regularizers
from keras.layers.advanced_activations import LeakyReLU, PReLU


class DNN:
    epoch_count = 0

    ##################################################################################################
    # feature_size: size of the input
    # no_of_classes: number of output classes
    # dense_layer_dense_layer_nodes: numpy array for fully connected layer node configuration
    #################################################################################################
    def __init__(self):
        self.model = None

    def create(self, feature_size, layer_nodes, dropouts, no_of_output_class, activation='softmax',
               alpha=0.1, l2_norm=0,l1_norm=0):

        """
        :param layer_nodes: number of of input features
        :param layer_nodes: 1 D numpy array for the the number of nodes in each layer
        :param no_of_output_class: Number of classes in output
        :param activation: type of activation to use
        :param alpha: values for negative gradient
        :param l2_norm: L2 regualrization constant
        :param l1_norm: L1 regualrization constant
        :return:
        """
        layer_count = 0

        input_layer = Input(shape=feature_size)

        # To keep track the last inserted layer
        last_layer = input_layer

        # Iterate to create the layers
        for i in range(layer_nodes.shape[0]):

            if activation != 'LeakyRelu':
                if i == 0:
                    last_layer = Dense(layer_nodes[layer_count], input_dim=feature_size,activation=activation,
                                       kernel_initializer=keras.initializers.glorot_normal(seed=None),
                                       kernel_regularizer=regularizers.l2(l2_norm),
                                       activity_regularizer=regularizers.l1(l1_norm))(last_layer)
                else:
                    last_layer = Dense(layer_nodes[layer_count],activation=activation,
                                       kernel_initializer=keras.initializers.glorot_normal(seed=None),
                                       kernel_regularizer=regularizers.l2(l2_norm),
                                       activity_regularizer=regularizers.l1(l1_norm))(last_layer)
            else:
                if i == 0:
                    last_layer = Dense(layer_nodes[layer_count], input_dim=feature_size,
                                       kernel_initializer=keras.initializers.glorot_normal(seed=None),
                                       kernel_regularizer=regularizers.l2(l2_norm),
                                       activity_regularizer=regularizers.l1(l1_norm))(last_layer)
                    last_layer = LeakyReLU(alpha=alpha)(last_layer)
                else:
                    last_layer = Dense(layer_nodes[layer_count],
                                       kernel_initializer=keras.initializers.glorot_normal(seed=None),
                                       kernel_regularizer=regularizers.l2(l2_norm),
                                       activity_regularizer=regularizers.l1(l1_norm))(last_layer)
                    last_layer = LeakyReLU(alpha=alpha)(last_layer)

            if dropouts[i] > 0:
                last_layer = Dropout(dropouts[i])(last_layer)
            layer_count += 1
        activation = 'sigmoid'
        output_layer = Dense(no_of_output_class, activation=activation,
                             kernel_initializer=keras.initializers.glorot_normal(seed=None),
                             kernel_regularizer=regularizers.l2(l2_norm),
                             activity_regularizer=regularizers.l1(l1_norm))(last_layer)
        self.model = Model(inputs=input_layer, outputs=output_layer)

    def compile(self, loss_function, optimizer_name='SGD', learning_rate=0.001, decay=0.0, momentum=None, metrics=None):
        """
        :param loss_function: loss function to use
        :param optimizer_name: name of the optimizer to use
        :param learning_rate: Initial learning rate
        :param decay: decay factor for learning rate
        :param momentum: to accelerate the optimizer towards the relevant direction
        :return:
        """
        optimizer = None
        if optimizer_name == 'SGD':
            if learning_rate is None:
                learning_rate = 0.01
                # raise a warning in case SGd is requested but no learnig rate is provided
                Warning("Using the default learning rate", learning_rate)
            if momentum is None:
                optimizer = keras.optimizers.SGD(lr=learning_rate, decay=decay)
            else:
                optimizer = keras.optimizers.SGD(lr=learning_rate, decay=decay, momentum=momentum)
        elif optimizer_name == 'RMSProp':
            optimizer = keras.optimizers.RMSprop()
        elif optimizer_name == 'Adagrad':
            optimizer = keras.optimizers.Adagrad()
        elif optimizer_name == 'Adadelta':
            optimizer = keras.optimizers.Adadelta()
        elif optimizer_name == 'Adam':
            optimizer = keras.optimizers.Adam(lr=learning_rate, decay=decay)
        elif optimizer_name == 'Adamax':
            optimizer = keras.optimizers.Adamax()
        elif optimizer_name == 'Nadam':
            optimizer = keras.optimizers.Nadam()
        self.model.compile(loss=loss_function, optimizer=optimizer, metrics=metrics)

    def train(self, X_train, Y_train, X_test=None, Y_test=None, epochs=50, class_weights=None, batch_size=64,
              verbose=2):
        """
        :param X_train: Features for the training data
        :param Y_train: labels for the training data
        :param X_test: Features for the test data
        :param Y_test: labels for the test data
        :param epochs: Number of epochs to train
        :param class_weights: weights given to each class
        :param batch_size: Number of training example to process in each batch
        :param verbose: boolean value to indicate the printing the training outputs or not
        :return:
        """
        if X_test is None:
            self.model.fit(X_train, Y_train, validation_data=None, epochs=epochs, class_weight=class_weights,
                           batch_size=batch_size,
                           verbose=verbose)
        else:
            self.model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epochs,
                           class_weight=class_weights,
                           batch_size=batch_size, verbose=verbose)

    def predict(self, X):
        """
        This function predicts the label for teh input
        :param X: Data to predict
        :return:
        """
        return np.argmax(self.model.predict(X), axis=1)

    def save(self, directory="./", name="model.h5"):
        self.model.save(directory + name)

    def load(self, filename):
        self.model = load_model(filename)

