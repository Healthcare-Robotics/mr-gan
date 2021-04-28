import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Activation, Lambda
from keras import backend as K
import tensorflow as tf
import keras

def run_class_activation_maps(problem='classification'):
    count = 5000
    if problem == 'classification':
        num_classes = 5
        y_train = np.random.randint(0, num_classes, size=count)
        X_train = np.random.rand(count, 10)
        for i, yy in enumerate(y_train):
            X_train[i, yy+2] = 0.1
            X_train[i, yy+3] = 0.2
            X_train[i, yy+4] = 0.3

        y_test = np.random.randint(0, num_classes, size=1000)
        X_test = np.random.rand(1000, 10)
        for i, yy in enumerate(y_test):
            X_test[i, yy+2] = 0.1
            X_test[i, yy+3] = 0.2
            X_test[i, yy+4] = 0.3

        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)

        model = Sequential()
        model.add(Dense(64, activation='relu', input_dim=10))
        model.add(Dense(64, activation='relu'))
        # model.add(Dense(num_classes, activation='softmax'))
        # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.add(Dense(num_classes, activation='sigmoid'))
        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    else:
        a = np.random.rand(count, 3)
        # a = np.random.uniform(0, 10, size=(count, 3))
        X_train = np.random.rand(count, 10)
        X_train[:, 2:5] = a
        y_train = np.sum(a, axis=-1)

        b = np.random.rand(1000, 3)
        X_test = np.random.rand(1000, 10)
        X_test[:, 2:5] = b
        y_test = np.sum(b, axis=-1)

        model = Sequential()
        model.add(Dense(64, activation='relu', input_dim=10))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1))
        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=50, batch_size=128)
    score = model.evaluate(X_test, y_test, batch_size=128)
    print(score)

    # ------- Class activation maps begins here --------

    def normalize(x):
        # utility function to normalize a tensor by its L2 norm
        return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

    sample = np.array([X_test[0]])
    if problem == 'classification':
        # loss = K.categorical_crossentropy(y_test[0], model.layers[-1].output)
        # loss = model.layers[-1].output[0, np.argmax(y_test[0])]
        loss = K.mean(K.square(model.layers[-1].output - y_test[0]), axis=-1)
    else:
        loss = K.mean(K.square(model.layers[-1].output - y_test[0]), axis=-1)
    dense_input = model.layers[0].input
    grads = normalize(K.gradients(loss, dense_input)[0])
    gradient_function = K.function([model.layers[0].input], [dense_input, grads])

    output, grads_val = gradient_function([sample])
    print(np.shape(output), np.shape(grads_val))
    output, grads_val = output[0], grads_val[0]

    class_activation_map = np.abs(grads_val)
    print(np.shape(class_activation_map))
    print(output, grads_val)
    print('Mean:', np.mean(class_activation_map), 'Max:', np.max(class_activation_map))
    # if np.max(class_activation_map) < 0:
    #     class_activation_map -= np.mean(class_activation_map)
    # class_activation_map = np.maximum(class_activation_map, 0)
    # heatmap = class_activation_map / np.max(class_activation_map)
    # Normalize class activation map
    class_activation_map = (class_activation_map - np.min(class_activation_map)) / (np.max(class_activation_map) - np.min(class_activation_map))
    # heatmap = class_activation_map
    heatmap = np.square(class_activation_map)
    print(np.min(class_activation_map), np.max(class_activation_map))

    plt.imshow(np.expand_dims(heatmap, axis=0), cmap='jet', aspect='auto')
    plt.plot((output - np.min(output)) / (np.max(output) - np.min(output)) - 0.5, 'w')
    if problem == 'classification':
        plt.title('Highlighted regions should be: %d, %d, %d' % (np.argmax(y_test[0])+2, np.argmax(y_test[0])+3, np.argmax(y_test[0])+4))
    else:
        plt.title('Highlighted regions should be: 2, 3, 4')
    plt.show()


# run_class_activation_maps(problem='classification')
run_class_activation_maps(problem='regression')


