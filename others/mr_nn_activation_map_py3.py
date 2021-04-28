import os, sys, time, glob, librosa, itertools, argparse
import numpy as np
import pickle
from sklearn.utils import shuffle
from sklearn import preprocessing, decomposition
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
import matplotlib.pyplot as plt

import keras
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Activation, Lambda
from keras.layers.noise import GaussianNoise
from keras import backend as K

'''
Material Recognition GAN
'''

# K.set_floatx('float64')

def dataset(modalities=0, forcetempTime=4, contactmicTime=0.2, leaveObjectOut=False, verbose=False):
    materials = ['plastic', 'glass', 'fabric', 'metal', 'wood', 'ceramic']
    X = []
    y = []
    objects = dict()
    for m, material in enumerate(materials):
        if verbose:
            print('Processing', material)
            sys.stdout.flush()
        with open('data_processed/processed_0.1sbefore_%s_times_%.2f_%.2f.pkl' % (material, forcetempTime, contactmicTime), 'rb') as f:
            allData = pickle.load(f, encoding='latin1')
            for j, (objName, objData) in enumerate(allData.items()):
                if leaveObjectOut:
                    objects[objName] = {'x': [], 'y': []}
                    X = objects[objName]['x']
                    y = objects[objName]['y']
                for i in range(len(objData['temperature'])):
                    y.append(m)

                    if modalities > 2:
                        # Mel-scaled power (energy-squared) spectrogram
                        sr = 48000
                        S = librosa.feature.melspectrogram(np.array(objData['contact'][i]), sr=sr, n_mels=128)
                        # Convert to log scale (dB)
                        log_S = librosa.logamplitude(S, ref_power=np.max)

                    if modalities == 0:
                        X.append(objData['force0'][i] + objData['force1'][i])
                    elif modalities == 1:
                        X.append(objData['temperature'][i])
                    elif modalities == 2:
                        X.append(objData['temperature'][i] + objData['force0'][i] + objData['force1'][i])
                    elif modalities == 3:
                        X.append(log_S.flatten())
                    elif modalities == 4:
                        X.append(objData['temperature'][i] + log_S.flatten().tolist())
                    elif modalities == 5:
                        X.append(objData['temperature'][i] + objData['force0'][i] + objData['force1'][i] + log_S.flatten().tolist())
                    elif modalities == 6:
                        X.append(objData['force0'][i] + objData['force1'][i] + log_S.flatten().tolist())

    if leaveObjectOut:
        return objects
    else:
        X = np.array(X)
        y = np.array(y)
        if verbose:
            print('X:', np.shape(X), 'y:', np.shape(y))
        return X, y

def mr_nn(X, y, percentlabeled=50, trainTestSets=None, verbose=False):
    # Non Deterministic output
    np.random.seed(np.random.randint(1e9))

    materials = ['plastic', 'glass', 'fabric', 'metal', 'wood', 'ceramic']
    test_ratio = 200*len(materials)
    num_labeled_examples = int(10*percentlabeled)

    # Split into train and test sets
    if trainTestSets is None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, stratify=y)
    else:
        X_train, X_test, y_train, y_test = trainTestSets
    if verbose:
        print('Num of class examples in test set:', [np.sum(y_test == i) for i in range(len(materials))])
        print('X_train:', np.shape(X_train), 'y_train:', np.shape(y_train), 'X_test:', np.shape(X_test), 'y_test:', np.shape(y_test))

    # Scale data to zero mean and unit variance
    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Select labeled data
    X_train, y_train = shuffle(X_train, y_train)
    x_labeled = np.concatenate([X_train[y_train==j][:num_labeled_examples] for j in range(len(materials))], axis=0)
    y_labeled = np.concatenate([[j]*num_labeled_examples for j in range(len(materials))], axis=0)
    if verbose:
        print('x_labeled:', np.shape(x_labeled), 'y_labeled:', np.shape(y_labeled))

    y_labeled = keras.utils.to_categorical(y_labeled, len(materials))
    y_test = keras.utils.to_categorical(y_test, len(materials))

    inputs = Input(shape=(X_train.shape[1],))
    x = GaussianNoise(0.3)(inputs)
    # x = Dense(1000, activation='relu')(x)
    # x = GaussianNoise(0.5)(x)
    # x = Dense(500, activation='relu')(x)
    # x = GaussianNoise(0.5)(x)
    # x = Dense(250, activation='relu')(x)
    # x = GaussianNoise(0.5)(x)
    # x = Dense(250, activation='relu')(x)
    # x = GaussianNoise(0.5)(x)
    # x = Dense(250, activation='relu')(x)

    x = Dense(128, activation='relu')(x)
    x = GaussianNoise(0.5)(x)
    x = Dense(64, activation='relu')(x)
    # outputs = Dense(len(materials))(x)
    # outputs = Activation('softmax')(outputs)
    # outputs = Dense(len(materials), activation='softmax')(x)
    outputs = Dense(len(materials), activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train on x_labeled, y_labeled. Test on X_test, y_test
    model.fit(x_labeled, y_labeled, batch_size=20, epochs=100, validation_split=0.0, verbose=0)
    testerror = 1.0 - model.evaluate(X_test, y_test, verbose=0)[1]

    # ------- Class activation maps begins here --------
    def normalize(x):
        # utility function to normalize a tensor by its L2 norm
        return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

    # sample = np.array([x_labeled[0]])
    # prediction = model.predict(sample)
    # predicted_class = np.argmax(prediction)
    # print prediction, predicted_class
    # nb_classes = 6
    # target_layer = lambda x: tf.multiply(x, K.one_hot([predicted_class], nb_classes))
    # x = model.layers[-1].output
    # x = Lambda(target_layer, output_shape=lambda y: y)(x)
    # model_cam = Model(model.layers[0].input, x)
    # loss = K.sum(model_cam.layers[-1].output)
    # dense_input = model_cam.layers[0].input
    # grads = normalize(K.gradients(loss, dense_input)[0])
    # gradient_function = K.function([model_cam.layers[0].input], [dense_input, grads])

    sample = np.array([X_test[0]])
    # loss = K.categorical_crossentropy(y_labeled[0], model.layers[-1].output)
    loss = K.mean(K.square(model.layers[-1].output - y_test[0]), axis=-1)
    dense_input = model.layers[0].input
    grads = normalize(K.gradients(loss, dense_input)[0])
    gradient_function = K.function([model.layers[0].input], [dense_input, grads])

    output, grads_val = gradient_function([sample])
    print(np.shape(output), np.shape(grads_val))
    output, grads_val = output[0], grads_val[0]

    # class_activation_map = output * grads_val
    class_activation_map = np.abs(grads_val)
    print(np.shape(class_activation_map))
    print('Mean:', np.mean(class_activation_map), 'Max:', np.max(class_activation_map))
    # if np.max(class_activation_map) < 0:
    #     class_activation_map -= np.mean(class_activation_map)
    # class_activation_map = np.maximum(class_activation_map, 0)
    # heatmap = class_activation_map / np.max(class_activation_map)
    # Normalize class activation map
    class_activation_map = (class_activation_map - np.min(class_activation_map)) / (np.max(class_activation_map) - np.min(class_activation_map))
    heatmap = class_activation_map
    # heatmap = np.square(class_activation_map)
    print(np.min(class_activation_map), np.max(class_activation_map))

    plt.imshow(np.expand_dims(heatmap, axis=0), cmap='jet', aspect='auto')
    plt.plot((output - np.min(output)) / (np.max(output) - np.min(output)) - 0.5, 'w')
    plt.show()
    # ------- Class activation maps ends here --------

    return testerror

if __name__ == '__main__':
    modalities = ['Force', 'Temperature', 'Force and Temperature', 'Contact mic', 'Temperature and Contact Mic', 'Force, Temperature, and Contact Mic', 'Force and Contact Mic']

    # Test various amounts of labeled training data
    print('\n', '-'*25, 'Testing various amounts of labeled training data', '-'*25)
    print('-'*100)
    # for modality in [2, 5]:
    for modality in [2]:
        print('-'*25, modalities[modality], 'modality', '-'*25)
        X, y = dataset(modalities=modality)
        # for percent in [1, 2, 4, 8, 16, 50, 100]:
        for percent in [8]:
            print('-'*15, 'Percentage of training data labeled: %d%%' % percent, '-'*15)
            errors = []
            # Average over Stratified 6-fold. Training set: 6000, Test set: 1200
            skf = StratifiedKFold(n_splits=6, shuffle=True)
            for trainIdx, testIdx in skf.split(X, y):
                errors.append(mr_nn(None, None, percentlabeled=percent, trainTestSets=[X[trainIdx], X[testIdx], y[trainIdx], y[testIdx]], verbose=True))
                print('Test error:', errors[-1], 'Test accuracy:', 1.0-errors[-1])
                sys.stdout.flush()
            print('Average error:', np.mean(errors), 'Average accuracy:', np.mean(1.0-np.array(errors)))
            sys.stdout.flush()

