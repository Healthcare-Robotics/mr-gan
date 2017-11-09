import os, sys, time, glob, librosa, itertools, argparse
os.environ["KERAS_BACKEND"] = 'theano'
import numpy as np
import cPickle as pickle
from sklearn.utils import shuffle
from sklearn import preprocessing, decomposition
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

import keras
from keras.models import Model
from keras.layers import Input, Dense
from keras.layers.noise import GaussianNoise

'''
Material Recognition GAN
'''

def dataset(modalities=0, forcetempTime=4, contactmicTime=0.2, leaveObjectOut=False, verbose=False):
    materials = ['plastic', 'glass', 'fabric', 'metal', 'wood', 'ceramic']
    X = []
    y = []
    objects = dict()
    for m, material in enumerate(materials):
        if verbose:
            print 'Processing', material
            sys.stdout.flush()
        with open('data_processed/processed_0.1sbefore_%s_times_%.2f_%.2f.pkl' % (material, forcetempTime, contactmicTime), 'rb') as f:
            allData = pickle.load(f)
            for j, (objName, objData) in enumerate(allData.iteritems()):
                if leaveObjectOut:
                    objects[objName] = {'x': [], 'y': []}
                    X = objects[objName]['x']
                    y = objects[objName]['y']
                for i in xrange(len(objData['temperature'])):
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
            print 'X:', np.shape(X), 'y:', np.shape(y)
        return X, y

def mr_nn(X, y, percentlabeled=50, trainTestSets=None, verbose=False):
    # Non Deterministic output
    # np.random.seed(np.random.randint(1e9))

    materials = ['plastic', 'glass', 'fabric', 'metal', 'wood', 'ceramic']
    test_ratio = 200*len(materials)
    num_labeled_examples = int(10*percentlabeled)

    # Split into train and test sets
    if trainTestSets is None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, stratify=y)
    else:
        X_train, X_test, y_train, y_test = trainTestSets
    if verbose:
        print 'Num of class examples in test set:', [np.sum(y_test == i) for i in xrange(len(materials))]
        print 'X_train:', np.shape(X_train), 'y_train:', np.shape(y_train), 'X_test:', np.shape(X_test), 'y_test:', np.shape(y_test)

    # Scale data to zero mean and unit variance
    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Select labeled data
    X_train, y_train = shuffle(X_train, y_train)
    x_labeled = np.concatenate([X_train[y_train==j][:num_labeled_examples] for j in xrange(len(materials))], axis=0)
    y_labeled = np.concatenate([[j]*num_labeled_examples for j in xrange(len(materials))], axis=0)
    if verbose:
        print 'x_labeled:', np.shape(x_labeled), 'y_labeled:', np.shape(y_labeled)

    y_labeled = keras.utils.to_categorical(y_labeled, len(materials))
    y_test = keras.utils.to_categorical(y_test, len(materials))

    inputs = Input(shape=(X_train.shape[1],))
    x = GaussianNoise(0.3)(inputs)
    x = Dense(1000, activation='relu')(x)
    x = GaussianNoise(0.5)(x)
    x = Dense(500, activation='relu')(x)
    x = GaussianNoise(0.5)(x)
    x = Dense(250, activation='relu')(x)
    x = GaussianNoise(0.5)(x)
    x = Dense(250, activation='relu')(x)
    x = GaussianNoise(0.5)(x)
    x = Dense(250, activation='relu')(x)
    outputs = Dense(len(materials))(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    # Train on x_labeled, y_labeled. Test on X_test, y_test
    model.fit(x_labeled, y_labeled, batch_size=20, epochs=100, validation_split=0.0, verbose=0)
    testerror = 1.0 - model.evaluate(X_test, y_test, verbose=0)[1]
    return testerror

if __name__ == '__main__':
    modalities = ['Force', 'Temperature', 'Force and Temperature', 'Contact mic', 'Temperature and Contact Mic', 'Force, Temperature, and Contact Mic', 'Force and Contact Mic']

    parser = argparse.ArgumentParser(description='Collecting data from a spinning platter of objects.')
    parser.add_argument('-t', '--tables', nargs='+', help='[Required] Tables to recompute', required=True)
    parser.add_argument('-v', '--verbose', help='Verbose', action='store_true')
    args = parser.parse_args()

    if '2' in args.tables:
        # Test various amounts of labeled training data
        print '\n', '-'*25, 'Testing various amounts of labeled training data', '-'*25
        print '-'*100
        for modality in [2, 5]:
            np.random.seed(1000)
            print '-'*25, modalities[modality], 'modality', '-'*25
            X, y = dataset(modalities=modality)
            for percent in [1, 2, 4, 8, 16, 50, 100]:
                print '-'*15, 'Percentage of training data labeled: %d%%' % percent, '-'*15
                errors = []
                # Average over Stratified 6-fold. Training set: 6000, Test set: 1200
                skf = StratifiedKFold(n_splits=6, shuffle=True)
                for trainIdx, testIdx in skf.split(X, y):
                    errors.append(mr_nn(None, None, percentlabeled=percent, trainTestSets=[X[trainIdx], X[testIdx], y[trainIdx], y[testIdx]], verbose=args.verbose))
                    # print 'Test error:', errors[-1], 'Test accuracy:', 1.0-errors[-1]
                    sys.stdout.flush()
                print 'Average error:', np.mean(errors), 'Average accuracy:', np.mean(1.0-np.array(errors))
                sys.stdout.flush()

    if '4' in args.tables:
        # Test generalization with leave-one-object-out validation
        print '\n', '-'*25, 'Testing generalization with leave-one-object-out validation', '-'*25
        print '-'*100
        for modality in [2, 5]:
            np.random.seed(1000)
            print '-'*25, modalities[modality], 'modality', '-'*25
            objects = dataset(modalities=modality, leaveObjectOut=True)
            for percent in [1, 4, 16, 50, 100]:
                print '-'*15, 'Percentage of training data labeled: %d%%' % percent, '-'*15
                errors = []
                # Average over all 72 training and test set splits
                for objName, objData in objects.iteritems():
                    Xtest = np.array(objData['x'])
                    ytest = np.array(objData['y'])
                    Xtrain = np.array(list(itertools.chain.from_iterable([data['x'] for name, data in objects.iteritems() if name != objName])))
                    ytrain = np.array(list(itertools.chain.from_iterable([data['y'] for name, data in objects.iteritems() if name != objName])))
                    errors.append(mr_nn(None, None, percentlabeled=percent, trainTestSets=[Xtrain, Xtest, ytrain, ytest], verbose=args.verbose))
                    print objName, 'Test error:', errors[-1], 'Test accuracy:', 1.0-errors[-1]
                    sys.stdout.flush()
                print 'Average leave-one-object-out error:', np.mean(errors), 'Average accuracy:', np.mean(1.0-np.array(errors))
                sys.stdout.flush()

