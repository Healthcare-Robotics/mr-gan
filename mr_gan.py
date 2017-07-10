import os, sys, time, glob, librosa, itertools, argparse
os.environ["KERAS_BACKEND"] = 'theano'
import numpy as np
import cPickle as pickle
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from keras.models import Model, Sequential
from keras.layers import Input, Dense, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.noise import GaussianNoise
from keras.regularizers import l2
from keras.optimizers import Adam
from keras import losses
from keras import backend as K

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

def mr_gan(X, y, percentlabeled=50, percentunlabeled=None, epochs=100, trainTestSets=None, verbose=False):
    # Non Deterministic output
    np.random.seed(np.random.randint(1e9))

    noise_size = 100
    batchSize = 50
    unlabeled_weight = 1
    materials = ['plastic', 'glass', 'fabric', 'metal', 'wood', 'ceramic']
    test_ratio = 200*len(materials)
    num_labeled_examples = int(10*percentlabeled)
    if percentunlabeled is not None:
        num_unlabeled_examples = int(10*percentunlabeled)

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
    if percentunlabeled is not None:
        x_unlabeled = np.concatenate([X_train[y_train==j][:num_labeled_examples+num_unlabeled_examples] for j in xrange(len(materials))], axis=0)

    # Generator
    gen_input = Input(shape=(noise_size,))
    x = Dense(500, activation='softplus')(gen_input)
    x = BatchNormalization(epsilon=2e-5, momentum=0.9)(x)
    x = Dense(500, activation='softplus')(x)
    gen_output = Dense(X_train.shape[1])(x)

    # Discriminator
    disc_input = Input(shape=(X_train.shape[1],))
    x = GaussianNoise(0.3)(disc_input)
    x = Dense(1000, activation='relu')(x)
    x = GaussianNoise(0.5)(x)
    x = Dense(500, activation='relu')(x)
    x = GaussianNoise(0.5)(x)
    x = Dense(250, activation='relu')(x)
    x = GaussianNoise(0.5)(x)
    x = Dense(250, activation='relu')(x)
    x = GaussianNoise(0.5)(x)
    disc_mid_output = Dense(250, activation='relu')(x)
    disc_output = Dense(len(materials))(disc_mid_output)

    # Formal models to be used with placeholders
    generator = Model(inputs=gen_input, outputs=gen_output)
    discriminator = Model(inputs=disc_input, outputs=disc_output)
    mid_output = Model(inputs=disc_input, outputs=disc_mid_output)

    # Define placeholders for data input and output
    labels = K.placeholder(ndim=1, dtype='int32')
    x_lab = K.placeholder(shape=(None, X_train.shape[1]))
    x_unl = K.placeholder(shape=(None, X_train.shape[1]))
    x_noise = K.placeholder(shape=(None, noise_size))

    output_labeled = discriminator(x_lab)
    output_unlabeled = discriminator(x_unl)
    output_fake = discriminator(generator(x_noise))

    # Define loss for discriminator
    label_lab = output_labeled[K.arange(batchSize), labels]
    label_unl = K.logsumexp(output_unlabeled, axis=1)
    loss_lab = -K.mean(label_lab) + K.mean(K.logsumexp(output_labeled, axis=1))
    loss_unl = -0.5*K.mean(label_unl) + 0.5*K.mean(K.softplus(K.logsumexp(output_unlabeled, axis=1))) + 0.5*K.mean(K.softplus(K.logsumexp(output_fake, axis=1)))

    # Define loss for generator with feature matching
    mom_gen = K.mean(mid_output(generator(x_noise)), axis=0)
    mom_real = K.mean(mid_output(x_unl), axis=0)
    loss_gen = K.mean(K.square(mom_gen - mom_real))
    # No feature matching
    # mom_gen = K.mean(discriminator(generator(x_noise)), axis=0)
    # mom_real = K.mean(discriminator(x_unl), axis=0)
    # loss_gen = K.mean(K.square(mom_gen - mom_real))

    # Training and test errors
    train_err = K.mean(K.not_equal(K.argmax(output_labeled, axis=1), labels))
    test_err = K.mean(K.not_equal(K.argmax(output_labeled, axis=1), labels))

    # Define updates for weights based on loss functions
    adam = Adam(lr=0.0006, beta_1=0.5)
    disc_param_updates = adam.get_updates(discriminator.trainable_weights, discriminator.constraints, loss_lab + unlabeled_weight*loss_unl)
    gen_param_updates = adam.get_updates(generator.trainable_weights, generator.constraints, loss_gen)
    # Define training and test functions
    train_batch_disc = K.function(inputs=[K.learning_phase(), x_lab, labels, x_unl, x_noise], outputs=[loss_lab, loss_unl, train_err], updates=disc_param_updates)
    train_batch_gen = K.function(inputs=[K.learning_phase(), x_unl, x_noise], outputs=loss_gen, updates=gen_param_updates)
    test_batch = K.function(inputs=[K.learning_phase(), x_lab, labels], outputs=test_err)

    numBatchesTrain = X_train.shape[0] / batchSize
    numBatchesTest = X_test.shape[0] / batchSize
    train_phase = 1
    test_phase = 0
    if verbose:
        print 'Epochs:', epochs
        print 'Batch size:', batchSize
        print 'Training batches per epoch:', numBatchesTrain
        print 'Testing batches per epoch:', numBatchesTest

    for epoch in xrange(1, epochs+1):
        begin = time.time()
        loss_lab = 0.0
        loss_unl = 0.0
        train_err = 0.0
        # Random permutations of training data
        inds = np.concatenate([np.random.permutation(x_labeled.shape[0]) for _ in xrange(X_train.shape[0] / x_labeled.shape[0])] + [np.random.permutation(X_train.shape[0] % x_labeled.shape[0])])
        trainx = x_labeled[inds]
        trainy = y_labeled[inds]
        if percentunlabeled is None:
            trainx_unl = X_train[np.random.permutation(X_train.shape[0])]
            trainx_unl2 = X_train[np.random.permutation(X_train.shape[0])]
            trainx_unl3 = X_train[np.random.permutation(X_train.shape[0])]
        else:
            inds = np.concatenate([np.random.permutation(x_unlabeled.shape[0]) for _ in xrange(X_train.shape[0] / x_unlabeled.shape[0])] + [np.random.permutation(X_train.shape[0] % x_unlabeled.shape[0])])
            trainx_unl = x_unlabeled[inds]
            inds = np.concatenate([np.random.permutation(x_unlabeled.shape[0]) for _ in xrange(X_train.shape[0] / x_unlabeled.shape[0])] + [np.random.permutation(X_train.shape[0] % x_unlabeled.shape[0])])
            trainx_unl2 = x_unlabeled[inds]
            inds = np.concatenate([np.random.permutation(x_unlabeled.shape[0]) for _ in xrange(X_train.shape[0] / x_unlabeled.shape[0])] + [np.random.permutation(X_train.shape[0] % x_unlabeled.shape[0])])
            trainx_unl3 = x_unlabeled[inds]

        for t in xrange(numBatchesTrain):
            # Train discriminator
            noise = np.random.normal(0, 1, size=[batchSize, noise_size])
            ll, lu, te = train_batch_disc([train_phase, trainx[t*batchSize:(t+1)*batchSize], trainy[t*batchSize:(t+1)*batchSize], trainx_unl[t*batchSize:(t+1)*batchSize], noise])
            loss_lab += ll
            loss_unl += lu
            train_err += te
            # Train generator
            noise = np.random.normal(0, 1, size=[batchSize, noise_size])
            loss = train_batch_gen([train_phase, trainx_unl2[t*batchSize:(t+1)*batchSize], noise])

        loss_lab /= numBatchesTrain
        loss_unl /= numBatchesTrain
        train_err /= numBatchesTrain

        # Test discriminator
        test_err = 0.0
        for t in xrange(numBatchesTest):
            test_err += test_batch([test_phase, X_test[t*batchSize:(t+1)*batchSize], y_test[t*batchSize:(t+1)*batchSize]])
        test_err /= numBatchesTest

        # Report results
        if verbose:
            print 'Epoch %d, time = %ds, loss labeled = %.4f, loss unlabeled = %.4f, train error = %.4f, test error = %.4f' % (epoch, time.time()-begin, loss_lab, loss_unl, train_err, test_err)
            sys.stdout.flush()

    testerror = test_batch([test_phase, X_test, y_test])
    if verbose:
        print 'Test error:', testerror, test_err
        sys.stdout.flush()
    return testerror

if __name__ == '__main__':
    modalities = ['Force', 'Temperature', 'Force and Temperature', 'Contact mic', 'Temperature and Contact Mic', 'Force, Temperature, and Contact Mic', 'Force and Contact Mic']

    parser = argparse.ArgumentParser(description='Collecting data from a spinning platter of objects.')
    parser.add_argument('-t', '--tables', nargs='+', help='[Required] Tables to recompute', required=True)
    parser.add_argument('-v', '--verbose', help='Verbose', action='store_true')
    args = parser.parse_args()

    if '1' in args.tables:
        # Test various amounts of labeled training data
        print '\n', '-'*25, 'Testing various amounts of labeled training data', '-'*25
        print '-'*100
        for modality in xrange(len(modalities)):
            print '-'*25, modalities[modality], 'modality', '-'*25
            X, y = dataset(modalities=modality)
            for percent in [1, 2, 4, 8, 16, 50, 100]:
                print '-'*15, 'Percentage of training data labeled: %d%%' % percent, '-'*15
                errors = []
                # Average over Stratified 6-fold. Training set: 6000, Test set: 1200
                skf = StratifiedKFold(n_splits=6, shuffle=True)
                for trainIdx, testIdx in skf.split(X, y):
                    errors.append(mr_gan(None, None, percentlabeled=percent, trainTestSets=[X[trainIdx], X[testIdx], y[trainIdx], y[testIdx]], verbose=args.verbose))
                    print 'Test error:', errors[-1], 'Test accuracy:', 1.0-errors[-1]
                    sys.stdout.flush()
                print 'Average error:', np.mean(errors), 'Average accuracy:', np.mean(1.0-np.array(errors))
                sys.stdout.flush()

    if '2' in args.tables:
        # Test generalization with leave-one-object-out validation
        print '\n', '-'*25, 'Testing generalization with leave-one-object-out validation', '-'*25
        print '-'*100
        for modality in [2, 5]:
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
                    errors.append(mr_gan(None, None, percentlabeled=percent, trainTestSets=[Xtrain, Xtest, ytrain, ytest], verbose=args.verbose))
                    print objName, 'Test error:', errors[-1], 'Test accuracy:', 1.0-errors[-1]
                    sys.stdout.flush()
                print 'Average leave-one-object-out error:', np.mean(errors), 'Average accuracy:', np.mean(1.0-np.array(errors))
                sys.stdout.flush()

    if '3' in args.tables:
        # Test various lengths of contact time in training data
        print '\n', '-'*25, 'Testing various lengths of contact time in training data', '-'*25
        print '-'*100
        for modality in xrange(3):
            print '-'*25, modalities[modality], 'modality', '-'*25
            for ftTime in [4, 3, 2, 1, 0.5, 0.2, 0.1]:
                print '-'*15, 'Length of training data: %.1fs' % ftTime, '-'*15
                X, y = dataset(modalities=modality, forcetempTime=ftTime)
                errors = []
                # Average over Stratified 6-fold. Training set: 6000, Test set: 1200
                skf = StratifiedKFold(n_splits=6, shuffle=True)
                for trainIdx, testIdx in skf.split(X, y):
                    errors.append(mr_gan(None, None, percentlabeled=100, trainTestSets=[X[trainIdx], X[testIdx], y[trainIdx], y[testIdx]], verbose=args.verbose))
                    print 'Test error:', errors[-1], 'Test accuracy:', 1.0-errors[-1]
                    sys.stdout.flush()
                print 'Average error:', np.mean(errors), 'Average accuracy:', np.mean(1.0-np.array(errors))
                sys.stdout.flush()

        print '\n', '-'*25, 'Testing various lengths of contact time in training data', '-'*25
        print '-'*100
        print '-'*25, modalities[3], 'modality', '-'*25
        for cTime in [1, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05]:
            print '-'*15, 'Length of training data: %.1fs' % cTime, '-'*15
            X, y = dataset(modalities=3, contactmicTime=cTime)
            errors = []
            # Average over Stratified 6-fold. Training set: 6000, Test set: 1200
            skf = StratifiedKFold(n_splits=6, shuffle=True)
            for trainIdx, testIdx in skf.split(X, y):
                errors.append(mr_gan(None, None, percentlabeled=100, trainTestSets=[X[trainIdx], X[testIdx], y[trainIdx], y[testIdx]], verbose=args.verbose))
                print 'Test error:', errors[-1], 'Test accuracy:', 1.0-errors[-1]
                sys.stdout.flush()
            print 'Average error:', np.mean(errors), 'Average accuracy:', np.mean(1.0-np.array(errors))
            sys.stdout.flush()

    if '4' in args.tables:
        # Test performance as amount of unlabeled data increases and amount of labeled data remains fixed
        print '\n', '-'*25, 'Testing performance as quantity of unlabeled data increases', '-'*25
        print '-'*100
        for modality in [2, 5]:
            print '-'*25, modalities[modality], 'modality', '-'*25
            X, y = dataset(modalities=modality)
            # for percentlabeled in [4, 8, 16]:
            for percentlabeled in [4]:
                print '-'*15, 'Percentage of training data labeled: %d%%' % percentlabeled, '-'*15
                for percentunlabeled in [0, 4, 8, 16, 32, 64, 100-percentlabeled]:
                    # Unlabeled examples per class: 0, 40, 80, 160, 320, 640, 960
                    print '-'*15, 'Percentage of training data unlabeled: %d%%' % percentunlabeled, '-'*15
                    errors = []
                    # Average over Stratified 6-fold. Training set: 6000, Test set: 1200
                    skf = StratifiedKFold(n_splits=6, shuffle=True)
                    for trainIdx, testIdx in skf.split(X, y):
                        errors.append(mr_gan(None, None, percentlabeled=percentlabeled, percentunlabeled=percentunlabeled, trainTestSets=[X[trainIdx], X[testIdx], y[trainIdx], y[testIdx]], verbose=args.verbose))
                        print 'Test error:', errors[-1], 'Test accuracy:', 1.0-errors[-1]
                        sys.stdout.flush()
                    print 'Average error:', np.mean(errors), 'Average accuracy:', np.mean(1.0-np.array(errors))
                    sys.stdout.flush()

