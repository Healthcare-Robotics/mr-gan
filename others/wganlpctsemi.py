import os, sys, time, glob, librosa, itertools, argparse, random, gc
import numpy as np
import tensorflow as tf
os.environ['KERAS_BACKEND'] = 'tensorflow'
# os.environ['KERAS_BACKEND'] = 'theano'
# Enforce reproducible results
os.environ['PYTHONHASHSEED'] = '54321'
np.random.seed(54321)
random.seed(54321)
from keras import backend as K
# Force TF and Keras to use a single thread. Multiple threads lead to non-reproducible results
# TODO: Use for NN and biLSTM. Is this necessary for GANs?
if os.environ['KERAS_BACKEND'] == 'tensorflow':
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    tf.set_random_seed(54321)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)
import cPickle as pickle
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.utils import shuffle
from sklearn import preprocessing, decomposition, gaussian_process, ensemble
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

import keras
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Activation, Dropout, LSTM, Bidirectional, Reshape, Conv1D, MaxPooling1D, Flatten, Concatenate, Lambda, Add, Merge, LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.noise import GaussianNoise
from keras.regularizers import l2
from keras.optimizers import Adam, Optimizer
from keras import losses
from keras.legacy import interfaces

import util
# from adamw_amsgrad import AdamW_Amsgrad


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

def preprocess(X, y, wavelengths, uvir=None, deriv_log_preprocess=None, doubleData=False):
    X = np.copy(X)
    y = np.copy(y)
    wavelengths = np.copy(wavelengths)

    if uvir == 'uv':
        # Ultraviolet light spectrum
        X = X[:, wavelengths < 400]
        wavelengths = wavelengths[wavelengths < 400]
    elif uvir == 'ir':
        # Near-infrared light spectrum
        X = X[:, wavelengths > 700]
        wavelengths = wavelengths[wavelengths > 700]

    if deriv_log_preprocess is None:
        return X, y, wavelengths

    if 'log' in deriv_log_preprocess:
        # Natural log
        for _ in xrange(int(deriv_log_preprocess[-1])):
            X = np.ma.log(X).filled(0)
    elif 'preprocess' in deriv_log_preprocess:
        # Natural log, first derivative, then remove average
        for _ in xrange(int(deriv_log_preprocess[-1])):
            X = np.ma.log(X).filled(0)
            if not doubleData:
                X = util.firstDeriv(X, wavelengths)
            else:
                X = np.concatenate([util.firstDeriv(X[:, :len(wavelengths)], wavelengths), util.firstDeriv(X[:, len(wavelengths):], wavelengths)], axis=-1)
            X -= np.expand_dims(np.mean(X, axis=-1), axis=-1)
    elif 'deriv' in deriv_log_preprocess:
        # Take derivative of features
        for _ in xrange(int(deriv_log_preprocess[-1])):
            if not doubleData:
                X = util.firstDeriv(X, wavelengths)
            else:
                X = np.concatenate([util.firstDeriv(X[:, :len(wavelengths)], wavelengths), util.firstDeriv(X[:, len(wavelengths):], wavelengths)], axis=-1)

    return X, y, wavelengths

def pcaScale(Xtrain, Xtest, PCA=0, scale=None):
    if PCA > 0:
        pca = decomposition.PCA(n_components=PCA)
        Xtrain = pca.fit_transform(Xtrain)
        Xtest = pca.transform(Xtest)

    if scale is not None:
        scaler = preprocessing.Normalizer() if scale == 'norm' else preprocessing.StandardScaler()
        Xtrain = scaler.fit_transform(Xtrain)
        Xtest = scaler.transform(Xtest)
    return Xtrain, Xtest

def learnNNSVM(Xtrain, ytrain, Xtest, ytest, percentLabeled=1.0, epochs=100, batchSize=64, materialCount=5, PCA=0, scale=None, verbose=False, kernel=0, algorithm='svm'):
    np.random.seed(54321)
    random.seed(54321)
    tf.set_random_seed(54321)
    if os.environ['KERAS_BACKEND'] == 'tensorflow':
        session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        tf.set_random_seed(54321)
        sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        K.set_session(sess)
    Xtrain, Xtest = pcaScale(Xtrain, Xtest, PCA=PCA, scale=scale)

    # Select labeled data
    # numLabeled = int(len(ytrain[ytrain==0]) * percentLabeled)
    Xtrain, ytrain = shuffle(Xtrain, ytrain)
    x_labeled = np.concatenate([Xtrain[ytrain==j][:int(len(ytrain[ytrain==j]) * percentLabeled)] for j in xrange(materialCount)], axis=0)
    y_labeled = np.concatenate([[j]*int(len(ytrain[ytrain==j]) * percentLabeled) for j in xrange(materialCount)], axis=0)
    if verbose:
        print percentLabeled, 'x_labeled:', np.shape(x_labeled), 'y_labeled:', np.shape(y_labeled), 'Xtrain:', np.shape(Xtrain), 'ytrain:', np.shape(ytrain)
    x_labeled, y_labeled = shuffle(x_labeled, y_labeled)

    if algorithm == 'nn':
        y_labeled = keras.utils.to_categorical(y_labeled, num_classes=materialCount)
        ytest = keras.utils.to_categorical(ytest, num_classes=materialCount)

        epochs = 200
        disc_input = Input(shape=(Xtrain.shape[1],))
        x = Dense(Xtrain.shape[1], activation='linear')(disc_input)
        x = Add()([disc_input, x])
        x1 = LeakyReLU()(x)
        x = Dropout(0.2)(x1)
        x = Dense(Xtrain.shape[1], activation='linear')(x)
        x = Add()([x1, x])
        x1 = LeakyReLU()(x)
        x = Dropout(0.2)(x1)
        x = Dense(Xtrain.shape[1], activation='linear')(x)
        x = Add()([x1, x])
        x1 = LeakyReLU()(x)
        x = Dropout(0.2)(x1)
        disc_output = Dense(materialCount, activation='softmax')(x)
        model = Model(inputs=disc_input, outputs=disc_output)

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.fit(x_labeled, y_labeled, epochs=epochs, batch_size=batchSize, validation_split=0.0, verbose=(1 if verbose else 0))
        # NOTE: Return accuracy
        return model.evaluate(Xtest, ytest, batch_size=batchSize, verbose=0)[-1]
    elif algorithm == 'lstm':
        y_labeled = keras.utils.to_categorical(y_labeled, num_classes=materialCount)
        ytest = keras.utils.to_categorical(ytest, num_classes=materialCount)

        epochs = 100
        model = Sequential()
        model.add(Bidirectional(LSTM(16, activation='tanh', return_sequences=True), input_shape=(np.shape(Xtrain)[-1], 1)))
        model.add(Bidirectional(LSTM(16, activation='tanh', return_sequences=True)))
        model.add(Bidirectional(LSTM(16, activation='tanh', return_sequences=False)))
        model.add(Dense(materialCount, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0025), metrics=['accuracy'])

        model.fit(np.expand_dims(x_labeled, axis=-1), y_labeled, epochs=epochs, batch_size=batchSize, validation_split=0.0, verbose=(1 if verbose else 0))
        # NOTE: Return accuracy
        return model.evaluate(np.expand_dims(Xtest, axis=-1), ytest, batch_size=batchSize, verbose=0)[-1]
    elif algorithm == 'svm':
        if kernel == 0: svm = SVC(kernel='rbf')
        elif kernel == 1: svm = SVC(kernel='linear')
        elif kernel == 2: svm = NuSVC(kernel='rbf')
        elif kernel == 3: svm = NuSVC(kernel='linear')
        elif kernel == 4: svm = LinearSVC()

        svm.fit(x_labeled, y_labeled)

        # NOTE: Return accuracy
        return svm.score(Xtest, ytest)
    elif algorithm == 'rf':
        model = ensemble.RandomForestClassifier(n_estimators=10)

        model.fit(x_labeled, y_labeled)

        # NOTE: Return accuracy
        return model.score(Xtest, ytest)


def l2_distance(x1, x2):
    return K.sqrt(K.sum(K.square(x1 - x2), axis=-1))

def learnGAN(Xtrain, ytrain, Xtest, ytest, percentLabeled=1.0, epochs=100, noise_size=100, batchSize=64, materialCount=5, PCA=0, scale=None, verbose=False, algorithm='gan', discTrainIters=1, genTrainIters=1):
    np.random.seed(54321)
    random.seed(54321)
    tf.set_random_seed(54321)
    if os.environ['KERAS_BACKEND'] == 'tensorflow':
        session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        tf.set_random_seed(54321)
        sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        K.set_session(sess)
    Xtrain, Xtest = pcaScale(Xtrain, Xtest, PCA=PCA, scale=scale)

    # Select labeled data
    # numLabeled = int(len(ytrain[ytrain==0]) * percentLabeled)
    Xtrain, ytrain = shuffle(Xtrain, ytrain)
    x_labeled = np.concatenate([Xtrain[ytrain==j][:int(len(ytrain[ytrain==j]) * percentLabeled)] for j in xrange(materialCount)], axis=0)
    y_labeled = np.concatenate([[j]*int(len(ytrain[ytrain==j]) * percentLabeled) for j in xrange(materialCount)], axis=0)
    if verbose:
        print percentLabeled, 'x_labeled:', np.shape(x_labeled), 'y_labeled:', np.shape(y_labeled), 'Xtrain:', np.shape(Xtrain), 'ytrain:', np.shape(ytrain)

    if algorithm == 'iwgan':
        gen_input = Input(shape=(noise_size,))
        x = Dense(64, activation='softplus')(gen_input)
        x = Dense(64, activation='softplus')(x)
        gen_output = Dense(Xtrain.shape[1])(x)

        # lamb2 = 2.0, lr = 0.00001
        # epochs = 200
        # disc_input = Input(shape=(Xtrain.shape[1],))
        # x = Dense(Xtrain.shape[1], activation='linear')(disc_input)
        # x = Add()([disc_input, x])
        # x1 = LeakyReLU()(x)
        # x = Dropout(0.2)(x1)
        # x = Dense(Xtrain.shape[1], activation='linear')(x)
        # x = Add()([x1, x])
        # x1 = LeakyReLU()(x)
        # x = Dropout(0.2)(x1)
        # x = Dense(Xtrain.shape[1], activation='linear')(x)
        # x = Add()([x1, x])
        # disc_mid_output = LeakyReLU()(x)
        # x = Dropout(0.2)(disc_mid_output)
        # disc_output = Dense(materialCount, activation='linear')(x)

        # lamb2 = 0.5, lr = 0.0001
        # Best score: 0.701111, python wganlpctsemi.py -t 0 -v -a iwgan # lr = 0.0002, lamb2 = 0.5, 64 nodes, scale
        # Best score: 0.699583, python wganlpctsemi.py -t 0 -v -a iwgan # lr = 0.0001, lamb2 = 2.0, 64 nodes, scale, dropout 0.4
        epochs = 200
        disc_input = Input(shape=(Xtrain.shape[1],))
        x = Dense(128, activation='linear')(disc_input)
        x1 = LeakyReLU()(x)
        x = Dropout(0.4)(x1)
        x = Dense(128, activation='linear')(x)
        x = Add()([x1, x])
        x1 = LeakyReLU()(x)
        x = Dropout(0.4)(x1)
        x = Dense(128, activation='linear')(x)
        x = Add()([x1, x])
        x1 = LeakyReLU()(x)
        x = Dropout(0.4)(x1)
        x = Dense(128, activation='linear')(x)
        x = Add()([x1, x])
        x1 = LeakyReLU()(x)
        x = Dropout(0.4)(x1)
        x = Dense(128, activation='linear')(x)
        x = Add()([x1, x])
        disc_mid_output = LeakyReLU()(x)
        x = Dropout(0.4)(disc_mid_output)
        disc_output = Dense(materialCount, activation='linear')(x)

    elif algorithm == 'iwganlstm':
        gen_input = Input(shape=(noise_size,))
        x = Dense(16, activation='softplus')(gen_input)
        # x = BatchNormalization(epsilon=2e-5, momentum=0.9)(x)
        x = Dense(16, activation='softplus')(x)
        gen_output = Dense(Xtrain.shape[1])(x)

        # noise_size = Xtrain.shape[1] # TODO: Try removing this and using a noise size of 100
        # gen_input = Input(shape=(noise_size,))
        # x = Reshape((noise_size, 1))(gen_input)
        # # x = Bidirectional(LSTM(4, activation='tanh', return_sequences=True))(x)
        # # gen_output = LSTM(1, activation='tanh', return_sequences=True)(x)
        # x = Bidirectional(LSTM(4, activation='tanh', return_sequences=False))(x)
        # gen_output = Dense(Xtrain.shape[1])(x)

        # Discriminator
        disc_input = Input(shape=(Xtrain.shape[1],))
        x = Reshape((Xtrain.shape[1], 1))(disc_input)
        # x = Bidirectional(LSTM(4, activation='tanh', return_sequences=True))(x)
        # x = GaussianNoise(0.2)(x)
        disc_mid_output = Bidirectional(LSTM(4, activation='tanh', return_sequences=False))(x)
        # x = GaussianNoise(0.2)(disc_mid_output)
        # disc_mid_output = Dense(250, activation='relu')(x)
        disc_output = Dense(materialCount)(disc_mid_output)

    # Formal models to be used with placeholders
    generator = Model(inputs=gen_input, outputs=gen_output)
    generator2 = Model(inputs=gen_input, outputs=gen_output)
    discriminator = Model(inputs=disc_input, outputs=disc_output)
    discriminator2 = Model(inputs=disc_input, outputs=disc_output)
    if algorithm in ['gan', 'iwgan', 'iwganlstm']:
        mid_output = Model(inputs=disc_input, outputs=disc_mid_output)
        mid_output2 = Model(inputs=disc_input, outputs=disc_mid_output)

    # Define placeholders for data input and output
    labels = K.placeholder(ndim=1, dtype='int32')
    x_lab = K.placeholder(shape=(None, Xtrain.shape[1]))
    epsilon = K.placeholder(shape=(None, Xtrain.shape[1]))
    x_unl = K.placeholder(shape=(None, Xtrain.shape[1]))
    x_noise = K.placeholder(shape=(None, noise_size))

    output_labeled = discriminator(x_lab)
    output_unlabeled = discriminator(x_unl)
    output_fake = discriminator(generator(x_noise))

    if algorithm in ['gan', 'ganlstm']:
        index_mask = K.reshape(K.one_hot(labels, materialCount), [-1, materialCount])
        label_lab = K.sum(output_labeled*index_mask, axis=1)
        loss_lab = -K.mean(label_lab) + K.mean(K.logsumexp(output_labeled, axis=1)) + 0.5*K.mean(K.softplus(K.logsumexp(output_fake, axis=1)))
        # loss_lab = K.mean(K.categorical_crossentropy(K.one_hot(labels, materialCount), output_labeled)) + K.mean(output_fake)
        # loss_lab = K.mean(K.categorical_crossentropy(K.one_hot(labels, materialCount), output_labeled)) + 0.5*K.mean(K.softplus(K.logsumexp(output_fake, axis=1)))
    elif algorithm in ['iwgan', 'iwganlstm']:
        # Compute the gradient penalty loss
        # Matches the loss from the original WGAN-GP / iWGAN paper (line 7 of algorithm 1)
        # Also found on WGAN-GP implementation: https://github.com/tjwei/GANotebooks/blob/master/wgan2-keras.ipynb
        # lamb = 10 is used in original WGAN-GP paper. The follow paper suggests lamb = 750: https://arxiv.org/pdf/1710.10196.pdf

        lamb = 10 if algorithm == 'iwgan' else 5
        lamb2 = 2.0
        # mixedInput = epsilon * x_lab + (1-epsilon) * generator(x_noise)
        # gradientMixed = K.gradients(discriminator(mixedInput), [mixedInput])[0]
        mixedInput = Input(shape=(Xtrain.shape[1],), tensor=epsilon * x_unl + (1-epsilon) * generator(x_noise))
        gradientMixed = K.gradients(K.mean(discriminator(mixedInput)), [mixedInput])[0]
        gradientMixedNorm = K.l2_normalize(gradientMixed, axis=1)
        # gradientPenalty = K.mean(K.square(gradientMixedNorm - 1.0))
        gradientPenalty = K.mean(K.square(K.maximum(K.zeros_like(gradientMixedNorm, dtype='float32'), gradientMixedNorm - 1.0))) # WGAN-LP, Lipschitz Penalty, Henning Petzka, Asja Fischer, and Denis Lukovnicov. "On the regularization of Wasserstein GANs."
        # ct = l2_distance(K.softmax(discriminator(x_unl)), K.softmax(discriminator2(x_unl))) + 0.1 * l2_distance(mid_output(x_unl), mid_output2(x_unl))
        d2 = discriminator2(x_unl)
        m2 = mid_output2(x_unl)
        ct = l2_distance(K.softmax(discriminator(x_unl)), K.softmax(d2) + K.random_normal(K.shape(d2), mean=0.0, stddev=0.0001)) + 0.1 * l2_distance(mid_output(x_unl), m2 + K.random_normal(K.shape(m2), mean=0.0, stddev=0.0001))
        # ct = l2_distance(discriminator(generator(x_noise)), discriminator2(generator(x_noise))) + 0.1 * l2_distance(mid_output(generator(x_noise)), mid_output2(generator(x_noise)))
        # ct = l2_distance(K.softmax(discriminator(generator(x_noise))), K.softmax(discriminator2(generator(x_noise)))) + 0.1 * l2_distance(mid_output(generator(x_noise)), mid_output2(generator(x_noise)))

        consistencyTerm = K.mean(K.maximum(K.zeros_like(ct, dtype='float32'), ct))
        index_mask = K.reshape(K.one_hot(labels, materialCount), [-1, materialCount])
        label_lab = K.sum(output_labeled*index_mask, axis=1)
        label_unl = K.logsumexp(output_unlabeled, axis=1)

        loss_lab = -K.mean(label_lab) + K.mean(K.logsumexp(output_labeled, axis=1))
        loss_unl = -K.mean(label_unl) + K.mean(K.softplus(K.logsumexp(output_unlabeled, axis=1))) + K.mean(K.softplus(K.logsumexp(output_fake, axis=1)))

        # classification_loss = loss_lab + loss_unl + lamb * gradientPenalty
        classification_loss = loss_lab + loss_unl + lamb * gradientPenalty + lamb2 * consistencyTerm

    if algorithm == 'gan':
        # Define loss for generator with feature matching
        mid_gen = K.mean(mid_output(generator(x_noise)), axis=0)
        mid_real = K.mean(mid_output(x_unl), axis=0)
        loss_gen = K.mean(K.square(mid_gen - mid_real))
    elif algorithm == 'ganlstm':
        # No feature matching
        mid_gen = K.mean(discriminator(generator(x_noise)), axis=0)
        mid_real = K.mean(discriminator(x_unl), axis=0)
        loss_gen = K.mean(K.square(mid_gen - mid_real))
    elif algorithm == 'iwgan' or algorithm == 'iwganlstm':
        # NOTE: This may not be valid to define Wasserstein distance over the output of a middle layer. Is this 1-lipschitz?
        # NOTE: This loss is suggested by: https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py
        mid_gen2 = mid_output(generator(x_noise))
        mid_real2 = mid_output(x_unl)
        # loss_gen = K.mean(mid_real * mid_gen)
        # NOTE: The below loss follows the loss used in the original WGAN-GP / iWGAN paper (line 12  in algorithm 1)
        mid_gen = discriminator(generator(x_noise))
        mid_real = discriminator(x_unl)
        # loss_gen = -K.mean(mid_gen) + K.mean(K.square(mid_gen2 - mid_real2))
        loss_gen = -K.mean(mid_gen)

    # Training and test errors
    accuracy = K.mean(K.equal(K.cast(K.argmax(output_labeled, axis=1), dtype='int32'), labels))

    # Define updates for weights based on loss functions
    # adam = Adam(amsgrad=True)
    if algorithm == 'iwgan':
        adam = Adam(0.0005, beta_1=0.5, beta_2=0.9)
        # adam = AdamW_Amsgrad(lr=0.0001, beta_1=0.5, beta_2=0.9, weight_decay=0.001, amsgrad=False)
    elif algorithm == 'iwganlstm':
        adam = Adam(0.001, beta_1=0.5, beta_2=0.9)

    disc_param_updates = adam.get_updates(params=discriminator.trainable_weights, loss=classification_loss)
    gen_param_updates = adam.get_updates(params=generator.trainable_weights, loss=loss_gen)
    # Define training and test functions
    train_batch_disc = K.function(inputs=[K.learning_phase(), x_lab, labels, x_unl, x_noise, epsilon], outputs=[classification_loss, loss_gen, accuracy], updates=disc_param_updates)
    train_batch_gen = K.function(inputs=[K.learning_phase(), x_unl, x_noise], outputs=[loss_gen], updates=gen_param_updates)
    test_batch = K.function(inputs=[K.learning_phase(), x_lab, labels], outputs=[accuracy])

    if Xtrain.shape[0] < batchSize:
        batchSize = Xtrain.shape[0]
    numBatchesTrain = Xtrain.shape[0] / batchSize
    numBatchesTest = Xtest.shape[0] / batchSize
    train_phase = 1
    test_phase = 0
    if verbose:
        print 'Epochs:', epochs
        print 'Batch size:', batchSize
        print 'Training batches per epoch:', numBatchesTrain
        print 'Testing batches per epoch:', numBatchesTest
        sys.stdout.flush()

    dLosses = []
    gLosses = []
    for epoch in xrange(1, epochs+1):
        begin = time.time()
        total_loss_lab = 0.0
        total_loss_gen = 0.0
        train_acc = 0.0

        # Setup data for each discriminator training iteration.
        trainDiscX = []
        trainDiscy = []
        trainDiscXunl = []
        for _ in xrange(discTrainIters):
            # Random permutations of training data
            inds = np.concatenate([np.random.permutation(x_labeled.shape[0]) for _ in xrange(Xtrain.shape[0] / x_labeled.shape[0])] + [np.random.permutation(Xtrain.shape[0] % x_labeled.shape[0])])
            trainDiscX.append(x_labeled[inds])
            trainDiscy.append(y_labeled[inds])
            trainDiscXunl.append(Xtrain[np.random.permutation(Xtrain.shape[0])])

        # Setup data for each generator training iteration.
        trainGenX = []
        for _ in xrange(genTrainIters):
            trainGenX.append(Xtrain[np.random.permutation(Xtrain.shape[0])])

        for t in xrange(numBatchesTrain):
            # Train discriminator
            for tx, ty, txunl in zip(trainDiscX, trainDiscy, trainDiscXunl):
                noise = np.random.normal(0, 1, size=[batchSize, noise_size])
                e = np.repeat(np.random.uniform(0, 1, size=[batchSize, 1]), Xtrain.shape[1], axis=-1)
                ll, lg, ta = train_batch_disc([train_phase, tx[t*batchSize:(t+1)*batchSize], ty[t*batchSize:(t+1)*batchSize], txunl[t*batchSize:(t+1)*batchSize], noise, e])
            total_loss_lab += ll
            total_loss_gen += lg
            train_acc += ta

            # Train generator
            for tx in trainGenX:
                noise = np.random.normal(0, 1, size=[batchSize, noise_size])
                loss = train_batch_gen([train_phase, tx[t*batchSize:(t+1)*batchSize], noise])[0]

        # Train on the remaining data not in the above batches
        remainSize = Xtrain.shape[0] % batchSize
        if remainSize != 0 and Xtrain.shape[0] >= batchSize:
            # Train discriminator
            for tx, ty, txunl in zip(trainDiscX, trainDiscy, trainDiscXunl):
                noise = np.random.normal(0, 1, size=[remainSize, noise_size])
                e = np.repeat(np.random.uniform(0, 1, size=[remainSize, 1]), Xtrain.shape[1], axis=-1)
                ll, lg, ta = train_batch_disc([train_phase, tx[(t+1)*batchSize:], ty[(t+1)*batchSize:], txunl[(t+1)*batchSize:], noise, e])

            # Train generator
            for tx in trainGenX:
                noise = np.random.normal(0, 1, size=[remainSize, noise_size])
                loss = train_batch_gen([train_phase, tx[(t+1)*batchSize:], noise])[0]

        total_loss_lab /= numBatchesTrain
        total_loss_gen /= numBatchesTrain
        train_acc /= numBatchesTrain

        test_acc = test_batch([test_phase, Xtest, ytest])[0]

        # predicts = discriminator.predict(Xtrain)
        # print np.argmax(predicts, axis=-1), np.mean(np.equal(np.argmax(predicts, axis=-1), y_train))

        # Report results
        if verbose:
            print 'Epoch %d, time = %ds, loss labeled = %.4f, loss generator = %.4f, train accuracy = %.4f, test accuracy = %.4f' % (epoch, time.time()-begin, total_loss_lab, total_loss_gen, train_acc, test_acc)
            sys.stdout.flush()

        # Store loss of most recent batch from this epoch
        # dLosses.append(total_loss_lab)
        # gLosses.append(total_loss_gen)

    testaccuracy = test_batch([test_phase, Xtest, ytest])[0]
    if verbose:
        print 'Test accuracy:', testaccuracy, test_acc
        sys.stdout.flush()
    return testaccuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Semi-supervised learning with GANs for material recognition on haptic data.')
    parser.add_argument('-t', '--test', nargs='+', help='Which test? (1) K-fold CV, (2) Generalizing to many new objects, (3) Leave-one-object-out', required=True)
    parser.add_argument('-a', '--algorithm', nargs='+', help='svm, nn, lstm, rf, gan, iwgan, iwganlstm', required=True)
    parser.add_argument('-v', '--verbose', help='Verbose', action='store_true')
    args = parser.parse_args()
    algorithm = args.algorithm[0]

    modalities = ['Force', 'Temperature', 'Force and Temperature', 'Contact mic', 'Temperature and Contact Mic', 'Force, Temperature, and Contact Mic', 'Force and Contact Mic']
    X, y = dataset(modalities=2)

    t = time.time()
    # bestScore = 0
    # bestParameters = []
    if '0' in args.test:
        # K-fold cross validation
        n_splits = 5
        epochs = 100
        batchSize = 64

        if algorithm == 'iwgan':
            gridSearch = [['scio', 'lumini'], ['spectrum'], [100, 200, 300, 400, 500], ['deriv1', 'deriv2', 'preprocess1', 'log1', None], [0], ['norm', 'scale'], [None]]
            gridSearch = [[0], ['scale'], [None]]
            epochs = 100
            batchSize = 64
            print 'Training with a WGAN-GP / iWGAN'
        elif algorithm == 'iwganlstm':
            # biLSTM
            gridSearch = [['scio', 'lumini'], ['spectrum'], [100, 200, 300, 400, 500], ['deriv1', 'deriv2', 'preprocess1', 'log1', None], [0], ['norm', 'scale'], [None]]
            gridSearch = [['scio'], ['spectrum'], [100, 200, 300, 400, 500], ['deriv1'], [0], ['norm'], [None]]
            epochs = 100
            batchSize = 128
            print 'Training with a WGAN-LP-CT biLSTM'
        elif algorithm == 'lstm':
            # biLSTM
            gridSearch = [['scio'], ['spectrum'], [100, 200, 300, 400, 500], ['deriv1'], [0], ['scale'], [None]]
            epochs = 100
            batchSize = 128
            print 'Training with a biLSTM'
        elif algorithm == 'nn':
            gridSearch = [['scio', 'lumini'], ['spectrum_raw', 'spectrum'], [100, 200, 300, 400, 500], ['deriv1', 'deriv2', 'preprocess1', 'log1', None], [0, 10], ['norm', 'scale'], [None]]
            gridSearch = [['scio'], ['spectrum'], [100, 200, 300, 400, 500], ['deriv1'], [0], ['norm'], [None]]
            epochs = 100
            batchSize = 64
            print 'Training with a NN'
        elif algorithm == 'svm':
            gridSearch = [['scio', 'lumini'], ['spectrum'], [100, 200, 300, 400, 500], ['deriv1', 'preprocess1', 'log1', None], [0, 10], ['norm', 'scale'], [0, 1, 2, 3, 4]]
            gridSearch = [['scio'], ['spectrum'], [100, 200, 300, 400, 500], ['deriv1'], [0], ['scale'], [1]]
            print 'Training with an SVM'
        elif algorithm == 'rf':
            gridSearch = [['scio', 'lumini'], ['spectrum'], [100, 200, 300, 400, 500], ['deriv1', 'preprocess1', 'log1', None], [0, 10], ['norm', 'scale'], [None]]
            gridSearch = [['scio'], ['spectrum'], [100, 200, 300, 400, 500], ['deriv1'], [0], ['norm'], [None]]
            print 'Training with a random forest'

        bestScores = []
        bestParameterSets = []
        percents = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
        percents = [0.01]
        for percentSamplesPerMaterial in percents:
            bestScore = 0
            bestParameters = []
            for pca in gridSearch[0]:
                for ns in gridSearch[1]:
                    for kernel in gridSearch[2]:
                        print 'Parameters:', pca, ns, kernel
                        accuracies = []
                        skf = StratifiedKFold(n_splits=6, shuffle=True)
                        for trainIdx, testIdx in list(skf.split(X, y)):
                            if 'gan' in algorithm:
                                accuracies.append(learnGAN(X[trainIdx], y[trainIdx], X[testIdx], y[testIdx], percentLabeled=percentSamplesPerMaterial, epochs=epochs, noise_size=100, batchSize=batchSize, materialCount=6, PCA=pca, scale=ns, verbose=args.verbose, algorithm=algorithm, discTrainIters=1, genTrainIters=1))
                            else:
                                accuracies.append(learnNNSVM(X[trainIdx], y[trainIdx], X[testIdx], y[testIdx], percentLabeled=percentSamplesPerMaterial, epochs=epochs, batchSize=batchSize, materialCount=6, PCA=pca, scale=ns, verbose=args.verbose, kernel=kernel, algorithm=algorithm))
                            print 'Test accuracy:', accuracies[-1]
                            sys.stdout.flush()
                        avgAccuracy = np.mean(accuracies)
                        print 'Average accuracy:', avgAccuracy
                        sys.stdout.flush()
                        if avgAccuracy == bestScore:
                            # Found equally good parameters
                            bestParameters.append([pca, ns, kernel])
                        if avgAccuracy > bestScore:
                            # Found better parameters
                            bestScore = avgAccuracy
                            bestParameters = [[pca, ns, kernel]]
            bestScores.append(bestScore)
            bestParameterSets.append(bestParameters)
        for i, percentSamplesPerMaterial in enumerate(percents):
            print 'Percent labeled:', percentSamplesPerMaterial
            print 'Best score:', bestScores[i]
            print 'Best parameters:', bestParameterSets[i]
    elif '1' in args.test:
        # Generalizing to many new objects
        numSamples = 100
        epochs = 100
        batchSize = 64
        objects = [plastics, fabrics, papers, woods, metals]
        objects_train = [plastics[:5], fabrics[:5], papers[:5], woods[:5], metals[:5]]
        objects_test = [plastics[5:], fabrics[5:], papers[5:], woods[5:], metals[5:]]

        if algorithm == 'iwgan':
            gridSearch = [['scio', 'lumini'], ['spectrum'], [100, 200, 300, 400, 500], ['deriv1', 'deriv2', 'preprocess1', 'log1', None], [0], ['norm', 'scale'], [None]]
            gridSearch = [['scio'], ['spectrum'], [100, 200, 300, 400, 500], ['deriv1'], [0], ['norm'], [None]]
            numSamples = 100
            epochs = 100
            batchSize = 64
            print 'Training with a WGAN-GP / iWGAN'
        elif algorithm == 'iwganlstm':
            # biLSTM
            gridSearch = [['scio', 'lumini'], ['spectrum'], [100, 200, 300, 400, 500], ['deriv1', 'preprocess1'], [0], ['norm', 'scale'], [None]]
            gridSearch = [['scio'], ['spectrum'], [100, 200, 300, 400, 500], ['deriv1'], [0], ['scale'], [None]]
            numSamples = 100
            epochs = 200
            batchSize = 128
            print 'Training with a WGAN-GP / iWGAN biLSTM'
        elif algorithm == 'nn':
            gridSearch = [['scio', 'lumini'], ['spectrum'], [100, 200, 300, 400, 500], ['deriv1', 'deriv2', 'preprocess1', 'log1', None], [0, 10], ['norm', 'scale'], [None]]
            gridSearch = [['scio'], ['spectrum'], [100, 200, 300, 400, 500], ['deriv1'], [0], ['norm'], [None]]
            numSamples = 100
            epochs = 100
            batchSize = 64
            print 'Training with a NN'
        elif algorithm == 'lstm':
            # biLSTM
            gridSearch = [['scio'], ['spectrum'], [100, 200, 300, 400, 500], ['deriv1'], [0], ['scale'], [None]]
            numSamples = 100
            epochs = 100
            batchSize = 128
            print 'Training with a biLSTM'
        elif algorithm == 'svm':
            gridSearch = [['scio', 'lumini'], ['spectrum'], [100, 200, 300, 400, 500], ['deriv1', 'deriv2', 'preprocess1', 'log1', None], [0, 10], ['norm', 'scale'], [0, 1, 2, 3, 4]]
            gridSearch = [['scio'], ['spectrum'], [100, 200, 300, 400, 500], ['deriv1'], [0], ['scale'], [1]]
            numSamples = 100
            print 'Training with an SVM'
        elif algorithm == 'rf':
            gridSearch = [['scio', 'lumini'], ['spectrum'], [100, 200, 300, 400, 500], ['deriv1', 'deriv2', 'preprocess1', 'log1', None], [0, 10], ['norm', 'scale'], [None]]
            gridSearch = [['scio'], ['spectrum'], [100, 200, 300, 400, 500], ['deriv1'], [0], ['norm'], [None]]
            numSamples = 100
            print 'Training with a random forest'

        bestScores = []
        bestParameterSets = []
        percents = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
        numTrainObjects = [5, 2, 1]
        for nto in numTrainObjects:
            nfolds = len(plastics) / nto
            for percentSamplesPerMaterial in percents:
                bestScore = 0
                bestParameters = []
                for dataset in gridSearch[0]:
                    if dataset == 'lumini' and luminiData is None:
                        luminiData, wavelengthsLumini = util.loadLuminiDataset()
                    elif dataset == 'scio' and scioData is None:
                        scioData, wavelengthsScio = util.loadScioDataset()
                    for spectrumExposure in (gridSearch[1] if dataset == 'scio' else gridSearch[2]):
                        for dlp in gridSearch[3]:
                            for pca in gridSearch[4]:
                                for ns in gridSearch[5]:
                                    for kernel in gridSearch[6]:
                                        print 'Parameters:', dataset, spectrumExposure, dlp, pca, ns, kernel
                                        # Perform cross validation (k-fold) by selecting n/k objects for training and then the remaining objects for testing
                                        accuracies = []
                                        for n in xrange(nfolds):
                                            objects_train = [plastics[n*nto:(n+1)*nto], fabrics[n*nto:(n+1)*nto], papers[n*nto:(n+1)*nto], woods[n*nto:(n+1)*nto], metals[n*nto:(n+1)*nto]]
                                            objects_test = [[p for p in obj if p not in objects_train[i]] for i, obj in enumerate(objects)]
                                            if dataset == 'scio':
                                                Xtrain, ytrain = util.processScioDataset(scioData, materials, objects_train, sampleCount=numSamples, spectrumRaw=spectrumExposure)
                                                Xtest, ytest = util.processScioDataset(scioData, materials, objects_test, sampleCount=numSamples, spectrumRaw=spectrumExposure)
                                                Xtrain, ytrain, _ = preprocess(Xtrain, ytrain, wavelengthsScio, uvir=None, deriv_log_preprocess=dlp, doubleData=(spectrumExposure=='spectrum_raw'))
                                                Xtest, ytest, _ = preprocess(Xtest, ytest, wavelengthsScio, uvir=None, deriv_log_preprocess=dlp, doubleData=(spectrumExposure=='spectrum_raw'))
                                            elif dataset == 'lumini':
                                                Xtrain, ytrain = util.processLuminiDataset(luminiData, materials, objects_train, sampleCount=numSamples, exposure=spectrumExposure, correctedValues=True)
                                                Xtest, ytest = util.processLuminiDataset(luminiData, materials, objects_test, sampleCount=numSamples, exposure=spectrumExposure, correctedValues=True)
                                                Xtrain, ytrain, _ = preprocess(Xtrain, ytrain, wavelengthsLumini, uvir=None, deriv_log_preprocess=dlp, doubleData=False)
                                                Xtest, ytest, _ = preprocess(Xtest, ytest, wavelengthsLumini, uvir=None, deriv_log_preprocess=dlp, doubleData=False)
                                            if 'gan' in algorithm:
                                                accuracies.append(learnGAN(Xtrain, ytrain, Xtest, ytest, percentLabeled=percentSamplesPerMaterial, epochs=epochs, noise_size=100, batchSize=batchSize, materialCount=len(materials), PCA=pca, scale=ns, verbose=args.verbose, algorithm=algorithm, discTrainIters=1, genTrainIters=1))
                                            else:
                                                accuracies.append(learnNNSVM(Xtrain, ytrain, Xtest, ytest, percentLabeled=percentSamplesPerMaterial, epochs=epochs, batchSize=batchSize, materialCount=len(materials), PCA=pca, scale=ns, verbose=args.verbose, kernel=kernel, algorithm=algorithm))
                                            Xtrain = None
                                            ytrain = None
                                            Xtest = None
                                            ytest = None
                                            gc.collect()
                                            print 'Test accuracy:', accuracies[-1]
                                            sys.stdout.flush()
                                        avgAccuracy = np.mean(accuracies)
                                        print 'Average accuracy:', avgAccuracy
                                        sys.stdout.flush()

                                        if avgAccuracy == bestScore:
                                            # Found equally good parameters
                                            bestParameters.append([dataset, spectrumExposure, dlp, pca, ns, kernel])
                                        if avgAccuracy > bestScore:
                                            # Found better parameters
                                            bestScore = avgAccuracy
                                            bestParameters = [[dataset, spectrumExposure, dlp, pca, ns, kernel]]
                bestScores.append(bestScore)
                bestParameterSets.append(bestParameters)
        for i, nto in enumerate(numTrainObjects):
            for j, percentSamplesPerMaterial in enumerate(percents):
                print 'Number of Training Objects:', nto, 'Percent labeled:', percentSamplesPerMaterial
                print 'Best score:', bestScores[i*len(percents) + j]
                print 'Best parameters:', bestParameterSets[i*len(percents) + j]
    elif '2' in args.test:
        objects = [plastics, fabrics, papers, woods, metals]
        numSamples = 100
        epochs = 100
        batchSize = 64
        if algorithm == 'iwgan':
            gridSearch = [['scio', 'lumini'], ['spectrum'], [100, 200, 300, 400, 500], ['deriv1', 'deriv2', 'preprocess1', 'log1', None], [0], ['norm', 'scale'], [None]]
            gridSearch = [['scio'], ['spectrum'], [100, 200, 300, 400, 500], ['deriv1'], [0], ['norm'], [None]]
            numSamples = 100
            epochs = 100
            batchSize = 64
            print 'Training with a WGAN-GP / iWGAN'
        elif algorithm == 'nn':
            gridSearch = [['scio', 'lumini'], ['spectrum_raw', 'spectrum'], [100, 200, 300, 400, 500], ['deriv1', 'deriv2', 'preprocess1', 'preprocess2', 'log1', None], [0, 10], ['norm', 'scale'], [None]]
            gridSearch = [['scio'], ['spectrum'], [100, 200, 300, 400, 500], ['deriv1'], [0], ['norm'], [None]]
            numSamples = 100
            epochs = 100
            batchSize = 64
            print 'Training with a NN'
        elif algorithm == 'svm':
            gridSearch = [['scio', 'lumini'], ['spectrum_raw', 'spectrum'], [100, 200, 300, 400, 500], ['deriv1', 'deriv2', 'preprocess1', 'log1', None], [0, 10], ['norm', 'scale'], [0, 1, 2, 3, 4]]
            gridSearch = [['scio'], ['spectrum'], [100, 200, 300, 400, 500], ['deriv1'], [0], ['scale'], [1]]
            numSamples = 100
            print 'Training with an SVM'
        elif algorithm == 'rf':
            gridSearch = [['scio', 'lumini'], ['spectrum_raw', 'spectrum'], [100, 200, 300, 400, 500], ['deriv1', 'deriv2', 'preprocess1', 'log1', None], [0, 10], ['norm', 'scale'], [None]]
            gridSearch = [['scio'], ['spectrum'], [100, 200, 300, 400, 500], ['deriv1'], [0], ['norm'], [None]]
            numSamples = 100
            print 'Training with a random forest'

        percentSamplesPerMaterial = 0.01 # [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
        bestScore = 0
        bestParameters = []
        for dataset in gridSearch[0]:
            if dataset == 'lumini' and luminiData is None:
                luminiData, wavelengthsLumini = util.loadLuminiDataset()
            elif dataset == 'scio' and scioData is None:
                scioData, wavelengthsScio = util.loadScioDataset()
            for spectrumExposure in (gridSearch[1] if dataset == 'scio' else gridSearch[2]):
                for dlp in gridSearch[3]:
                    for pca in gridSearch[4]:
                        for ns in gridSearch[5]:
                            for kernel in gridSearch[6]:
                                print 'Parameters:', dataset, spectrumExposure, dlp, pca, ns, kernel
                                genCVAccuracies = []
                                for i, objectSet in enumerate(objects):
                                    for objName in objectSet:
                                        newSet = [x for x in objectSet if x != objName]
                                        objects_train = [x if i != j else newSet for j, x in enumerate(objects)]
                                        objects_test = [[]]*i + [[objName]] + [[]]*(len(materials) - 1 - i)
                                        if dataset == 'scio':
                                            Xtrain, ytrain = util.processScioDataset(scioData, materials, objects_train, sampleCount=numSamples, spectrumRaw=spectrumExposure)
                                            Xtest, ytest = util.processScioDataset(scioData, materials, objects_test, sampleCount=numSamples, spectrumRaw=spectrumExposure)
                                            Xtrain, ytrain, _ = preprocess(Xtrain, ytrain, wavelengthsScio, uvir=None, deriv_log_preprocess=dlp, doubleData=(spectrumExposure=='spectrum_raw'))
                                            Xtest, ytest, _ = preprocess(Xtest, ytest, wavelengthsScio, uvir=None, deriv_log_preprocess=dlp, doubleData=(spectrumExposure=='spectrum_raw'))
                                        elif dataset == 'lumini':
                                            Xtrain, ytrain = util.processLuminiDataset(luminiData, materials, objects_train, sampleCount=numSamples, exposure=spectrumExposure, correctedValues=True)
                                            Xtest, ytest = util.processLuminiDataset(luminiData, materials, objects_test, sampleCount=numSamples, exposure=spectrumExposure, correctedValues=True)
                                            Xtrain, ytrain, _ = preprocess(Xtrain, ytrain, wavelengthsLumini, uvir=None, deriv_log_preprocess=dlp, doubleData=False)
                                            Xtest, ytest, _ = preprocess(Xtest, ytest, wavelengthsLumini, uvir=None, deriv_log_preprocess=dlp, doubleData=False)
                                        if 'gan' in algorithm:
                                            accuracy = learnGAN(Xtrain, ytrain, Xtest, ytest, percentLabeled=percentSamplesPerMaterial, epochs=epochs, noise_size=100, batchSize=batchSize, materialCount=len(materials), PCA=pca, scale=ns, verbose=args.verbose, algorithm=algorithm, discTrainIters=1, genTrainIters=1)
                                        else:
                                            accuracy = learnNNSVM(Xtrain, ytrain, Xtest, ytest, percentLabeled=percentSamplesPerMaterial, epochs=epochs, batchSize=batchSize, materialCount=len(materials), PCA=pca, scale=ns, verbose=args.verbose, kernel=kernel, algorithm=algorithm)
                                        genCVAccuracies.append(accuracy)
                                        # print objects_test, accuracy
                                avgAccuracy = np.mean(genCVAccuracies)
                                print 'Average accuracy:', avgAccuracy
                                sys.stdout.flush()

                                if avgAccuracy == bestScore:
                                    # Found equally good parameters
                                    bestParameters.append([dataset, spectrumExposure, dlp, pca, ns, kernel])
                                if avgAccuracy > bestScore:
                                    # Found better parameters
                                    bestScore = avgAccuracy
                                    bestParameters.append([dataset, spectrumExposure, dlp, pca, ns, kernel])
        print 'Best score:', bestScore
        print 'Best parameters:', bestParameters

    # print 'Best score:', bestScore
    # print 'Best parameters:', bestParameters
    print 'Run time:', time.time() - t, 'seconds'
    sys.stdout.flush()

