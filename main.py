from keras.models import Sequential
from keras.layers import LSTM, Dense, GRU
from keras.optimizers import Adam
from keras.utils import plot_model

import numpy as np
import input_data as dataset


def split_data(data, size, val, test):
    samples = data[0]
    labels = data[1]

    test_size = int(size * test)
    val_size = int(size * val)
    train_size = size - (test_size + val_size)

    x_train, y_train, x_val, y_val, x_test, y_test = samples[:train_size, ], labels[:train_size, ], samples[
                                                                                                    train_size:train_size + val_size, ], labels[
                                                                                                                                         train_size:train_size + val_size, ], samples[
                                                                                                                                                                              train_size + val_size:, ], labels[
                                                                                                                                                                                                         train_size + val_size:, ]
    return x_train, y_train, x_val, y_val, x_test, y_test


def reshape_data(data, sq_size):
    data_3dim = np.reshape(data, (data.shape[0], 1, data.shape[1]))
    shape0 = int(data.shape[0] / sq_size)
    shape1 = sq_size
    shape2 = data.shape[1]
    return np.resize(data_3dim, (shape0, shape1, shape2))


def compute_error_rate(predictions, truth):
    pred_around = np.around(predictions)
    errors = pred_around == truth
    unique, counts = np.unique(errors, return_counts=True)
    seqerr = dict(zip(unique, counts))
    f = False
    t = True
    total = truth.shape[0] * truth.shape[1]
    seq_err_rate = seqerr[f] / total
    acc = seqerr[t] / total
    return seq_err_rate, acc


if __name__ == "__main__":
    # input data
    nb = 50000
    timesteps = 1
    nb_samples = timesteps * nb
    val = 0.1
    test = 0.2
    data_dim = 102

    data = dataset.load_data(nb_samples)

    x_train, y_train, x_val, y_val, x_test, y_test = split_data(data, nb_samples, val, test)

    # first approach
    trainX = reshape_data(x_train, timesteps)
    trainY = reshape_data(y_train, timesteps)
    valX = reshape_data(x_val, timesteps)
    valY = reshape_data(y_val, timesteps)
    testX = reshape_data(x_test, timesteps)
    testY = reshape_data(y_test, timesteps)

    # model paramters
    results = []
    batch_size = 16
    nb_epochs = 1000
    learning_rate = 0.0001
    cell_nb = 128

    LSTMmodel = Sequential()
    LSTMmodel.add(LSTM(cell_nb,return_sequences=True,input_shape=(timesteps,data_dim)))
    LSTMmodel.add(LSTM(cell_nb, return_sequences=True))
    LSTMmodel.add(Dense(1,activation='sigmoid'))
    LSTMmodel.compile(loss='binary_crossentropy', optimizer=Adam(lr=learning_rate),metrics=['accuracy','mse','mae'])
    plot_model(model=LSTMmodel, to_file='LSTM.png', show_shapes=True, show_layer_names=True)
    LSTMmodel.fit(trainX, trainY, batch_size=batch_size, epochs=nb_epochs,validation_data=(valX,valY),verbose=1)
    predictions= LSTMmodel.predict(testX)
    err,acc = compute_error_rate(predictions,testY)
    score = LSTMmodel.evaluate(testX, testY, batch_size=batch_size,verbose=1)
    results.append({'model': 'LSTM','loss':score[0],'accuracy':score[1],'ser':1-score[1],'err':err,'acc':acc})

    GRUmodel = Sequential()
    GRUmodel.add(GRU(cell_nb, return_sequences=True, input_shape=(timesteps, data_dim)))
    GRUmodel.add(GRU(cell_nb, return_sequences=True))
    GRUmodel.add(Dense(1, activation='sigmoid'))
    GRUmodel.compile(loss='binary_crossentropy', optimizer=Adam(lr=learning_rate), metrics=['accuracy', 'mse', 'mae'])
    plot_model(model=model2, to_file='GRU_2.png', show_shapes=True, show_layer_names=True)
    GRUmodel.fit(trainX, trainY, batch_size=batch_size, epochs=nb_epochs, validation_data=(valX, valY), verbose=1)
    predictions = GRUmodel.predict(testX)
    err, acc = compute_error_rate(predictions, testY)
    score = GRUmodel.evaluate(testX, testY, batch_size=batch_size, verbose=1)
    results.append(
        {'model': 'GRU', 'loss': score[0], 'accuracy': score[1], 'ser': 1 - score[1], 'err': err, 'acc': acc})

    print(results)
