import os
import time
import warnings
import numpy as np
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Hide messy TensorFlow warnings
warnings.filterwarnings("ignore")  # Hide messy Numpy warnings


## This is a library of methods to build a reccurent neural net with long-short term memory layers.
## These methods are built to handle data of pairs (x,y) and used mainly to predict the next 60 positions of a Robot given a positions

def load_data(filename, seq_len, normalise_window):
    ## reads a txt file of (x,y) positions and build shifted sequences of seq_len
    ## Split the data into training and testing data
    ## return X_train, y_train, X_test, y_test to be fed to the model.
    data=[]
    with open(filename, 'r') as d:  # using the with keyword ensures files are closed properly
        for line in d.readlines():
            parts = line.split(',')  # change this to whatever the deliminator is
            data.append([int(parts[0]), int(parts[1].replace('\n',''))])

    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])

    dresult = np.array(result)
    if normalise_window:
        result = normalise_windows(result)
    result = np.array(result)

    row = int(round(0.95 * result.shape[0]))
    train = result[:row, :]
    np.random.shuffle(train)
    x_train = train[:, :-1]
    y_train = train[:, -1]

    x_test = result[row:, :-1]
    d_x_test = dresult[row:, :-1]
    y_test = result[row:, -1]
    d_y_test = dresult[row:, -1]


    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 2))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 2))
    d_x_test=np.reshape(d_x_test, (d_x_test.shape[0],d_x_test.shape[1], 2))

    prediction_len=60
    y_seq=[]
    d_y_seq=[]
    for i in range(int(len(y_test)/prediction_len)):
        curr_y=[]
        d_curr_y = []
        index=i * prediction_len
        for j in range(prediction_len):
            curr_y.append(y_test[index+j])
            d_curr_y.append(d_y_test[index + j])
        y_seq.append(curr_y)
        d_y_seq.append(d_curr_y)

    y_test=y_seq
    d_y_test = d_y_seq
    return [x_train, y_train, x_test, y_test, d_x_test, d_y_test ]

def get_last_sequence(filename, seq_len, normalise_window):
    ## read a txt file and return the last sequence of (x,y) positions
    data = []
    with open(filename, 'r') as d:  # using the with keyword ensures files are closed properly
        for line in d.readlines():
            parts = line.split(',')
            data.append([int(parts[0]), int(parts[1].replace('\n', ''))])

    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])

    dresult = np.array(result)
    if normalise_window:
        result = normalise_windows(result)
    result = np.array(result)

    return result[-1],dresult[-1]


def normalise_windows(window_data):
    ## normalising data to get convergence in training
    normalised_data = []
    for window in window_data:
        normalised_window = [[((float(p[0]) / float(window[0][0])) - 1),((float(p[1]) / float(window[0][1])) - 1)] for p in window]
        normalised_data.append(normalised_window)
    return normalised_data

def build_model(layers):
    ## We will build a recurrent neural network with long short term memory considering the positions as a time series
    ## We use Keras to build the KNN and LSTM layers
    model = Sequential()
    model.add(LSTM(
        input_dim=layers[0],
        output_dim=layers[1],
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        layers[2],
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
        output_dim=layers[3]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print("> Compilation Time : ", time.time() - start)
    return model

def predict_sequences_multiple(model, data, d_data, prediction_len,denormalise):
    ## Predict sequence of prediction_len steps before shifting prediction run forward by prediction_len steps
    ## d_xx stand for denormalized variables we needed to denormalise everything in order to predict the real positions
    prediction_seqs = []
    d_prediction_seqs=[]
    for i in range(int(len(data)/prediction_len)):
        curr_frame = data[i*prediction_len]
        d_curr_frame = d_data[i*prediction_len]

        predicted = []
        d_predicted=[]
        for j in range(prediction_len):
            prediction = model.predict(curr_frame[newaxis, :, :])
            d_prediction=np.array([[(prediction[0][0]+1)*d_curr_frame[0][0],(prediction[0][1]+1)*d_curr_frame[0][1]]])
            predicted.append(prediction)
            d_predicted.append(d_prediction)
            curr_frame = curr_frame[1:]
            d_curr_frame = d_curr_frame[1:]
            curr_frame = np.append(curr_frame, predicted[-1], axis=0)
            d_curr_frame = np.append(d_curr_frame, d_predicted[-1], axis=0)
        prediction_seqs.append(predicted)
        d_prediction_seqs.append(d_predicted)
    if (denormalise):
        return d_prediction_seqs
    return prediction_seqs

def predict_sequence(model, data, d_data, prediction_len,denormalise):
    ## Predict one sequence of prediction_len given data
    predicted = []
    d_predicted = []
    for j in range(prediction_len):
        prediction = model.predict(data[newaxis, :, :])
        d_prediction = np.array([[(prediction[0][0] + 1) * d_data[0][0], (prediction[0][1] + 1) * d_data[0][1]]])
        predicted.append(prediction)
        d_predicted.append(d_prediction)
        data = data[1:]
        d_data = d_data[1:]
        data = np.append(data, predicted[-1], axis=0)
        d_data = np.append(d_data, d_predicted[-1], axis=0)
    if(denormalise):
        return d_predicted
    return predicted
