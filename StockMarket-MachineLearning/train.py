import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
import os

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM


start = dt.datetime(2012, 1, 1)
end = dt.datetime(2021, 1, 1)


def main():

    args = argparse.ArgumentParser()
    args.add_argument('-c', '--company', help='Company stocks to train on', required=True, type=str)
    args.add_argument('-d', '--days', help='Amount of previous days to train on', required=True, type=int)
    args.add_argument('-e', '--epochs', help='Amount of times to train', required=True, type=int)
    args.add_argument('-b', '--batch-size', help='Batch size', required=True, type=int)
   
    args = args.parse_args()

    print('NOTE: To change the start and end date of the stock, edit *start* and *end*')

    if os.path.exists(f'models/{args.company}.model'):
        if input(f'File {args.company}.model already exists, overwrite? ').lower() == 'y':
            train(args.company, args.days, args.batch_size, args.epochs)
        elif input(f'Compare prediction of {args.company} to actual data? ').lower() == 'y':
            model = load(args.company)
            compare_prediction_to_real(model, company)
        else:
            exit(0)
    else:
        train(args.company, args.days, args.batch_size, args.epochs)


def train(company, prediction_base, batch_size, epochs):
   
    data = web.DataReader(company, 'yahoo', start, end)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    x_train = []
    y_train = []

    for x in range(prediction_base, len(scaled_data)):
        x_train.append(scaled_data[x-prediction_base:x, 0])
        y_train.append(scaled_data[x, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = Sequential()

    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
    if not os.path.exists('models'):
        os.mkdir('models')
    model.save(f'models/{company}.model', overwrite=True)
    print('Model saved')
    if input('Compare predicted prices to real prices? ').lower == 'y':
        compare_prediction_to_real(model, company)
    else:
        exit(0)



def compare_prediction_to_real(model, company):
    prediction_base = 60
    data = web.DataReader(company, 'yahoo', start, end)
    test_start = dt.datetime(2020, 1, 1)
    test_end = dt.datetime.now()
    test_data = web.DataReader(company, 'yahoo', test_start, test_end)
    actual_prices = test_data['Close'].values

    total_dataset = pd.concat((data['Close'], test_data['Close']))
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_base:].values
    model_inputs = model_inputs.reshape(-1, 1)

    model_inputs = scaler.transform(model_inputs)

    x_test = []

    for x in range(prediction_base, len(model_inputs)):
        x_test.append(model_inputs[x-prediction_base:x, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predicted_prices = model.predict(x_test)

    predicted_prices = scaler.inverse_transform(predicted_prices)
    plt.plot(actual_prices, color='black', label=f'Actual {company} Price')
    plt.plot(predicted_prices, color='green', label =f'Predicted {company} Price')
    plt.title(f'{company} Share Price')
    plt.xlabel('Time')
    plt.ylabel(f'{company} Share Price')
    plt.legend()
    plt.show()


def load(company):
    model = load_model(f'models/{company}.model')
    return model

if __name__ == '__main__':
    main()