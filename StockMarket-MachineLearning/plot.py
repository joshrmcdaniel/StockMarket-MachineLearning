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


def main():
    args = argparse.ArgumentParser()
    args.add_argument('-d', '--days-back', type=int, required=True)
    args.add_argument('-m', '--model', help='Path to model', type=str, required=True)
    
    args = args.parse_args()

    try:
        model = load_model(args.model)
    except OSError:
        raise FileNotFoundError('Cannot find model')

    if os.path.sep in args.model:
        company = args.model[args.model.rindex(os.path.sep)+1:args.model.rindex('.')]
    else:
        company = args.model[:args.model.rindex('.')]

    plot(model, company, args.days_back)

def plot(model, company, days_back):
    start = dt.datetime.now() - dt.timedelta(days_back)
    end = dt.datetime.now()
    data = web.DataReader(company, 'yahoo', start, end)
    actual_prices = data['Close'].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    model_inputs = data.values
    model_inputs = model_inputs.reshape(-1, 1)

    model_inputs = scaler.transform(model_inputs)

    x_test = []

    for x in range(days_back, len(model_inputs)):
        x_test.append(model_inputs[:x, 0])

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


if __name__ == '__main__':
    main()