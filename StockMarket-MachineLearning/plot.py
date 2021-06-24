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
    args.add_argument('-d', '--days-back', type=int, default=90)
    args.add_argument('-m', '--model', help='Model to use', type=str, required=True)
    
    args = args.parse_args()

    graph = plot_predictions(args.model, args.days_back)
    graph.legend()
    graph.show()


def plot_predictions(company, days_back):
    try:
        model = load_model(f'models/{company}.model')
    except OSError:
        raise FileNotFoundError('Cannot find model')
    start = dt.datetime(2012, 1, 1)
    end = dt.datetime(2021, 1, 1) 
    data = web.DataReader(company, 'yahoo', start, end)
    test_start = dt.datetime(2020, 1, 1)
    test_end = dt.datetime.now()
    test_data = web.DataReader(company, 'yahoo', test_start, test_end)
    actual_prices = test_data['Close'].values

    total_dataset = pd.concat((data['Close'], test_data['Close']))
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    model_inputs = total_dataset[len(total_dataset) - len(test_data) - days_back:].values
    model_inputs = model_inputs.reshape(-1, 1)

    model_inputs = scaler.transform(model_inputs)

    x_test = []

    for x in range(days_back, len(model_inputs)):
        x_test.append(model_inputs[x-days_back:x, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predicted_prices = model.predict(x_test)

    predicted_prices = scaler.inverse_transform(predicted_prices)
    plt.plot(actual_prices, color='black', label=f'Actual {company} Price')
    plt.plot(predicted_prices, color='green', label =f'Predicted {company} Price')
    plt.title(f'{company} Share Price')
    plt.xlabel('Time')
    plt.ylabel(f'{company} Share Price')
    return plt


if __name__ == '__main__':
    main()