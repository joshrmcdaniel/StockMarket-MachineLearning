import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
import os
import argparse


from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM


def main():
    args = argparse.ArgumentParser()
    args.add_argument('-c', '--company', help='Company to predict on (must have trained model for specific company!)', required=True)
    args.add_argument('-m', '--model', help='Directory containing model (default: models/)', required=False, default='models')
    args = args.parse_args()

    start = dt.datetime(2012, 1, 1)
    end = dt.datetime(2021, 5, 1)
    data = web.DataReader(args.company, 'yahoo', start, end)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    prediction_base = 60

    try:
        model = load_model(args.model+os.path.sep+company+'.model')
    except FileNotFoundError:
        print('Model not found')
        exit(1)

    test_start = dt.datetime(2020, 1, 1)
    test_end = dt.datetime.now()

    test_data = web.DataReader(company, 'yahoo', test_start, test_end)

    total_dataset = pd.concat((data['Close'], test_data['Close']))

    model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_base:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)


    real_data = [model_inputs[len(model_inputs) + 1 - prediction_base:len(model_inputs) + 1, 0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

    prediction = model.predict(real_data)
    prediction = scaler.inverse_transform(prediction)
    print(f'Prediction for tomorrow: ${prediction[0][0]:.2f}')


if __name__ == '__main__':
    main()