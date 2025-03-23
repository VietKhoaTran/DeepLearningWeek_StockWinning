import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

class LSTMModel:
    def __init__(self, ticker, start_date, end_date, prev_days=60, prediction_days=1):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.prev_days = prev_days
        self.prediction_days = prediction_days
        self.scaler = MinMaxScaler()
        self.model = None
        self.df = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

    def add_indicators(self):
        df = self.df

        #Adding SMAs into df
        df['SMA50'] = df['Close'].rolling(window=50).mean()
        df['SMA200'] = df['Close'].rolling(window=200).mean()

        #Adding EMAs for MACD
        df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()

        #Adding MACD
        df['MACD'] = df['EMA12'] - df['EMA26']

        #Adding VMA
        df['VMA'] = df['Volume'].rolling(window=20).mean()

        #Adding Average True Range(ATR)
        high = df['High']
        low = df['Low']
        df['diff'] = high - low
        df['v1'] = abs(high - df['Close'].shift())
        df['v2'] = abs(low - df['Close'].shift())
        df['TR'] = df[['diff', 'v1', 'v2']].max(axis=1)
        df['ATR'] = df['TR'].rolling(window=14).mean()

    def preprocess(self):
        if self.df is None:
            print("No df available")
            return

        self.add_indicators()

        vars = ['Close', 'SMA50', 'SMA200', 'EMA12', 'EMA26', 'MACD', 'VMA', 'ATR']
        self.df[vars] = self.df[vars].bfill().ffill()
        scaled_df = self.scaler.fit_transform(self.df[vars])

        X, y = [], []
        for i in range(len(scaled_df) - self.prev_days - self.prediction_days):
            prev_seq = scaled_df[i:i+self.prev_days]
            X.append(prev_seq)
            target = scaled_df[i+self.prev_days+self.prediction_days]
            y.append(target[0])

        X, y = np.array(X), np.array(y)
        split_index = int(0.8 * len(X))
        self.X_train, self.X_test = X[:split_index], X[split_index:]
        self.y_train, self.y_test = y[:split_index], y[split_index:]

    def get_df(self):
        try:
            df = yf.download(self.ticker, start=self.start_date, end=self.end_date, interval="1d")
            if df.empty:
                print(f"No data found for {self.ticker}")
                return
            self.df = df[['Close', 'High', 'Low', 'Volume']]
        except:
            print("Error getting df")

    def build_model(self):
        vars = ['Close', 'SMA50', 'SMA200', 'EMA12', 'EMA26', 'MACD', 'VMA', 'ATR']
        self.model = Sequential()
        self.model.add(LSTM(units=50,
                            return_sequences=True,
                            input_shape=(self.prev_days, len(vars))))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=50, return_sequences=False))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(units=1))
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def train_model(self, epochs=50, batch_size=32):

        if self.model is None:
            print("No model yet")
            return

        '''
        Args:
            epochs: Number of iterations
            batch_size: Number of samples for one gradient update
        '''

        res = self.model.fit(
            self.X_train,
            self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.X_test, self.y_test),
            verbose=1
        )

        return res

    def predict(self):
        """Make predictions using the trained LSTM model."""
        if self.model is None:
            return
        
        # Make predictions
        predictions = self.model.predict(self.X_test)

        if predictions is None or len(predictions) == 0:
            return None

        predictions = self.scaler.inverse_transform(
            np.concatenate([predictions, np.zeros((len(predictions), 7))], axis=1)
        )[:, 0]
        return predictions

    def predict_future(self, days=30):
        if self.model is None:
            return None

        last_seq = self.X_test[-1]

        predictions = []
        for _ in range(days):
            pred = self.model.predict(last_seq.reshape(1, self.prev_days, last_seq.shape[1]))
            predictions.append(pred[0, 0])

            new_input = np.append(last_seq[1:], [[pred[0, 0]] + [0] * (last_seq.shape[1] - 1)], axis=0)

            last_seq = new_input

        predictions = self.scaler.inverse_transform(
            np.concatenate([np.array(predictions).reshape(-1, 1), np.zeros((days, 7))], axis=1)
        )[:, 0]

        return predictions

    def plot_predictions(self):
        '''
        Drawing graph of prediction vs actual price
        '''
        predictions = self.predict()
        if predictions is None or len(predictions) == 0:
            return

        actual_prices = self.df['Close'].iloc[-len(predictions):].values

        if len(predictions) != len(actual_prices):
            return

        plt.figure(figsize=(12, 6))
        plt.plot(self.df.index[-len(predictions):], actual_prices, label="Actual Price")
        plt.plot(self.df.index[-len(predictions):], predictions, label="Predicted Price", linestyle='dashed')
        plt.legend()
        plt.title(f"{self.ticker} Stock Price Prediction")
        plt.show()

    def plot_future_predictions(self, days=30):
        predictions = self.predict_future(days=days)

        if predictions is None:
            return

        last_date = self.df.index[-1]
        future_dates = pd.date_range(last_date, periods=days + 1)[1:]

        plt.figure(figsize=(12, 6))
        plt.plot(self.df.index, self.df["Close"], label="Actual Price", color='blue')
        plt.plot(future_dates, predictions, label=f"Predicted {days} Days", linestyle="dashed", color='red')
        plt.legend()
        plt.title(f"{self.ticker} Stock Price Prediction for Next {days} Days")
        plt.show()

    def pred_next_day(self):

        if self.model is None:
            return None

        last_seq = self.X_test[-1].reshape(1, self.prev_days, self.X_test.shape[2])
        pred_price = self.model.predict(last_seq)[0, 0]
        next_day_price = self.scaler.inverse_transform(
            np.concatenate([[pred_price], np.zeros(7)]).reshape(1, -1)
        )[0, 0]

        return next_day_price

    def plot_next_day_pred(self):
        pred_next_day = self.pred_next_day()

        if pred_next_day is None:
            return

        plt.figure(figsize=(12, 6))
        plt.plot(self.df.index, self.df['Close'], label="Actual Price", color='blue')
        last_date = self.df.index[-1]
        next_day_date = last_date + pd.Timedelta(days=1)
        plt.scatter(next_day_date, pred_next_day, color='red',
                    label=f"Predicted Next Day: {pred_next_day:.2f}", zorder=5)

        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title(f"{self.ticker} - Actual vs Predicted Next Day Price")
        plt.legend()
        plt.show()