# %% Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot
from pandas import datetime
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings('ignore')
plt.style.use('./plot.mplstyle')

# %% Import data

data = pd.read_csv('data/SPY.csv')
data['Date'] = pd.to_datetime(data['startEpochTime'], unit='s')
data.set_index('Date', inplace=True)

data = data[-10000:]

data.head(10)
data['closePrice'].plot()

# %%
# plt.figure()
# lag_plot(data['openPrice'], lag=1)
# plt.title('SPY Index - Autocorrelation plot with lag = 1')
# plt.show()

# %% Build the model
TRAINING_LENGTH = int(len(data) * 0.8)
train_data = data[0:TRAINING_LENGTH]
test_data = data[TRAINING_LENGTH:]
training_data = train_data['closePrice'].values
test_data = test_data['closePrice'].values
history = [x for x in training_data]
model_predictions = []
N_test_observations = len(test_data)

for time_point in range(N_test_observations):
    model = ARIMA(history, order=(1, 1, 10))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    model_predictions.append(yhat)
    true_test_value = test_data[time_point]
    history.append(true_test_value)

MSE_error = mean_squared_error(test_data, model_predictions)

print('Testing Mean Squared Error is {}'.format(MSE_error))

# %% Plot the results
test_set_range = data[int(len(data) * 0.7):].index

plt.plot(test_set_range, model_predictions, color='blue',
         marker='o', linestyle='dashed', label='Predicted Price')
plt.plot(test_set_range, test_data, color='red', label='Actual Price')
plt.title('TESLA Prices Prediction')
plt.xlabel('Date')
plt.ylabel('Prices')
# plt.xticks(np.arange(881, 1259, 50), data.Date[881:1259:50])
plt.legend()
plt.show()
