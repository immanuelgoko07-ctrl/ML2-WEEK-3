##  1. Environment Setup

Install required packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow statsmodels openml
```

---

##  2. Load the Dataset from OpenML

```python
import openml
import pandas as pd

# Download dataset (ID=31 airquality)
dataset = openml.datasets.get_dataset(31)
df, *_ = dataset.get_data(dataset_format="dataframe")

# Inspect
print(df.head())
print(df.info())
```

---

##  3. Data Preprocessing

### Handling Missing Values

```python
# Check missing values
print(df.isna().sum())

# Fill missing continuous values with median
df = df.fillna(df.median())
```

---

##  4. Feature Engineering

###  Create a Datetime Column

```python
df['Date'] = pd.to_datetime(dict(year=1973, month=df['Month'], day=df['Day']))
df = df.set_index('Date').sort_index()
```

###  Lag Features & Rolling Averages

```python
# Lag features
for lag in [1,3,7]:
    df[f'Ozone_lag{lag}'] = df['Ozone'].shift(lag)

# Rolling means
df['Ozone_ma7'] = df['Ozone'].rolling(window=7).mean()
```

### Time Indicators

```python
df['Month'] = df.index.month
df['DayOfYear'] = df.index.dayofyear
```

###  Drop Rows After Feature Engineering

```python
df = df.dropna()
```

---

##  5. Train–Test Split

```python
train_size = int(len(df)*0.8)
train = df[:train_size]
test = df[train_size:]
```

---

##  6. Traditional Time Series Model – ARIMA

```python
import statsmodels.api as sm

# Fit ARIMA on ozone
model = sm.tsa.ARIMA(train['Ozone'], order=(2,1,2), seasonal_order=(1,1,1,12))
arima_fit = model.fit()
print(arima_fit.summary())

# Forecast
forecast = arima_fit.forecast(steps=len(test))
```

---

## 7. Neural Network Model – LSTM (Deep Learning)

###  Prepare Data for LSTM

```python
import numpy as np

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[['Ozone']])

X, y = [], []
for i in range(7, len(scaled)):
    X.append(scaled[i-7:i])
    y.append(scaled[i])

X, y = np.array(X), np.array(y)

# Train-test
X_train, X_test = X[:train_size-7], X[train_size-7:]
y_train, y_test = y[:train_size-7], y[train_size-7:]
```

###  Build LSTM

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential([
    LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(1)
])

model.compile(optimizer='adam', loss='mae')
history = model.fit(X_train, y_train, epochs=30, batch_size=16, validation_split=0.1)
```

###  Predict

```python
y_pred_scaled = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred_scaled)
```

---

##  8. Evaluation

### Metrics

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error

# For ARIMA
rmse_arima = mean_squared_error(test['Ozone'], forecast, squared=False)
mae_arima = mean_absolute_error(test['Ozone'], forecast)

# For LSTM
rmse_lstm = mean_squared_error(test['Ozone'].values, y_pred.flatten(), squared=False)
mae_lstm = mean_absolute_error(test['Ozone'].values, y_pred.flatten())

print(f"ARIMA RMSE={rmse_arima}, MAE={mae_arima}")
print(f"LSTM RMSE={rmse_lstm}, MAE={mae_lstm}")
```

---

##  9. Visualization

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(14,6))
plt.plot(test.index, test['Ozone'], label='Actual Ozone')
plt.plot(test.index, forecast, label='ARIMA Forecast')
plt.plot(test.index, y_pred, label='LSTM Prediction')
plt.legend()
plt.title("Ozone Air Quality Forecast")
plt.xlabel("Date")
plt.ylabel("Ozone Levels")
plt.show()
```

---

##  10. Interpretation of Results

| Model     | Strengths                                          | Weaknesses                        |
| --------- | -------------------------------------------------- | --------------------------------- |
| **ARIMA** | Captures linear trends & seasonality well, simpler | Struggles with nonlinear patterns |
| **LSTM**  | Learns complex temporal dynamics                   | Requires more data & tuning       |

---

##  Summary of Deliverables

 Data cleaning: missing value handling
 
 Temporal indexing & aggregation (daily)
 
 Feature engineering: lags, rolling means, time indicators
 
 Train–test split
 
 Two different models: ARIMA & LSTM
 
 Evaluation metrics: RMSE & MAE
 
 Visualization & interpretation

