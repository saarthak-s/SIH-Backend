# import pandas as pd
# from pmdarima import auto_arima
# from sklearn.metrics import mean_absolute_percentage_error
#
#
# #
# # model_fit = joblib.load('model.joblib')  # Forecast future values
# # print(model_fit.summary())
# # forecast = model_fit.get_forecast(steps=1)
# # print(forecast.predicted_mean)
#
#
# df = pd.read_excel("/Users/vigyatgoel/Desktop/SIH2024/backend/sih_backend/src/ArimaModel/processed.xlsx")
# df['Datetime'] = pd.to_datetime(df['Datetime'], format="%d/%m/%Y %H:%M")
# # print(df.dtypes)
# df.set_index(df['Datetime'], inplace=True)
# df = df.asfreq('15min')  # Set frequency explicitly
# df.index.freq = '15min'  # Ensure frequency is set correctly
# df.sort_index()
#
# train_start = df.index[0] + pd.Timedelta(days=2587)
# train_end = train_start + pd.Timedelta(days=29) + pd.Timedelta(hours=7)
# train = df.loc[train_start:train_end]
#
# test_start = train_end + pd.Timedelta(minutes=15)
# test_end = train_start + pd.Timedelta(days=31) - pd.Timedelta(minutes=15)
# test = df.loc[test_start:test_end]
#
# print(train)
# print(test)
#
# auto_model = auto_arima(train['Load'], seasonal=True, m=96, suppress_warnings=True, n_jobs=-1, stepwise=False)
#
#
# print(auto_model.summary())
#
# # Number of periods to predict into the future
# n_periods = len(test)  # For example, predicting 12 months ahead
#
# # Make future predictions
# future_predictions = auto_model.predict(n_periods=n_periods)
#
# # Convert to a pandas DataFrame for better visualization
# future_df = pd.DataFrame({
#     'Forecast': future_predictions
# })
#
# mape = mean_absolute_percentage_error(test['Load'], future_df['Forecast'])
# print('mape :', mape * 100)
#
# # Print the predictions
# # print(future_df)


import time
import joblib
import pandas as pd
from matplotlib import pyplot as plt
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_percentage_error

# Load the data
df = pd.read_excel("/Users/vigyatgoel/Desktop/SIH2024/backend/sih_backend/src/ArimaModel/processed.xlsx")
df['Datetime'] = pd.to_datetime(df['Datetime'], format="%d/%m/%Y %H:%M")
df.set_index(df['Datetime'], inplace=True)
df = df.asfreq('15min')  # Set frequency explicitly
df.index.freq = '15min'  # Ensure frequency is set correctly
df.sort_index()

for i in range(1):
    train_start = df.index[0] + pd.Timedelta(days=2587 + i)
    train_end = train_start + pd.Timedelta(days=29) + pd.Timedelta(hours=7)
    train = df.loc[train_start:train_end]

    test_start = train_end + pd.Timedelta(minutes=15)
    test_end = train_start + pd.Timedelta(days=31) - pd.Timedelta(minutes=15)
    test = df.loc[test_start:test_end]

    print(train)
    print(test)

    # Auto-ARIMA model fitting
    model = auto_arima(train['Load'],
                       seasonal=True,       # Enable seasonal ARIMA
                       m=96,                # Seasonal period (for 15-min intervals in 24 hours, 96 periods)
                       stepwise=True,       # Stepwise search for faster computation
                       suppress_warnings=True)

    # Save model
    # joblib.dump(model, 'auto_arima_model.joblib')

    # Forecast future values
    forecast_periods = len(test)
    forecast = model.predict(n_periods=forecast_periods)

    # Create forecast DataFrame
    forecast_df = pd.DataFrame({'Forecast': forecast}, index=test.index[:forecast_periods])

    # Adjust the forecast period to match the testing period
    forecast_df = forecast_df.loc[train_end + pd.Timedelta(hours=24 - 7):test_end]
    test = test.loc[train_end + pd.Timedelta(hours=24 - 7):test_end]

    print(forecast_df)
    print(test)

    # Plot the forecast and actual data
    plt.figure(figsize=(12, 6))
    plt.plot(test.index, forecast_df['Forecast'], label='Forecast')
    plt.plot(test.index, test['Load'], label='Actual')
    plt.xlabel("Date")
    plt.ylabel("Load")
    plt.legend()
    plt.show()

    # Calculate MAPE (Mean Absolute Percentage Error)
    mape = mean_absolute_percentage_error(test['Load'], forecast_df['Forecast'])
    print('MAPE:', mape * 100)

    # Wait before the next iteration
    # time.sleep(5)
