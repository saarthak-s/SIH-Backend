import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

# weather_df = pd.read_excel(r'C:\Users\Saarthak\Desktop\datasets\Delhi 2017-2024 Load\weather may-june2024.xlsx')
# weather_df['Datetime'] = pd.to_datetime(weather_df['Datetime'], format="%d/%m/%Y %H:%M")
# weather_df.set_index(weather_df['Datetime'], inplace=True)
#
# # Assuming 'weather_df' is your weather DataFrame
# weather_resampled = weather_df.resample('15min').ffill()  # Resample to 15 minutes
# weather_resampled = weather_resampled.asfreq('15min')  # Set frequency explicitly
# weather_resampled.index.freq = '15min'
# weather_df.sort_index()
# # weather_resampled = weather_resampled.loc['2024-06-01 00:00:00':'2024-06-26 23:45:00']
# last_row = weather_resampled.iloc[-1]̐̐̐
# new_timestamps = ['2024-06-26 23:15:00', '2024-06-26 23:30:00', '2024-06-26 23:45:00']
# new_rows = pd.DataFrame([last_row.values] * len(new_timestamps), columns=weather_resampled.columns,
#                         index=new_timestamps)
#
# # Concatenate the new rows with the existing DataFrame
# weather_resampled = pd.concat([weather_resampled, new_rows])̐
# weather_resampled.to_excel(r'C:\Users\Saarthak\Desktop\datasets\Delhi 2017-2024 Load\weather.xlsx')
# print(weather_resampled.to_string())

# Loading Data into Dataframe
# df = pd.read_csv("/Users/vigyatgoel/Desktop/SIH2024/backend/sih_backend/src/ArimaModel/processed.csv")
# df['Datetime'] = pd.to_datetime(df['Datetime'], format="%d/%m/%Y %H:%M")
# # print(df.dtypes)
# df.set_index(df['Datetime'], inplace=True)
# df = df.asfreq('15min')  # Set frequency explicitly
# df.index.freq = '15min'  # Ensure frequency is set correctly
# df.sort_index()

import pandas as pd

weather_df = pd.read_excel(r'C:\Users\Saarthak\Desktop\datasets\Delhi 2017-2024 Load\tempampril-may.xlsx')
weather_df['Datetime'] = pd.to_datetime(weather_df['Datetime'], format="%d/%m/%Y %H:%M")
weather_df.set_index(weather_df['Datetime'], inplace=True)

# Assuming 'weather_df' is your weather DataFrame
weather_resampled = weather_df.resample('15min').ffill()  # Resample to 15 minutes
weather_resampled = weather_resampled.asfreq('15min')  # Set frequency explicitly
weather_resampled.index.freq = '15min'
weather_df.sort_index()
# weather_resampled = weather_resampled.loc['2024-06-01 00:00:00':'2024-06-26 23:45:00']
# last_row = weather_resampled.iloc[-1]
# new_timestamps = ['2024-06-26 23:15:00', '2024-06-26 23:30:00', '2024-06-26 23:45:00']
# new_rows = pd.DataFrame([last_row.values] * len(new_timestamps), columns=weather_resampled.columns,
#                         index=new_timestamps)

# Concatenate the new rows with the existing DataFrame
# weather_resampled = pd.concat([weather_resampled, new_rows])
# weather_resampled.to_excel(r'C:\Users\Saarthak\Desktop\datasets\Delhi 2017-2024 Load\weather.xlsx')
print(weather_resampled.to_string())

# Read the CSV file
df = pd.read_csv(r"C:\Users\Saarthak\Desktop\datasets\Delhi 2017-2024 Load\processed.csv")

# Correct the format of the 'Datetime' column
df['Datetime'] = pd.to_datetime(df['Datetime'], format="%Y-%m-%d %H:%M:%S")

# Set the 'Datetime' column as the index
df.set_index(df['Datetime'], inplace=True)

# Resample data to 15-minute frequency
df = df.asfreq('15min')  # Set frequency explicitly

# Ensure frequency is set correctly
df.index.freq = '15min'

# Sort the index (optional, if not already sorted)
df.sort_index(inplace=True)

# Check the result
# print(df.head())
# forecast_start = pd.to_datetime('2024-06-1 00:00:00')

# print(df)
for i in range(4):
    train_start = df.index[0] + pd.Timedelta(days=2557 + i)
    train_end = train_start + pd.Timedelta(days=29) + pd.Timedelta(hours=7)
    train = df.loc[train_start:train_end]

    test_start = train_end + pd.Timedelta(minutes=15)
    test_end = train_start + pd.Timedelta(days=31) - pd.Timedelta(minutes=15)
    test = df.loc[test_start:test_end]
    train['wind'] = weather_resampled['temp']
    test['wind'] = weather_resampled['temp']
    test['wind'] = test['wind'].ffill()
    print(train)
    print(test)

    model = SARIMAX(endog=train['Load'],
                    exog=train['wind'],
                    order=(2, 0, 0),
                    seasonal_order=(1, 1, 0, 96))

    # Fit the model
    model_fit = model.fit(low_memory=True)
    # Print the model summary
    # model_fit = joblib.load('model.joblib')  # Forecast future values
    # print(model_fit.summary())
    # time.sleep(5)
    forecast_periods = len(test)
    # forecast = model_fit.get_forecast(steps=forecast_periods, exog=test['wind'])
    forecast = model_fit.get_forecast(exog=test['wind'], steps=forecast_periods)
    forecast_mean = forecast.predicted_mean
    # forecast_ci = forecast.conf_int()
    # print(forecast_mean)
    forecast_df = pd.DataFrame({'Forecast': forecast_mean})
    # print(forecast_df)
    # forecast_df.to_excel(r'C:\Users\Saarthak\Desktop\datasets\Delhi 2017-2024 Load\forecast.xlsx')

    # joblib.dump(model_fit, 'model.joblib')
    forecast_df = pd.DataFrame({'Forecast': forecast_mean})

    forecast_df = forecast_df.loc[train_end + pd.Timedelta(hours=24 - 7):test_end]
    test = test.loc[train_end + pd.Timedelta(hours=24 - 7):test_end]

    print(forecast_df)
    print(test)
    # Plot the forecast
    plt.figure(figsize=(12, 6))
    plt.plot(test.index, forecast_df['Forecast'], label='forecast')
    plt.plot(test.index, test['Load'], label='test')
    plt.xlabel("Date")
    plt.ylabel("Load")
    plt.legend()
    plt.show()
    mape = mean_absolute_percentage_error(test['Load'], forecast_df['Forecast'])
    print('mape :', mape * 100)

    # forecast_start = forecast_start + pd.Timedelta(days=1)
    # time.sleep(5)
