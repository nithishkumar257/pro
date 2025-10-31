import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Load the data
data = pd.read_csv('pharma_sales.csv', parse_dates=['Date'])

# Ensure the Date column is datetime
data['Date'] = pd.to_datetime(data['Date'])

# List of products
products = data['Product'].unique()

# Prepare DataFrame to store forecasts
forecast_results = pd.DataFrame()

# Loop through each product
for product in products:
    print(f'Processing product: {product}')
    
    # Filter data for the current product
    product_data = data[data['Product'] == product]
    product_data = product_data.set_index('Date').sort_index()
    
    # Check data length
    if len(product_data) < 24:
        print(f'Not enough data for product {product}. Skipping...')
        continue
    
    # Train-Test Split
    split_point = int(len(product_data) * 0.8)
    train = product_data.iloc[:split_point]
    test = product_data.iloc[split_point:]
    
    # Fit SARIMA Model
    try:
        model = SARIMAX(train['Sales'],
                        order=(1, 1, 1),
                        seasonal_order=(1, 1, 1, 12),
                        enforce_stationarity=False,
                        enforce_invertibility=False)
        model_fit = model.fit(disp=False)
    except Exception as e:
        print(f'Error fitting model for {product}: {e}')
        continue
    
    # Forecast Test Data
    test_forecast = model_fit.get_forecast(steps=len(test))
    test_pred = test_forecast.predicted_mean
    test_conf_int = test_forecast.conf_int()
    
    # Compute RMSE
    rmse = np.sqrt(mean_squared_error(test['Sales'], test_pred))
    print(f'RMSE for {product}: {rmse:.2f}')
    
    # Forecast Next 12 Months
    forecast_steps = 12
    forecast = model_fit.get_forecast(steps=forecast_steps)
    forecast_index = pd.date_range(start=product_data.index[-1] + pd.DateOffset(months=1), periods=forecast_steps, freq='MS')
    forecast_values = forecast.predicted_mean
    forecast_conf_int = forecast.conf_int()
    
    # Plot Forecast
    plt.figure(figsize=(12, 6))
    plt.plot(product_data['Sales'], label='Historical Sales')
    plt.plot(test.index, test_pred, label='Test Forecast')
    plt.plot(forecast_index, forecast_values, label='Future Forecast', linestyle='--')
    plt.fill_between(forecast_index,
                     forecast_conf_int.iloc[:, 0],
                     forecast_conf_int.iloc[:, 1], color='k', alpha=0.1)

    # Add logic for seasonal variations and market trends
    plt.axvline(x=test.index[0], color='gray', linestyle='--', label='Train-Test Split')
    peak_seasons = ['01', '07']  # Example: January and July are peak seasons
    for season in peak_seasons:
        plt.axvline(x=forecast_index[forecast_index.month == int(season)][0], color='red', linestyle='--', label='Peak Season')

    # Mark special events or promotions (example: a known sales peak event)
    special_event_date = pd.to_datetime('2025-06-01')
    plt.axvline(x=special_event_date, color='blue', linestyle='--', label='Special Event')

    plt.title(f'Sales Forecast for {product}')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.show()
    
    # Store Forecast
    forecast_df = pd.DataFrame({
        'Product': product,
        'Date': forecast_index,
        'Forecasted_Sales': forecast_values,
        'RMSE': rmse  # Add RMSE for each product
    })
    forecast_results = pd.concat([forecast_results, forecast_df], ignore_index=True)

# Save Forecast Results
forecast_results.to_csv('forecasted_sales_all_products.csv', index=False)
print("Forecasting completed. Results saved to 'forecasted_sales_all_products.csv'")
