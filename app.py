import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
import streamlit as st

# -----------------------------
# 1. Load and Preprocess Data
# -----------------------------

# Load datasets 
transactions_01 = pd.read_csv('data/Transactional_data_retail_01.csv')
transactions_02 = pd.read_csv('data/Transactional_data_retail_02.csv')

# Combine the transactions data
transactions = pd.concat([transactions_01, transactions_02])

# Filter out negative quantities (returns or errors)
transactions = transactions[transactions['Quantity'] >= 0].copy()

# Create a Revenue column
transactions['Revenue'] = transactions['Quantity'] * transactions['Price']

# -----------------------------------------
# 2. Identify Top 10 Stock Codes by Quantity
# -----------------------------------------

# Group by StockCode to get total quantities sold
top_products_by_quantity = transactions.groupby('StockCode')['Quantity'].sum().nlargest(10).index.tolist()

# --------------------------------
# 3. Streamlit App for Forecasting
# --------------------------------

# App title
st.title("Demand Forecasting for Top Products")

# User input for stock code and number of weeks to forecast
stock_code = st.selectbox('Select Stock Code:', top_products_by_quantity)
weeks_to_forecast = st.slider('Weeks to Forecast:', min_value=1, max_value=15, value=15)

# Filter data for selected stock code
product_data = transactions[transactions['StockCode'] == stock_code].copy()

st.write("Filtered Product Data:")
st.write(product_data)

# Convert 'InvoiceDate' to datetime
product_data['InvoiceDate'] = pd.to_datetime(product_data['InvoiceDate'], errors='coerce', dayfirst=True)

# Drop rows where date conversion failed
product_data = product_data.dropna(subset=['InvoiceDate'])

# Set 'InvoiceDate' as the index for resampling
product_data.set_index('InvoiceDate', inplace=True)

# Resample data by week
weekly_sales = product_data.resample('W').sum()['Quantity']

st.write("Weekly Sales Data:")
st.write(weekly_sales)

if weekly_sales.empty:
    st.error("Weekly sales data is empty. Ensure that the data contains valid dates and quantities.")
else:
    st.write(f"Number of data points for forecasting: {len(weekly_sales)}")

    # Check if there are enough data points
    if len(weekly_sales) < 3:
        st.warning("Not enough data points for forecasting.")
    else:
        try:
            st.write("Fitting Prophet model...")

            # Prepare data for Prophet
            df_prophet = weekly_sales.reset_index().rename(columns={'InvoiceDate': 'ds', 'Quantity': 'y'})

            # Create and fit the model
            model = Prophet()
            model.fit(df_prophet)

            # Make a future dataframe for the forecast
            future = model.make_future_dataframe(periods=weeks_to_forecast, freq='W')

            # Forecast
            forecast = model.predict(future)

            # Extract forecasted values
            forecast_values = forecast[['ds', 'yhat']].set_index('ds').iloc[-weeks_to_forecast:]

            st.write("Forecast Values:")
            st.write(forecast_values)

            # Plot historical and forecasted data
            st.write(f"Historical and Forecasted Sales for Stock Code {stock_code}")
            plt.figure(figsize=(10, 6))
            plt.plot(weekly_sales, label="Historical Sales")
            plt.plot(forecast_values['yhat'], label="Forecasted Sales", linestyle='--')
            plt.xlabel("Date")
            plt.ylabel("Quantity Sold")
            plt.title(f"Sales Forecast for {stock_code}")
            plt.legend()
            st.pyplot(plt)

            # --------------------------------
            # CSV Download Functionality
            # --------------------------------

            @st.cache_data
            def convert_df_to_csv(df):
                return df.to_csv().encode('utf-8')  # Ensure index is included

            # Convert forecast data to CSV for download
            forecast_df = pd.DataFrame(forecast_values, columns=['yhat'])

            # Check if forecast_df has data before downloading
            if not forecast_df.empty:
                csv_data = convert_df_to_csv(forecast_df)
                st.download_button(
                    label="Download Forecast as CSV",
                    data=csv_data,
                    file_name=f'{stock_code}_forecast.csv',
                    mime='text/csv'
                )
            else:
                st.warning("Forecast data is empty. Unable to download CSV.")

        except Exception as e:
            st.error(f"Error in forecasting: {str(e)}")
