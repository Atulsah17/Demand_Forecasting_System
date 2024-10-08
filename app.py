import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import streamlit as st

# -----------------------------
# 1. Load and Preprocess Data
# -----------------------------

def load_data(file_path):
    """Load CSV data and handle empty files."""
    try:
        data = pd.read_csv(file_path)
        if data.empty:
            st.error(f"The file {file_path} is empty.")
        return data
    except pd.errors.EmptyDataError:
        st.error(f"No columns to parse from file: {file_path}")
        return pd.DataFrame()

# Load datasets 
transactions_01 = load_data('data/Transactional_data_retail_01.csv')
transactions_02 = load_data('data/Transactional_data_retail_02.csv')

# Check if both datasets are not empty before concatenation
if not transactions_01.empty and not transactions_02.empty:
    # Combine the transactions data
    transactions = pd.concat([transactions_01, transactions_02])
else:
    transactions = pd.DataFrame()  # Empty DataFrame in case of missing data

if not transactions.empty:
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
                st.write("Fitting ETS model (without seasonality)...")

                # Apply ETS model from statsmodels without seasonality
                ets_model = sm.tsa.ExponentialSmoothing(
                    weekly_sales,
                    trend='add',  # Additive trend
                    seasonal=None,  # Disable seasonality
                    initialization_method="estimated"
                ).fit()

                # Forecast for the specified number of weeks
                forecast_values = ets_model.forecast(weeks_to_forecast)

                # Debugging Step: Check forecast values
                st.write("Forecast Values:")
                st.write(forecast_values)

                # Create a forecast index
                forecast_index = pd.date_range(start=weekly_sales.index[-1] + pd.Timedelta(days=1), periods=weeks_to_forecast, freq='W')
                forecast_series = pd.Series(forecast_values, index=forecast_index)

                # Debug: Check if forecast_series has data
                st.write("Forecast Series:")
                st.write(forecast_series)

                # Plot historical and forecasted data
                st.write(f"Historical and Forecasted Sales for Stock Code {stock_code}")
                plt.figure(figsize=(10, 6))
                plt.plot(weekly_sales, label="Historical Sales")
                plt.plot(forecast_series, label="Forecasted Sales", linestyle='--')
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
                forecast_df = pd.DataFrame(forecast_series, columns=['Forecasted Quantity'])

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

else:
    st.error("One or both transaction data files are empty or invalid.")
