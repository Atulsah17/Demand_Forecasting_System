# Demand Forecasting System

## Project Overview

This project provides a **Demand Forecasting System** for optimizing inventory and supply chain efficiency. Using historical sales data, it forecasts the demand for the next 15 weeks for the top 10 best-selling products. The goal is to estimate future demand trends accurately to ensure optimal stock levels and meet customer demands.

### Key Features

- **Exploratory Data Analysis (EDA)**: Provides customer, item, and transaction-level summary statistics.
- **Top Products**: Identifies the top 10 products by total quantity sold and highest revenue generated.
- **Time Series Forecasting**: Supports various time series models like ARIMA, ETS, Prophet, and LSTM for demand forecasting.
- **Machine Learning Models**: Uses non-time series models like DecisionTree and XGBoost for demand prediction based on customer and product features.
- **Interactive Web Application**: Developed using Streamlit for real-time demand forecasting with CSV download functionality.
- **Evaluation Metrics**: Incorporates RMSE and MAE for model performance evaluation.
- **ACF/PACF Analysis**: Includes Auto-Correlation Function (ACF) and Partial Auto-Correlation Function (PACF) for trend and seasonality analysis.

## Project Structure

```bash
├── data/
│   ├── Transactional_data_retail_01.csv
│   ├── Transactional_data_retail_02.csv
│   ├── CustomerDemographics.csv
│   ├── ProductInfo.csv
├── app.py                 # Main Streamlit application file
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── .gitignore              # Ignoring unnecessary files for git

Installation Instructions
Clone the repository to your local machine:
git clone https://github.com/yourusername/demand-forecasting-system.git

Install the required Python packages:
pip install -r requirements.txt

How to Run the Application
Ensure that the required datasets are available in the data/ directory.

Run the Streamlit application:
streamlit run app.py

Models Used
ARIMA: Auto-Regressive Integrated Moving Average, a popular time series model.
Prophet: A model developed by Facebook for accurate time series forecasting.
Machine Learning: DecisionTree, XGBoost models that use customer and product data for non-time series prediction.
Forecasting Details
The application predicts demand for the top 10 best-selling products.
Users can select a product and specify the number of weeks for forecasting (up to 15 weeks).
The app plots historical sales data and forecasted demand.
Users can download forecast results as a CSV file.
Known Issues and Future Improvements
Data Limitations: Ensure that the data files include valid dates and quantities to prevent missing data issues.
Model Improvements: Experiment with advanced models like LSTM for better accuracy.



