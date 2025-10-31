import json
import numpy as np
import pandas as pd
from datetime import datetime
import statsmodels.api as sm
import logging
import io
import csv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_csv_data(csv_content):
    """
    Parse CSV content into pandas DataFrame, handling the specific 5-minute bar format
    
    Args:
        csv_content (str): CSV data as a string
        
    Returns:
        pandas.DataFrame: Parsed dataframe with properly formatted columns
    """
    # Parse CSV into DataFrame
    df = pd.read_csv(io.StringIO(csv_content), 
                     parse_dates=['DateTime'],
                     dayfirst=False,
                     dtype={
                         'Open': str,
                         'High': str,
                         'Low': str,
                         'Close': str,
                         'Volume(from bar)': float,
                         'Cumulative delta (By volume)_Cumulative open': float,
                         'Cumulative delta (By volume)_Cumulative high': float,
                         'Cumulative delta (By volume)_Cumulative low': float,
                         'Cumulative delta (By volume)_Cumulative close': float,
                         'Delta': float
                     })
    
    # Clean and convert price columns (handle European format with comma as decimal)
    price_columns = ['Open', 'High', 'Low', 'Close']
    for col in price_columns:
        if col in df.columns:
            # Replace comma with dot and convert to float
            df[col] = df[col].str.replace(',', '.', regex=False).astype(float)
    
    return df

def timeseries_analysis(request):
    """
    Cloud Function that performs time series analysis on provided 5-minute bar CSV data
    Args:
        request (flask.Request): HTTP request object.
    Returns:
        The response text or any set of values that can be turned into a
        Response object using `make_response`.
    """
    try:
        # Check if request has a file upload (CSV)
        if request.files and 'file' in request.files:
            # Process CSV file upload
            file = request.files['file']
            content = file.read().decode('utf-8')
            df = parse_csv_data(content)
            
        # If no file, check for JSON with CSV content as string
        else:
            request_json = request.get_json(silent=True)
            
            if not request_json:
                return json.dumps({
                    'error': 'No data provided in request. Please upload a CSV file or provide CSV content as string in JSON',
                    'status': 'error'
                }), 400
            
            # Extract CSV content from JSON
            csv_content = request_json.get('csv_data')
            if not csv_content:
                return json.dumps({
                    'error': 'No CSV data provided in the request',
                    'status': 'error'
                }), 400

            df = parse_csv_data(csv_content)

        # Extract other parameters from request
        request_args = request.args if hasattr(request, 'args') else {}
        request_json = request.get_json(silent=True) or {}

        # Get parameters from either query params or JSON body
        model_type = request_args.get('model_type') or request_json.get('model_type', 'arima')
        
        # Handle forecast_periods as string or int
        forecast_periods_str = request_args.get('forecast_periods') or request_json.get('forecast_periods', '12')
        try:
            forecast_periods = int(forecast_periods_str)
        except (ValueError, TypeError):
            forecast_periods = 12  # Default to 1 hour (12 periods of 5 min)
            
        target_column = request_args.get('target_column') or request_json.get('target_column', 'Close')
        
        # Validate the DataFrame and target column
        if target_column not in df.columns:
            return json.dumps({
                'error': f'Target column "{target_column}" not found in data. Available columns: {list(df.columns)}',
                'status': 'error'
            }), 400

        # Validate DateTime column exists
        if 'DateTime' not in df.columns:
            return json.dumps({
                'error': 'DateTime column not found in data. Required for time series analysis.',
                'status': 'error'
            }), 400

        logger.info(f"Processing {len(df)} records with model: {model_type}, target: {target_column}")

        # Use DateTime as index and sort
        try:
            df = df.sort_values('DateTime')
        except Exception as e:
            return json.dumps({
                'error': f'Error sorting by DateTime: {str(e)}. Ensure DateTime column is properly formatted.',
                'status': 'error'
            }), 400

        # Perform the analysis based on the requested model type
        if model_type.lower() == 'arima':
            result = perform_arima_analysis(df, forecast_periods, target_column)
        elif model_type.lower() == 'exponential_smoothing':
            result = perform_exponential_smoothing(df, forecast_periods, target_column)
        else:
            return json.dumps({
                'error': f'Unsupported model type: {model_type}',
                'status': 'error'
            }), 400
        
        # Add metadata about the analysis
        result['metadata'] = {
            'target_column': target_column,
            'data_points': len(df),
            'start_date': df['DateTime'].min().strftime('%Y-%m-%d %H:%M:%S'),
            'end_date': df['DateTime'].max().strftime('%Y-%m-%d %H:%M:%S'),
            'interval': '5 minutes'
        }
        
        # Return the results
        return json.dumps({
            'results': result,
            'status': 'success'
        }), 200
        
    except Exception as e:
        logger.error(f"Error in timeseries_analysis: {str(e)}")
        return json.dumps({
            'error': f'Error processing request: {str(e)}',
            'status': 'error'
        }), 500

def perform_arima_analysis(df, forecast_periods, target_column):
    """
    Perform ARIMA time series analysis on 5-minute interval data
    
    Args:
        df (pandas.DataFrame): DataFrame with 5-minute bar data
        forecast_periods (int): Number of 5-minute periods to forecast
        target_column (str): Column to analyze (e.g., 'Close', 'Open', etc.)
        
    Returns:
        dict: Results of the analysis
    """
    # Create a time series from the data
    ts = df.set_index('DateTime')[target_column]
    
    # Check for stationarity and apply differencing if needed
    # For simplicity, we're using fixed parameters here
    # In a production system, you might want to use auto_arima from pmdarima
    p, d, q = 1, 1, 1
    
    # Fit the ARIMA model
    try:
        model = sm.tsa.ARIMA(ts, order=(p, d, q))
        model_fit = model.fit()
    except Exception as e:
        logger.warning(f"Error fitting ARIMA model with (1,1,1): {str(e)}. Trying (1,0,0).")
        try:
            # Fallback to simpler model if complex one fails
            model = sm.tsa.ARIMA(ts, order=(1, 0, 0))
            model_fit = model.fit()
        except Exception as e2:
            logger.error(f"Both ARIMA models failed. Error: {str(e2)}")
            raise ValueError(f"Unable to fit ARIMA model to the data: {str(e2)}")
    
    # Generate forecast
    forecast = model_fit.forecast(steps=forecast_periods)
    
    # Create forecast dates (continuing from last date in the dataset with 5-minute intervals)
    last_date = df['DateTime'].max()
    forecast_index = pd.date_range(start=last_date, periods=forecast_periods+1, freq='5min')[1:]
    
    # Prepare results - round to nearest 0.25 increment (0.00, 0.25, 0.50, or 0.75)
    forecast_data = [{'datetime': date.strftime('%Y-%m-%d %H:%M:%S'), 
                      'value': round(round(float(val) * 4) / 4, 2)} 
                    for date, val in zip(forecast_index, forecast)]
    
    # Add model summary statistics
    model_summary = {
        'aic': round(float(model_fit.aic), 2),
        'bic': round(float(model_fit.bic), 2),
        'model_type': 'ARIMA',
        'parameters': {'p': p, 'd': d, 'q': q}
    }
    
    # Add confidence intervals if available
    try:
        forecast_obj = model_fit.get_forecast(steps=forecast_periods)
        conf_int = forecast_obj.conf_int()

        # Add confidence intervals to forecast data - round to nearest 0.25 increment
        for i, entry in enumerate(forecast_data):
            entry['lower_ci'] = round(round(float(conf_int.iloc[i, 0]) * 4) / 4, 2)
            entry['upper_ci'] = round(round(float(conf_int.iloc[i, 1]) * 4) / 4, 2)
    except Exception as e:
        logger.warning(f"Could not compute confidence intervals: {str(e)}")
    
    return {
        'forecast': forecast_data,
        'model_summary': model_summary
    }

def perform_exponential_smoothing(df, forecast_periods, target_column):
    """
    Perform Exponential Smoothing time series analysis on 5-minute bar data
    
    Args:
        df (pandas.DataFrame): DataFrame with 5-minute bar data
        forecast_periods (int): Number of 5-minute periods to forecast
        target_column (str): Column to analyze (e.g., 'Close', 'Open', etc.)
        
    Returns:
        dict: Results of the analysis
    """
    # Create a time series from the data
    ts = df.set_index('DateTime')[target_column]
    
    # Determine seasonal period (288 = number of 5-min intervals in a day)
    # Adjust this based on your specific data patterns
    seasonal_periods = min(288, len(ts) // 2)  # Don't use more periods than half the data
    
    # For very short time series, use simple exponential smoothing without seasonality
    if len(ts) < 24:  # Less than 2 hours of data
        model = sm.tsa.SimpleExpSmoothing(ts)
        seasonal_type = None
        trend_type = None
    else:
        # Try to use Holt-Winters with seasonality if we have enough data
        try:
            model = sm.tsa.ExponentialSmoothing(
                ts, 
                trend='add',
                seasonal='add', 
                seasonal_periods=seasonal_periods
            )
            seasonal_type = 'additive'
            trend_type = 'additive'
        except Exception as e:
            logger.warning(f"Error setting up seasonal model: {str(e)}. Falling back to simple model.")
            # Fallback to simpler model
            model = sm.tsa.Holt(ts)
            seasonal_type = None
            trend_type = 'additive'
    
    # Fit the model
    try:
        model_fit = model.fit()
    except Exception as e:
        logger.warning(f"Error fitting complex model: {str(e)}. Trying simpler model.")
        # If fitting fails, try the simplest model
        model = sm.tsa.SimpleExpSmoothing(ts)
        model_fit = model.fit()
        seasonal_type = None
        trend_type = None

    # Generate forecast
    forecast = model_fit.forecast(steps=forecast_periods)
    
    # Create forecast dates (continuing from last date in the dataset with 5-minute intervals)
    last_date = df['DateTime'].max()
    forecast_index = pd.date_range(start=last_date, periods=forecast_periods+1, freq='5min')[1:]
    
    # Prepare results - round to nearest 0.25 increment (0.00, 0.25, 0.50, or 0.75)
    forecast_data = [{'datetime': date.strftime('%Y-%m-%d %H:%M:%S'), 
                      'value': round(round(float(val) * 4) / 4, 2)} 
                    for date, val in zip(forecast_index, forecast)]
    
    # Add model summary statistics
    model_summary = {
        'aic': round(float(model_fit.aic), 2) if hasattr(model_fit, 'aic') else None,
        'model_type': 'Exponential Smoothing',
        'parameters': {
            'trend': trend_type,
            'seasonal': seasonal_type,
            'seasonal_periods': seasonal_periods if seasonal_type else None
        }
    }
    
    return {
        'forecast': forecast_data,
        'model_summary': model_summary
    }

# Simple health check endpoint
def health_check(request):
    """
    A simple health check endpoint to verify the function is running
    """
    return json.dumps({
        'status': 'ok',
        'message': 'Time Series Analysis function is operational'
    })
