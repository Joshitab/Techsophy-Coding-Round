import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re

def format_currency(amount):
    """Format amount as currency string"""
    if pd.isna(amount) or amount is None:
        return "$0.00"
    
    if amount >= 1000000:
        return f"${amount/1000000:.1f}M"
    elif amount >= 1000:
        return f"${amount/1000:.1f}K"
    else:
        return f"${amount:.2f}"

def calculate_percentage_change(current, previous):
    """Calculate percentage change between two values"""
    if previous == 0:
        return 0 if current == 0 else 100
    
    return ((current - previous) / previous) * 100

def validate_date_range(start_date, end_date):
    """Validate date range"""
    if start_date > end_date:
        return False, "Start date must be before end date"
    
    if (end_date - start_date).days > 365:
        return False, "Date range cannot exceed 365 days"
    
    if end_date > datetime.now().date():
        return False, "End date cannot be in the future"
    
    return True, ""

def clean_resource_id(resource_id):
    """Clean and standardize resource ID"""
    if pd.isna(resource_id):
        return "unknown"
    
    # Remove special characters and convert to lowercase
    cleaned = re.sub(r'[^a-zA-Z0-9-_]', '', str(resource_id))
    return cleaned.lower()

def categorize_cost(cost):
    """Categorize cost into ranges"""
    if cost < 10:
        return "Very Low"
    elif cost < 50:
        return "Low"
    elif cost < 200:
        return "Medium"
    elif cost < 500:
        return "High"
    else:
        return "Very High"

def calculate_growth_rate(values, periods=7):
    """Calculate growth rate over specified periods"""
    if len(values) < periods:
        return 0
    
    recent_avg = np.mean(values[-periods:])
    previous_avg = np.mean(values[-periods*2:-periods])
    
    if previous_avg == 0:
        return 0
    
    return ((recent_avg - previous_avg) / previous_avg) * 100

def get_time_buckets(timestamp):
    """Get time buckets for grouping (hour, day, week, month)"""
    return {
        'hour': timestamp.hour,
        'day': timestamp.day,
        'week': timestamp.isocalendar()[1],
        'month': timestamp.month,
        'quarter': (timestamp.month - 1) // 3 + 1,
        'year': timestamp.year
    }

def calculate_cost_per_unit(cost, units):
    """Calculate cost per unit with error handling"""
    if units == 0 or pd.isna(units):
        return 0
    return cost / units

def detect_outliers(data, method='iqr', threshold=1.5):
    """Detect outliers in data using IQR method"""
    if len(data) < 4:
        return []
    
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    return outliers.index.tolist()

def aggregate_by_period(data, period='daily', value_column='cost'):
    """Aggregate data by time period"""
    if period == 'hourly':
        return data.groupby(data['timestamp'].dt.hour)[value_column].sum()
    elif period == 'daily':
        return data.groupby(data['timestamp'].dt.date)[value_column].sum()
    elif period == 'weekly':
        return data.groupby(data['timestamp'].dt.to_period('W'))[value_column].sum()
    elif period == 'monthly':
        return data.groupby(data['timestamp'].dt.to_period('M'))[value_column].sum()
    else:
        return data.groupby(data['timestamp'].dt.date)[value_column].sum()

def calculate_efficiency_score(cost, usage_hours, baseline_cost_per_hour=1.0):
    """Calculate efficiency score for resource usage"""
    if usage_hours == 0:
        return 0
    
    cost_per_hour = cost / usage_hours
    efficiency = baseline_cost_per_hour / cost_per_hour if cost_per_hour > 0 else 0
    
    # Normalize to 0-100 scale
    return min(100, max(0, efficiency * 100))

def format_time_duration(hours):
    """Format hours into human-readable duration"""
    if hours < 1:
        return f"{int(hours * 60)} minutes"
    elif hours < 24:
        return f"{hours:.1f} hours"
    else:
        days = hours // 24
        remaining_hours = hours % 24
        return f"{int(days)} days, {remaining_hours:.1f} hours"

def generate_date_range(start_date, end_date, freq='D'):
    """Generate date range with specified frequency"""
    return pd.date_range(start=start_date, end=end_date, freq=freq)

def safe_divide(numerator, denominator, default=0):
    """Safely divide two numbers with default value"""
    if denominator == 0 or pd.isna(denominator):
        return default
    return numerator / denominator

def calculate_trend_direction(values):
    """Calculate trend direction from a series of values"""
    if len(values) < 2:
        return "insufficient_data"
    
    # Simple linear regression slope
    x = np.arange(len(values))
    slope = np.polyfit(x, values, 1)[0]
    
    if slope > 0.05:
        return "increasing"
    elif slope < -0.05:
        return "decreasing"
    else:
        return "stable"

def get_business_hours_mask(timestamp_series):
    """Get mask for business hours (9 AM - 5 PM, Monday-Friday)"""
    business_hours = (
        (timestamp_series.dt.hour >= 9) & 
        (timestamp_series.dt.hour < 17) & 
        (timestamp_series.dt.weekday < 5)
    )
    return business_hours

def normalize_values(values, method='min-max'):
    """Normalize values using specified method"""
    if method == 'min-max':
        min_val = values.min()
        max_val = values.max()
        if max_val == min_val:
            return pd.Series([0.5] * len(values))
        return (values - min_val) / (max_val - min_val)
    elif method == 'z-score':
        return (values - values.mean()) / values.std()
    else:
        return values

def create_summary_stats(data, column):
    """Create summary statistics for a column"""
    if column not in data.columns:
        return {}
    
    return {
        'count': data[column].count(),
        'mean': data[column].mean(),
        'median': data[column].median(),
        'std': data[column].std(),
        'min': data[column].min(),
        'max': data[column].max(),
        'q25': data[column].quantile(0.25),
        'q75': data[column].quantile(0.75),
        'sum': data[column].sum()
    }
