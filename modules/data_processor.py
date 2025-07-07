import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

class DataProcessor:
    """Handles data ingestion, cleaning, and preprocessing for cloud cost analysis"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.required_columns = ['timestamp', 'resource_id', 'resource_type', 'cost', 'region']
        self.optional_columns = ['usage_hours', 'instance_size', 'tags']
    
    def process_data(self, raw_data):
        """
        Process raw cloud usage data into a standardized format
        
        Args:
            raw_data (pd.DataFrame): Raw data from CSV or API
            
        Returns:
            pd.DataFrame: Processed and cleaned data
        """
        try:
            if raw_data is None or raw_data.empty:
                self.logger.warning("No data provided for processing")
                return None
            
            # Validate required columns
            if not self._validate_columns(raw_data):
                self.logger.error("Required columns missing from data")
                return None
            
            # Clean and standardize data
            processed_data = self._clean_data(raw_data.copy())
            
            # Add derived columns
            processed_data = self._add_derived_columns(processed_data)
            
            # Validate data quality
            if not self._validate_data_quality(processed_data):
                self.logger.warning("Data quality issues detected")
            
            self.logger.info(f"Successfully processed {len(processed_data)} records")
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error processing data: {str(e)}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            return None
    
    def _validate_columns(self, data):
        """Validate that required columns are present"""
        missing_columns = [col for col in self.required_columns if col not in data.columns]
        
        if missing_columns:
            self.logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        return True
    
    def _clean_data(self, data):
        """Clean and standardize the data"""
        
        # Convert timestamp to datetime
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')
        
        # Clean numeric columns
        numeric_columns = ['cost', 'usage_hours']
        for col in numeric_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Clean string columns
        string_columns = ['resource_id', 'resource_type', 'region', 'instance_size']
        for col in string_columns:
            if col in data.columns:
                try:
                    # Convert to string first, then apply string methods
                    data[col] = data[col].astype(str)
                    # Simple string cleaning without using .str accessor
                    data[col] = data[col].apply(lambda x: str(x).strip() if pd.notna(x) else str(x))
                except Exception as e:
                    self.logger.warning(f"Could not clean column {col}: {str(e)}")
                    # Just convert to string as fallback
                    data[col] = data[col].astype(str)
        
        # Remove rows with critical missing values
        data = data.dropna(subset=['timestamp', 'resource_id', 'cost'])
        
        # Fill missing values for optional columns
        if 'usage_hours' in data.columns:
            data['usage_hours'] = data['usage_hours'].fillna(0)
        
        if 'instance_size' in data.columns:
            data['instance_size'] = data['instance_size'].fillna('Unknown')
        
        return data
    
    def _add_derived_columns(self, data):
        """Add derived columns for analysis"""
        
        # Add date components
        data['date'] = data['timestamp'].dt.date
        data['hour'] = data['timestamp'].dt.hour
        data['day_of_week'] = data['timestamp'].dt.dayofweek
        data['month'] = data['timestamp'].dt.month
        
        # Add cost per hour if usage hours available
        if 'usage_hours' in data.columns:
            data['cost_per_hour'] = data['cost'] / data['usage_hours'].replace(0, 1)
        
        # Add resource utilization category
        if 'usage_hours' in data.columns:
            data['utilization_category'] = data['usage_hours'].apply(self._categorize_utilization)
        
        return data
    
    def _categorize_utilization(self, usage_hours):
        """Categorize resource utilization"""
        if usage_hours == 0:
            return 'Idle'
        elif usage_hours < 4:
            return 'Low'
        elif usage_hours < 12:
            return 'Medium'
        elif usage_hours < 20:
            return 'High'
        else:
            return 'Very High'
    
    def _validate_data_quality(self, data):
        """Validate data quality and log issues"""
        issues = []
        
        # Check for negative costs
        if (data['cost'] < 0).any():
            issues.append("Negative cost values detected")
        
        # Check for future dates
        if (data['timestamp'] > datetime.now()).any():
            issues.append("Future timestamps detected")
        
        # Check for duplicate records
        duplicates = data.duplicated(subset=['timestamp', 'resource_id']).sum()
        if duplicates > 0:
            issues.append(f"{duplicates} duplicate records detected")
        
        # Check for extreme values
        cost_q99 = data['cost'].quantile(0.99)
        extreme_costs = (data['cost'] > cost_q99 * 10).sum()
        if extreme_costs > 0:
            issues.append(f"{extreme_costs} extreme cost values detected")
        
        if issues:
            for issue in issues:
                self.logger.warning(issue)
            return False
        
        return True
    
    def get_data_summary(self, data):
        """Get summary statistics of the processed data"""
        if data is None or data.empty:
            return {}
        
        summary = {
            'total_records': len(data),
            'date_range': {
                'start': data['timestamp'].min(),
                'end': data['timestamp'].max()
            },
            'total_cost': data['cost'].sum(),
            'unique_resources': data['resource_id'].nunique(),
            'resource_types': data['resource_type'].value_counts().to_dict(),
            'regions': data['region'].value_counts().to_dict(),
            'cost_stats': {
                'mean': data['cost'].mean(),
                'median': data['cost'].median(),
                'std': data['cost'].std(),
                'min': data['cost'].min(),
                'max': data['cost'].max()
            }
        }
        
        return summary
