import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import logging

class CostAnomalyDetector:
    """Detects anomalies in cloud cost data using statistical and ML methods"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
    
    def detect_anomalies(self, data, threshold=2.0):
        """
        Detect cost anomalies in the data
        
        Args:
            data (pd.DataFrame): Cost data
            threshold (float): Anomaly detection threshold (standard deviations)
            
        Returns:
            list: List of detected anomalies
        """
        try:
            if data is None or data.empty:
                self.logger.warning("No data provided for anomaly detection")
                return []
            
            anomalies = []
            
            # Statistical anomaly detection
            statistical_anomalies = self._detect_statistical_anomalies(data, threshold)
            anomalies.extend(statistical_anomalies)
            
            # ML-based anomaly detection
            ml_anomalies = self._detect_ml_anomalies(data)
            anomalies.extend(ml_anomalies)
            
            # Time-based anomaly detection
            time_anomalies = self._detect_time_based_anomalies(data)
            anomalies.extend(time_anomalies)
            
            # Resource-specific anomaly detection
            resource_anomalies = self._detect_resource_anomalies(data)
            anomalies.extend(resource_anomalies)
            
            # Remove duplicates and sort by severity
            unique_anomalies = self._deduplicate_anomalies(anomalies)
            sorted_anomalies = sorted(unique_anomalies, key=lambda x: x.get('severity_score', 0), reverse=True)
            
            self.logger.info(f"Detected {len(sorted_anomalies)} anomalies")
            return sorted_anomalies
            
        except Exception as e:
            self.logger.error(f"Error detecting anomalies: {str(e)}")
            return []
    
    def _detect_statistical_anomalies(self, data, threshold):
        """Detect anomalies using statistical methods"""
        anomalies = []
        
        try:
            # Calculate z-scores for cost
            mean_cost = data['cost'].mean()
            std_cost = data['cost'].std()
            
            if std_cost == 0:
                return anomalies
            
            data['z_score'] = (data['cost'] - mean_cost) / std_cost
            
            # Identify outliers
            outliers = data[abs(data['z_score']) > threshold]
            
            for _, row in outliers.iterrows():
                anomaly = {
                    'timestamp': row['timestamp'],
                    'resource_id': row['resource_id'],
                    'resource_type': row['resource_type'],
                    'actual_cost': row['cost'],
                    'expected_cost': mean_cost,
                    'deviation_percentage': abs((row['cost'] - mean_cost) / mean_cost) * 100,
                    'z_score': row['z_score'],
                    'severity': self._calculate_severity(row['z_score']),
                    'severity_score': abs(row['z_score']),
                    'detection_method': 'statistical',
                    'description': f"Cost significantly deviates from normal (z-score: {row['z_score']:.2f})"
                }
                anomalies.append(anomaly)
        
        except Exception as e:
            self.logger.error(f"Error in statistical anomaly detection: {str(e)}")
        
        return anomalies
    
    def _detect_ml_anomalies(self, data):
        """Detect anomalies using machine learning"""
        anomalies = []
        
        try:
            # Prepare features for ML model
            features = []
            indices = []
            
            for idx, row in data.iterrows():
                feature_vector = [
                    row['cost'],
                    row['hour'],
                    row['day_of_week'],
                    row['usage_hours'] if 'usage_hours' in row and pd.notna(row['usage_hours']) else 0
                ]
                features.append(feature_vector)
                indices.append(idx)
            
            if len(features) < 10:  # Need minimum samples for ML
                return anomalies
            
            # Scale features
            features_scaled = self.scaler.fit_transform(features)
            
            # Detect anomalies
            anomaly_labels = self.isolation_forest.fit_predict(features_scaled)
            
            # Process results
            for i, label in enumerate(anomaly_labels):
                if label == -1:  # Anomaly detected
                    idx = indices[i]
                    row = data.loc[idx]
                    
                    anomaly = {
                        'timestamp': row['timestamp'],
                        'resource_id': row['resource_id'],
                        'resource_type': row['resource_type'],
                        'actual_cost': row['cost'],
                        'expected_cost': data['cost'].median(),
                        'deviation_percentage': abs((row['cost'] - data['cost'].median()) / data['cost'].median()) * 100,
                        'severity': 'Medium',
                        'severity_score': 1.0,
                        'detection_method': 'ml',
                        'description': "Anomaly detected by machine learning model"
                    }
                    anomalies.append(anomaly)
        
        except Exception as e:
            self.logger.error(f"Error in ML anomaly detection: {str(e)}")
        
        return anomalies
    
    def _detect_time_based_anomalies(self, data):
        """Detect anomalies based on time patterns"""
        anomalies = []
        
        try:
            # Group by time periods and detect anomalies
            daily_costs = data.groupby(data['timestamp'].dt.date)['cost'].sum()
            
            # Calculate rolling statistics
            rolling_mean = daily_costs.rolling(window=7, min_periods=3).mean()
            rolling_std = daily_costs.rolling(window=7, min_periods=3).std()
            
            # Detect anomalies
            for date, cost in daily_costs.items():
                if pd.notna(rolling_mean[date]) and pd.notna(rolling_std[date]):
                    expected_cost = rolling_mean[date]
                    threshold_cost = expected_cost + 2 * rolling_std[date]
                    
                    if cost > threshold_cost:
                        # Find the specific records for this date
                        date_records = data[data['timestamp'].dt.date == date]
                        
                        for _, row in date_records.iterrows():
                            anomaly = {
                                'timestamp': row['timestamp'],
                                'resource_id': row['resource_id'],
                                'resource_type': row['resource_type'],
                                'actual_cost': row['cost'],
                                'expected_cost': expected_cost / len(date_records),
                                'deviation_percentage': ((cost - expected_cost) / expected_cost) * 100,
                                'severity': 'High' if cost > threshold_cost * 1.5 else 'Medium',
                                'severity_score': (cost - expected_cost) / rolling_std[date],
                                'detection_method': 'time_based',
                                'description': f"Daily cost spike detected on {date}"
                            }
                            anomalies.append(anomaly)
        
        except Exception as e:
            self.logger.error(f"Error in time-based anomaly detection: {str(e)}")
        
        return anomalies
    
    def _detect_resource_anomalies(self, data):
        """Detect anomalies at the resource level"""
        anomalies = []
        
        try:
            # Group by resource and detect anomalies
            for resource_id, group in data.groupby('resource_id'):
                if len(group) < 5:  # Skip resources with insufficient data
                    continue
                
                # Calculate resource-specific statistics
                mean_cost = group['cost'].mean()
                std_cost = group['cost'].std()
                
                if std_cost == 0:
                    continue
                
                # Find outliers within this resource
                outliers = group[abs(group['cost'] - mean_cost) > 2 * std_cost]
                
                for _, row in outliers.iterrows():
                    anomaly = {
                        'timestamp': row['timestamp'],
                        'resource_id': row['resource_id'],
                        'resource_type': row['resource_type'],
                        'actual_cost': row['cost'],
                        'expected_cost': mean_cost,
                        'deviation_percentage': abs((row['cost'] - mean_cost) / mean_cost) * 100,
                        'severity': self._calculate_severity_by_deviation(row['cost'], mean_cost, std_cost),
                        'severity_score': abs(row['cost'] - mean_cost) / std_cost,
                        'detection_method': 'resource_specific',
                        'description': f"Resource {resource_id} shows unusual cost pattern"
                    }
                    anomalies.append(anomaly)
        
        except Exception as e:
            self.logger.error(f"Error in resource-specific anomaly detection: {str(e)}")
        
        return anomalies
    
    def _calculate_severity(self, z_score):
        """Calculate severity based on z-score"""
        abs_z = abs(z_score)
        if abs_z >= 3:
            return 'Critical'
        elif abs_z >= 2.5:
            return 'High'
        elif abs_z >= 2:
            return 'Medium'
        else:
            return 'Low'
    
    def _calculate_severity_by_deviation(self, actual, expected, std):
        """Calculate severity based on deviation from expected value"""
        deviation = abs(actual - expected) / std
        if deviation >= 3:
            return 'Critical'
        elif deviation >= 2.5:
            return 'High'
        elif deviation >= 2:
            return 'Medium'
        else:
            return 'Low'
    
    def _deduplicate_anomalies(self, anomalies):
        """Remove duplicate anomalies"""
        seen = set()
        unique_anomalies = []
        
        for anomaly in anomalies:
            key = (anomaly['timestamp'], anomaly['resource_id'])
            if key not in seen:
                seen.add(key)
                unique_anomalies.append(anomaly)
        
        return unique_anomalies
    
    def analyze_anomaly_patterns(self, anomalies):
        """Analyze patterns in detected anomalies"""
        if not anomalies:
            return {}
        
        analysis = {
            'total_anomalies': len(anomalies),
            'severity_distribution': {},
            'resource_type_distribution': {},
            'detection_method_distribution': {},
            'temporal_patterns': {}
        }
        
        # Severity distribution
        for anomaly in anomalies:
            severity = anomaly.get('severity', 'Unknown')
            analysis['severity_distribution'][severity] = analysis['severity_distribution'].get(severity, 0) + 1
        
        # Resource type distribution
        for anomaly in anomalies:
            resource_type = anomaly.get('resource_type', 'Unknown')
            analysis['resource_type_distribution'][resource_type] = analysis['resource_type_distribution'].get(resource_type, 0) + 1
        
        # Detection method distribution
        for anomaly in anomalies:
            method = anomaly.get('detection_method', 'Unknown')
            analysis['detection_method_distribution'][method] = analysis['detection_method_distribution'].get(method, 0) + 1
        
        # Temporal patterns
        hours = [anomaly['timestamp'].hour for anomaly in anomalies if 'timestamp' in anomaly]
        if hours:
            hour_counts = {}
            for hour in hours:
                hour_counts[hour] = hour_counts.get(hour, 0) + 1
            analysis['temporal_patterns']['hourly'] = hour_counts
        
        return analysis
