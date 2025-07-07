import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging

class UsageAnalyzer:
    """Analyzes usage patterns and identifies trends in cloud resource consumption"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler()
    
    def analyze_patterns(self, data):
        """
        Analyze usage patterns in the data
        
        Args:
            data (pd.DataFrame): Processed usage data
            
        Returns:
            dict: Analysis results containing patterns and insights
        """
        try:
            if data is None or data.empty:
                self.logger.warning("No data provided for pattern analysis")
                return {}
            
            patterns = []
            
            # Analyze hourly patterns
            hourly_patterns = self._analyze_hourly_patterns(data)
            if hourly_patterns:
                patterns.extend(hourly_patterns)
            
            # Analyze daily patterns
            daily_patterns = self._analyze_daily_patterns(data)
            if daily_patterns:
                patterns.extend(daily_patterns)
            
            # Analyze resource utilization
            utilization_patterns = self._analyze_utilization_patterns(data)
            if utilization_patterns:
                patterns.extend(utilization_patterns)
            
            # Analyze seasonal patterns
            seasonal_patterns = self._analyze_seasonal_patterns(data)
            if seasonal_patterns:
                patterns.extend(seasonal_patterns)
            
            # Identify underutilized resources
            underutilized = self._identify_underutilized_resources(data)
            if underutilized:
                patterns.extend(underutilized)
            
            self.logger.info(f"Identified {len(patterns)} usage patterns")
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error analyzing usage patterns: {str(e)}")
            return {}
    
    def _analyze_hourly_patterns(self, data):
        """Analyze hourly usage patterns"""
        patterns = []
        
        try:
            # Group by hour and calculate average usage
            hourly_usage = data.groupby('hour')['cost'].mean()
            
            # Find peak hours
            peak_threshold = hourly_usage.quantile(0.75)
            peak_hours = hourly_usage[hourly_usage > peak_threshold].index.tolist()
            
            if peak_hours:
                patterns.append({
                    'type': 'peak_hours',
                    'description': f"Peak usage typically occurs during hours: {', '.join(map(str, peak_hours))}",
                    'hours': peak_hours,
                    'average_cost': hourly_usage[peak_hours].mean(),
                    'confidence': 0.8
                })
            
            # Find low usage hours
            low_threshold = hourly_usage.quantile(0.25)
            low_hours = hourly_usage[hourly_usage < low_threshold].index.tolist()
            
            if low_hours:
                patterns.append({
                    'type': 'low_hours',
                    'description': f"Low usage typically occurs during hours: {', '.join(map(str, low_hours))}",
                    'hours': low_hours,
                    'average_cost': hourly_usage[low_hours].mean(),
                    'confidence': 0.8
                })
            
            # Add hourly pattern data for visualization
            patterns.append({
                'type': 'hourly_pattern',
                'hourly_pattern': hourly_usage.to_dict()
            })
            
        except Exception as e:
            self.logger.error(f"Error analyzing hourly patterns: {str(e)}")
        
        return patterns
    
    def _analyze_daily_patterns(self, data):
        """Analyze daily usage patterns"""
        patterns = []
        
        try:
            # Group by day of week
            daily_usage = data.groupby('day_of_week')['cost'].mean()
            
            # Map day numbers to names
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            daily_usage.index = [day_names[i] for i in daily_usage.index]
            
            # Find weekday vs weekend patterns
            weekday_usage = daily_usage[['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']].mean()
            weekend_usage = daily_usage[['Saturday', 'Sunday']].mean()
            
            if weekday_usage > weekend_usage * 1.2:
                patterns.append({
                    'type': 'weekday_pattern',
                    'description': f"Higher usage on weekdays (avg: ${weekday_usage:.2f}) vs weekends (avg: ${weekend_usage:.2f})",
                    'weekday_avg': weekday_usage,
                    'weekend_avg': weekend_usage,
                    'confidence': 0.7
                })
            elif weekend_usage > weekday_usage * 1.2:
                patterns.append({
                    'type': 'weekend_pattern',
                    'description': f"Higher usage on weekends (avg: ${weekend_usage:.2f}) vs weekdays (avg: ${weekday_usage:.2f})",
                    'weekday_avg': weekday_usage,
                    'weekend_avg': weekend_usage,
                    'confidence': 0.7
                })
            
        except Exception as e:
            self.logger.error(f"Error analyzing daily patterns: {str(e)}")
        
        return patterns
    
    def _analyze_utilization_patterns(self, data):
        """Analyze resource utilization patterns"""
        patterns = []
        
        try:
            if 'usage_hours' in data.columns:
                # Analyze utilization by resource type
                utilization_by_type = data.groupby('resource_type')['usage_hours'].agg(['mean', 'std', 'count'])
                
                for resource_type, stats in utilization_by_type.iterrows():
                    avg_usage = stats['mean']
                    std_usage = stats['std']
                    count = stats['count']
                    
                    if avg_usage < 6:  # Low utilization
                        patterns.append({
                            'type': 'low_utilization',
                            'description': f"{resource_type} resources show low utilization (avg: {avg_usage:.1f} hours)",
                            'resource_type': resource_type,
                            'average_usage': avg_usage,
                            'resource_count': count,
                            'confidence': 0.9 if std_usage < avg_usage else 0.6
                        })
                    elif avg_usage > 20:  # High utilization
                        patterns.append({
                            'type': 'high_utilization',
                            'description': f"{resource_type} resources show high utilization (avg: {avg_usage:.1f} hours)",
                            'resource_type': resource_type,
                            'average_usage': avg_usage,
                            'resource_count': count,
                            'confidence': 0.9 if std_usage < avg_usage else 0.6
                        })
        
        except Exception as e:
            self.logger.error(f"Error analyzing utilization patterns: {str(e)}")
        
        return patterns
    
    def _analyze_seasonal_patterns(self, data):
        """Analyze seasonal usage patterns"""
        patterns = []
        
        try:
            # Group by month
            monthly_usage = data.groupby('month')['cost'].mean()
            
            # Calculate coefficient of variation
            cv = monthly_usage.std() / monthly_usage.mean()
            
            if cv > 0.3:  # Significant variation
                peak_month = monthly_usage.idxmax()
                low_month = monthly_usage.idxmin()
                
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                
                patterns.append({
                    'type': 'seasonal',
                    'description': f"Seasonal pattern detected: Peak in {month_names[peak_month-1]}, Low in {month_names[low_month-1]}",
                    'peak_month': peak_month,
                    'low_month': low_month,
                    'variation_coefficient': cv,
                    'confidence': 0.8 if cv > 0.5 else 0.6
                })
        
        except Exception as e:
            self.logger.error(f"Error analyzing seasonal patterns: {str(e)}")
        
        return patterns
    
    def _identify_underutilized_resources(self, data):
        """Identify consistently underutilized resources"""
        patterns = []
        
        try:
            if 'usage_hours' in data.columns:
                # Find resources with consistently low usage
                resource_usage = data.groupby('resource_id')['usage_hours'].agg(['mean', 'std', 'count'])
                
                # Filter for resources with sufficient data points
                reliable_resources = resource_usage[resource_usage['count'] >= 5]
                
                # Identify underutilized resources
                underutilized = reliable_resources[
                    (reliable_resources['mean'] < 4) &  # Less than 4 hours average
                    (reliable_resources['std'] < 2)     # Low variability
                ]
                
                if len(underutilized) > 0:
                    patterns.append({
                        'type': 'underutilized',
                        'description': f"{len(underutilized)} resources are consistently underutilized",
                        'resource_count': len(underutilized),
                        'resources': underutilized.index.tolist(),
                        'average_usage': underutilized['mean'].mean(),
                        'confidence': 0.9
                    })
        
        except Exception as e:
            self.logger.error(f"Error identifying underutilized resources: {str(e)}")
        
        return patterns
    
    def cluster_resources(self, data, n_clusters=5):
        """
        Cluster resources based on usage patterns
        
        Args:
            data (pd.DataFrame): Usage data
            n_clusters (int): Number of clusters
            
        Returns:
            dict: Clustering results
        """
        try:
            if data is None or data.empty:
                return {}
            
            # Prepare features for clustering
            features = []
            resource_ids = []
            
            for resource_id, group in data.groupby('resource_id'):
                if len(group) < 5:  # Skip resources with insufficient data
                    continue
                
                feature_vector = [
                    group['cost'].mean(),
                    group['cost'].std(),
                    group['usage_hours'].mean() if 'usage_hours' in group.columns else 0,
                    group['usage_hours'].std() if 'usage_hours' in group.columns else 0,
                    len(group)
                ]
                
                features.append(feature_vector)
                resource_ids.append(resource_id)
            
            if len(features) < n_clusters:
                return {}
            
            # Perform clustering
            features_scaled = self.scaler.fit_transform(features)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(features_scaled)
            
            # Organize results
            cluster_results = {}
            for i, resource_id in enumerate(resource_ids):
                cluster_id = clusters[i]
                if cluster_id not in cluster_results:
                    cluster_results[cluster_id] = []
                cluster_results[cluster_id].append(resource_id)
            
            return cluster_results
            
        except Exception as e:
            self.logger.error(f"Error clustering resources: {str(e)}")
            return {}
