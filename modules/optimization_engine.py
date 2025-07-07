import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

class OptimizationEngine:
    """Generates cost optimization recommendations based on usage patterns and analysis"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.recommendation_rules = self._initialize_recommendation_rules()
    
    def generate_recommendations(self, data, usage_patterns, cloud_provider='AWS'):
        """
        Generate cost optimization recommendations
        
        Args:
            data (pd.DataFrame): Usage data
            usage_patterns (list): Identified usage patterns
            cloud_provider (str): Cloud provider type
            
        Returns:
            list: List of optimization recommendations
        """
        try:
            if data is None or data.empty:
                self.logger.warning("No data provided for optimization")
                return []
            
            recommendations = []
            
            # Right-sizing recommendations
            rightsizing_recs = self._generate_rightsizing_recommendations(data, usage_patterns)
            recommendations.extend(rightsizing_recs)
            
            # Scheduling recommendations
            scheduling_recs = self._generate_scheduling_recommendations(data, usage_patterns)
            recommendations.extend(scheduling_recs)
            
            # Reserved instance recommendations
            reserved_recs = self._generate_reserved_instance_recommendations(data, cloud_provider)
            recommendations.extend(reserved_recs)
            
            # Storage optimization recommendations
            storage_recs = self._generate_storage_recommendations(data)
            recommendations.extend(storage_recs)
            
            # Idle resource recommendations
            idle_recs = self._generate_idle_resource_recommendations(data)
            recommendations.extend(idle_recs)
            
            # Auto-scaling recommendations
            autoscaling_recs = self._generate_autoscaling_recommendations(data, usage_patterns)
            recommendations.extend(autoscaling_recs)
            
            # Sort recommendations by potential savings
            sorted_recommendations = sorted(recommendations, key=lambda x: x.get('estimated_savings', 0), reverse=True)
            
            self.logger.info(f"Generated {len(sorted_recommendations)} optimization recommendations")
            return sorted_recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            return []
    
    def _initialize_recommendation_rules(self):
        """Initialize recommendation rules and thresholds"""
        return {
            'underutilization_threshold': 0.3,  # 30% utilization
            'overutilization_threshold': 0.8,   # 80% utilization
            'idle_threshold': 0.05,             # 5% utilization
            'cost_anomaly_threshold': 2.0,      # 2 standard deviations
            'savings_threshold': 50             # Minimum $50 savings to recommend
        }
    
    def _generate_rightsizing_recommendations(self, data, usage_patterns):
        """Generate rightsizing recommendations"""
        recommendations = []
        
        try:
            if 'usage_hours' not in data.columns:
                return recommendations
            
            # Analyze utilization by resource
            resource_utilization = data.groupby(['resource_id', 'resource_type', 'instance_size']).agg({
                'usage_hours': ['mean', 'max', 'std'],
                'cost': ['mean', 'sum']
            }).reset_index()
            
            # Flatten column names
            resource_utilization.columns = ['resource_id', 'resource_type', 'instance_size', 
                                          'avg_usage', 'max_usage', 'std_usage', 'avg_cost', 'total_cost']
            
            # Find underutilized resources
            underutilized = resource_utilization[
                (resource_utilization['avg_usage'] < 6) &  # Less than 6 hours average
                (resource_utilization['max_usage'] < 12) &  # Never more than 12 hours
                (resource_utilization['total_cost'] > 100)  # Significant cost
            ]
            
            for _, resource in underutilized.iterrows():
                # Estimate savings from downsizing
                estimated_savings = resource['total_cost'] * 0.4  # 40% savings from downsizing
                
                recommendations.append({
                    'title': f"Rightsize {resource['resource_type']} Instance",
                    'description': f"Resource {resource['resource_id']} is underutilized (avg: {resource['avg_usage']:.1f}h, max: {resource['max_usage']:.1f}h)",
                    'resource_id': resource['resource_id'],
                    'resource_type': resource['resource_type'],
                    'current_size': resource['instance_size'],
                    'suggested_action': "Downsize to smaller instance type",
                    'estimated_savings': estimated_savings,
                    'implementation': "Stop instance, change instance type, restart",
                    'effort': 'Medium',
                    'priority': 'High' if estimated_savings > 500 else 'Medium',
                    'risks': 'Potential performance impact during peak loads',
                    'timeline': '1-2 weeks',
                    'category': 'rightsizing'
                })
            
            # Find oversized resources
            oversized = resource_utilization[
                (resource_utilization['avg_usage'] > 20) &  # More than 20 hours average
                (resource_utilization['max_usage'] >= 24)   # At maximum capacity
            ]
            
            for _, resource in oversized.iterrows():
                recommendations.append({
                    'title': f"Consider Scaling {resource['resource_type']} Instance",
                    'description': f"Resource {resource['resource_id']} is highly utilized (avg: {resource['avg_usage']:.1f}h)",
                    'resource_id': resource['resource_id'],
                    'resource_type': resource['resource_type'],
                    'current_size': resource['instance_size'],
                    'suggested_action': "Monitor for performance issues, consider scaling",
                    'estimated_savings': 0,  # This is a performance recommendation
                    'implementation': "Monitor performance metrics, add auto-scaling",
                    'effort': 'Low',
                    'priority': 'Medium',
                    'risks': 'Performance degradation if not addressed',
                    'timeline': 'Immediate monitoring',
                    'category': 'performance'
                })
        
        except Exception as e:
            self.logger.error(f"Error generating rightsizing recommendations: {str(e)}")
        
        return recommendations
    
    def _generate_scheduling_recommendations(self, data, usage_patterns):
        """Generate scheduling recommendations based on usage patterns"""
        recommendations = []
        
        try:
            # Look for patterns that suggest scheduling opportunities
            for pattern in usage_patterns:
                if pattern.get('type') == 'low_hours':
                    low_hours = pattern.get('hours', [])
                    if len(low_hours) >= 8:  # At least 8 hours of low usage
                        
                        # Calculate potential savings
                        avg_hourly_cost = data['cost'].sum() / len(data)
                        potential_savings = avg_hourly_cost * len(low_hours) * 30 * 0.8  # 80% savings during off hours
                        
                        recommendations.append({
                            'title': 'Implement Scheduled Scaling',
                            'description': f'Low usage detected during hours {low_hours}. Consider scheduling resources to scale down.',
                            'suggested_action': 'Configure auto-scaling schedules to reduce capacity during low usage hours',
                            'estimated_savings': potential_savings,
                            'implementation': 'Set up CloudWatch events or equivalent to trigger scaling actions',
                            'effort': 'High',
                            'priority': 'High' if potential_savings > 1000 else 'Medium',
                            'risks': 'Potential service disruption if scaling is too aggressive',
                            'timeline': '2-4 weeks',
                            'category': 'scheduling'
                        })
                
                elif pattern.get('type') == 'weekday_pattern':
                    weekend_avg = pattern.get('weekend_avg', 0)
                    weekday_avg = pattern.get('weekday_avg', 0)
                    
                    if weekday_avg > weekend_avg * 1.5:
                        potential_savings = (weekday_avg - weekend_avg) * 8 * 4 * 0.6  # 60% savings on weekends
                        
                        recommendations.append({
                            'title': 'Weekend Resource Scaling',
                            'description': f'Significantly lower usage on weekends (${weekend_avg:.2f}) vs weekdays (${weekday_avg:.2f})',
                            'suggested_action': 'Scale down non-critical resources on weekends',
                            'estimated_savings': potential_savings,
                            'implementation': 'Configure weekend scaling policies',
                            'effort': 'Medium',
                            'priority': 'Medium',
                            'risks': 'May affect weekend maintenance windows',
                            'timeline': '1-2 weeks',
                            'category': 'scheduling'
                        })
        
        except Exception as e:
            self.logger.error(f"Error generating scheduling recommendations: {str(e)}")
        
        return recommendations
    
    def _generate_reserved_instance_recommendations(self, data, cloud_provider):
        """Generate reserved instance recommendations"""
        recommendations = []
        
        try:
            # Group by resource type and calculate consistent usage
            agg_dict = {
                'resource_id': 'nunique',
                'cost': 'sum'
            }
            
            # Only include usage_hours if it exists
            if 'usage_hours' in data.columns:
                agg_dict['usage_hours'] = 'mean'
            
            resource_usage = data.groupby(['resource_type', 'instance_size']).agg(agg_dict).reset_index()
            
            # Set column names based on what we have
            if 'usage_hours' in agg_dict:
                resource_usage.columns = ['resource_type', 'instance_size', 'unique_resources', 'avg_usage', 'total_cost']
                # Find resources suitable for reserved instances
                suitable_for_ri = resource_usage[
                    (resource_usage['avg_usage'] > 12) &  # More than 12 hours average usage
                    (resource_usage['unique_resources'] >= 2) &  # Multiple instances
                    (resource_usage['total_cost'] > 500)  # Significant cost
                ]
            else:
                resource_usage.columns = ['resource_type', 'instance_size', 'unique_resources', 'total_cost']
                # Find resources suitable for reserved instances (without usage criteria)
                suitable_for_ri = resource_usage[
                    (resource_usage['unique_resources'] >= 2) &  # Multiple instances
                    (resource_usage['total_cost'] > 500)  # Significant cost
                ]
            
            for _, resource in suitable_for_ri.iterrows():
                # Estimate savings (typically 30-60% for 1-year reserved instances)
                estimated_savings = resource['total_cost'] * 0.4 * 12  # 40% savings over 1 year
                
                description = f'Consistent usage of {resource["unique_resources"]} {resource["instance_size"]} instances'
                if 'avg_usage' in resource:
                    description += f' with high utilization (avg: {resource["avg_usage"]:.1f}h)'
                
                recommendations.append({
                    'title': f'Reserved Instance Opportunity - {resource["resource_type"]}',
                    'description': description,
                    'resource_type': resource['resource_type'],
                    'instance_size': resource['instance_size'],
                    'instance_count': resource['unique_resources'],
                    'suggested_action': 'Purchase 1-year reserved instances',
                    'estimated_savings': estimated_savings,
                    'implementation': f'Purchase reserved instances through {cloud_provider} console',
                    'effort': 'Low',
                    'priority': 'High',
                    'risks': 'Commitment to 1-year term, reduced flexibility',
                    'timeline': 'Immediate',
                    'category': 'reserved_instances'
                })
        
        except Exception as e:
            self.logger.error(f"Error generating reserved instance recommendations: {str(e)}")
        
        return recommendations
    
    def _generate_storage_recommendations(self, data):
        """Generate storage optimization recommendations"""
        recommendations = []
        
        try:
            # Find storage-related resources
            storage_resources = data[data['resource_type'].str.contains('S3|EBS|Storage', case=False, na=False)]
            
            if not storage_resources.empty:
                # Group by storage type
                storage_summary = storage_resources.groupby('resource_type').agg({
                    'cost': 'sum',
                    'resource_id': 'nunique'
                }).reset_index()
                
                for _, storage in storage_summary.iterrows():
                    if storage['cost'] > 200:  # Significant storage cost
                        # Estimate savings from storage optimization
                        estimated_savings = storage['cost'] * 0.3 * 12  # 30% savings annually
                        
                        recommendations.append({
                            'title': f'Storage Optimization - {storage["resource_type"]}',
                            'description': f'${storage["cost"]:.2f} monthly cost across {storage["resource_id"]} storage resources',
                            'resource_type': storage['resource_type'],
                            'suggested_action': 'Implement lifecycle policies, review storage classes, clean up unused data',
                            'estimated_savings': estimated_savings,
                            'implementation': 'Audit storage usage, implement automated lifecycle policies',
                            'effort': 'Medium',
                            'priority': 'Medium',
                            'risks': 'Data availability if lifecycle policies are too aggressive',
                            'timeline': '2-3 weeks',
                            'category': 'storage'
                        })
        
        except Exception as e:
            self.logger.error(f"Error generating storage recommendations: {str(e)}")
        
        return recommendations
    
    def _generate_idle_resource_recommendations(self, data):
        """Generate recommendations for idle resources"""
        recommendations = []
        
        try:
            if 'usage_hours' not in data.columns:
                return recommendations
            
            # Find truly idle resources
            idle_resources = data.groupby(['resource_id', 'resource_type']).agg({
                'usage_hours': ['mean', 'max'],
                'cost': 'sum'
            }).reset_index()
            
            idle_resources.columns = ['resource_id', 'resource_type', 'avg_usage', 'max_usage', 'total_cost']
            
            # Resources with very low usage
            truly_idle = idle_resources[
                (idle_resources['avg_usage'] < 0.5) &  # Less than 30 minutes average
                (idle_resources['max_usage'] < 2) &    # Never more than 2 hours
                (idle_resources['total_cost'] > 50)    # Costing more than $50
            ]
            
            for _, resource in truly_idle.iterrows():
                recommendations.append({
                    'title': f'Terminate Idle Resource - {resource["resource_type"]}',
                    'description': f'Resource {resource["resource_id"]} has minimal usage (avg: {resource["avg_usage"]:.1f}h) but costs ${resource["total_cost"]:.2f}',
                    'resource_id': resource['resource_id'],
                    'resource_type': resource['resource_type'],
                    'suggested_action': 'Investigate and potentially terminate unused resource',
                    'estimated_savings': resource['total_cost'] * 12,  # Annual savings
                    'implementation': 'Verify resource is not needed, create backup if necessary, terminate',
                    'effort': 'Low',
                    'priority': 'High',
                    'risks': 'Potential service disruption if resource is actually needed',
                    'timeline': '1 week',
                    'category': 'idle_resources'
                })
        
        except Exception as e:
            self.logger.error(f"Error generating idle resource recommendations: {str(e)}")
        
        return recommendations
    
    def _generate_autoscaling_recommendations(self, data, usage_patterns):
        """Generate auto-scaling recommendations"""
        recommendations = []
        
        try:
            # Check if usage_hours exists
            if 'usage_hours' not in data.columns:
                # Generate recommendations based on cost variability instead
                variable_resources = data.groupby(['resource_id', 'resource_type']).agg({
                    'cost': ['mean', 'std', 'min', 'max', 'sum']
                }).reset_index()
                
                variable_resources.columns = ['resource_id', 'resource_type', 'avg_cost', 'std_cost', 'min_cost', 'max_cost', 'total_cost']
                
                # Find resources with high cost variability
                high_variability = variable_resources[
                    (variable_resources['std_cost'] > variable_resources['avg_cost'] * 0.5) &  # High cost variation
                    (variable_resources['max_cost'] > variable_resources['min_cost'] * 2) &  # 2x cost variation
                    (variable_resources['total_cost'] > 200)  # Significant cost
                ]
                
                for _, resource in high_variability.iterrows():
                    # Estimate savings from auto-scaling
                    estimated_savings = resource['total_cost'] * 0.25 * 12  # 25% savings annually
                    
                    recommendations.append({
                        'title': f'Auto-Scaling Setup - {resource["resource_type"]}',
                        'description': f'Resource {resource["resource_id"]} shows high cost variability (std: ${resource["std_cost"]:.2f})',
                        'resource_id': resource['resource_id'],
                        'resource_type': resource['resource_type'],
                        'suggested_action': 'Configure auto-scaling to handle variable workloads',
                        'estimated_savings': estimated_savings,
                        'implementation': 'Set up auto-scaling policies based on CPU, memory, or custom metrics',
                        'effort': 'High',
                        'priority': 'Medium',
                        'risks': 'Complex setup, potential over-scaling',
                        'timeline': '3-4 weeks',
                        'category': 'autoscaling'
                    })
            else:
                # Look for variable usage patterns that could benefit from auto-scaling
                variable_resources = data.groupby(['resource_id', 'resource_type']).agg({
                    'usage_hours': ['mean', 'std', 'min', 'max'],
                    'cost': 'sum'
                }).reset_index()
                
                variable_resources.columns = ['resource_id', 'resource_type', 'avg_usage', 'std_usage', 'min_usage', 'max_usage', 'total_cost']
                
                # Find resources with high variability
                high_variability = variable_resources[
                    (variable_resources['std_usage'] > 4) &  # High standard deviation
                    (variable_resources['max_usage'] > variable_resources['min_usage'] * 3) &  # 3x variation
                    (variable_resources['total_cost'] > 200)  # Significant cost
                ]
                
                for _, resource in high_variability.iterrows():
                    # Estimate savings from auto-scaling
                    estimated_savings = resource['total_cost'] * 0.25 * 12  # 25% savings annually
                    
                    recommendations.append({
                        'title': f'Auto-Scaling Setup - {resource["resource_type"]}',
                        'description': f'Resource {resource["resource_id"]} shows high usage variability (std: {resource["std_usage"]:.1f}h)',
                        'resource_id': resource['resource_id'],
                        'resource_type': resource['resource_type'],
                        'suggested_action': 'Configure auto-scaling to handle variable workloads',
                        'estimated_savings': estimated_savings,
                        'implementation': 'Set up auto-scaling policies based on CPU, memory, or custom metrics',
                        'effort': 'High',
                        'priority': 'Medium',
                        'risks': 'Complex setup, potential over-scaling',
                        'timeline': '3-4 weeks',
                        'category': 'autoscaling'
                    })

        
        except Exception as e:
            self.logger.error(f"Error generating auto-scaling recommendations: {str(e)}")
        
        return recommendations
    
    def calculate_total_savings(self, recommendations):
        """Calculate total potential savings from all recommendations"""
        total_savings = sum(rec.get('estimated_savings', 0) for rec in recommendations)
        return total_savings
    
    def prioritize_recommendations(self, recommendations):
        """Prioritize recommendations based on savings and effort"""
        def priority_score(rec):
            savings = rec.get('estimated_savings', 0)
            effort_weights = {'Low': 1, 'Medium': 0.7, 'High': 0.4}
            effort = rec.get('effort', 'Medium')
            return savings * effort_weights.get(effort, 0.5)
        
        return sorted(recommendations, key=priority_score, reverse=True)
