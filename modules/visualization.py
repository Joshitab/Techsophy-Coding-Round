import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

class CostVisualization:
    """Handles creation of cost analysis visualizations"""
    
    def __init__(self):
        self.color_scheme = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'warning': '#d62728',
            'info': '#9467bd'
        }
    
    def create_cost_trend_chart(self, data, period='daily'):
        """Create cost trend visualization"""
        try:
            if period == 'daily':
                trend_data = data.groupby(data['timestamp'].dt.date)['cost'].sum().reset_index()
                trend_data.columns = ['date', 'total_cost']
                x_col, title = 'date', 'Daily Cost Trend'
            elif period == 'hourly':
                trend_data = data.groupby(data['timestamp'].dt.hour)['cost'].mean().reset_index()
                trend_data.columns = ['hour', 'avg_cost']
                x_col, title = 'hour', 'Average Cost by Hour'
            else:
                trend_data = data.groupby(data['timestamp'].dt.to_period('W'))['cost'].sum().reset_index()
                trend_data.columns = ['week', 'total_cost']
                x_col, title = 'week', 'Weekly Cost Trend'
            
            fig = px.line(
                trend_data, 
                x=x_col, 
                y=trend_data.columns[1],
                title=title,
                labels={trend_data.columns[1]: 'Cost ($)', x_col: x_col.capitalize()}
            )
            
            fig.update_layout(
                xaxis_title=x_col.capitalize(),
                yaxis_title='Cost ($)',
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            return self._create_error_chart(f"Error creating cost trend: {str(e)}")
    
    def create_resource_distribution_chart(self, data, group_by='resource_type'):
        """Create resource cost distribution chart"""
        try:
            if group_by == 'resource_type':
                dist_data = data.groupby('resource_type')['cost'].sum().reset_index()
                title = 'Cost Distribution by Resource Type'
            elif group_by == 'region':
                dist_data = data.groupby('region')['cost'].sum().reset_index()
                title = 'Cost Distribution by Region'
            else:
                dist_data = data.groupby('instance_size')['cost'].sum().reset_index()
                title = 'Cost Distribution by Instance Size'
            
            fig = px.pie(
                dist_data,
                values='cost',
                names=group_by,
                title=title
            )
            
            fig.update_traces(
                textposition='inside',
                textinfo='percent+label',
                hovertemplate='<b>%{label}</b><br>Cost: $%{value:.2f}<br>Percentage: %{percent}<extra></extra>'
            )
            
            return fig
            
        except Exception as e:
            return self._create_error_chart(f"Error creating distribution chart: {str(e)}")
    
    def create_utilization_heatmap(self, data):
        """Create utilization heatmap"""
        try:
            if 'usage_hours' not in data.columns:
                return self._create_error_chart("Usage hours data not available")
            
            # Create hourly and daily usage matrix
            data['hour'] = data['timestamp'].dt.hour
            data['day_name'] = data['timestamp'].dt.day_name()
            
            heatmap_data = data.groupby(['day_name', 'hour'])['usage_hours'].mean().reset_index()
            heatmap_pivot = heatmap_data.pivot(index='day_name', columns='hour', values='usage_hours')
            
            # Reorder days
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            heatmap_pivot = heatmap_pivot.reindex(day_order)
            
            fig = px.imshow(
                heatmap_pivot,
                labels=dict(x="Hour of Day", y="Day of Week", color="Usage Hours"),
                title="Resource Utilization Heatmap",
                color_continuous_scale="RdYlBu_r"
            )
            
            fig.update_layout(
                xaxis_title="Hour of Day",
                yaxis_title="Day of Week"
            )
            
            return fig
            
        except Exception as e:
            return self._create_error_chart(f"Error creating heatmap: {str(e)}")
    
    def create_anomaly_visualization(self, data, anomalies):
        """Create anomaly visualization"""
        try:
            # Create scatter plot of costs over time
            fig = px.scatter(
                data,
                x='timestamp',
                y='cost',
                color='resource_type',
                title='Cost Anomalies Over Time',
                labels={'cost': 'Cost ($)', 'timestamp': 'Date'}
            )
            
            # Add anomaly points
            if anomalies:
                anomaly_data = []
                for anomaly in anomalies:
                    anomaly_data.append({
                        'timestamp': anomaly['timestamp'],
                        'cost': anomaly['actual_cost'],
                        'severity': anomaly['severity']
                    })
                
                anomaly_df = pd.DataFrame(anomaly_data)
                
                # Add anomaly markers
                fig.add_trace(go.Scatter(
                    x=anomaly_df['timestamp'],
                    y=anomaly_df['cost'],
                    mode='markers',
                    marker=dict(
                        size=12,
                        color='red',
                        symbol='x',
                        line=dict(width=2, color='darkred')
                    ),
                    name='Anomalies',
                    hovertemplate='<b>Anomaly</b><br>Date: %{x}<br>Cost: $%{y:.2f}<extra></extra>'
                ))
            
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Cost ($)",
                hovermode='closest'
            )
            
            return fig
            
        except Exception as e:
            return self._create_error_chart(f"Error creating anomaly visualization: {str(e)}")
    
    def create_savings_potential_chart(self, recommendations):
        """Create savings potential visualization"""
        try:
            if not recommendations:
                return self._create_error_chart("No recommendations available")
            
            # Prepare data
            savings_data = []
            for rec in recommendations:
                savings_data.append({
                    'title': rec.get('title', 'Unknown'),
                    'category': rec.get('category', 'other'),
                    'estimated_savings': rec.get('estimated_savings', 0),
                    'effort': rec.get('effort', 'Medium'),
                    'priority': rec.get('priority', 'Medium')
                })
            
            savings_df = pd.DataFrame(savings_data)
            
            # Create bar chart
            fig = px.bar(
                savings_df,
                x='estimated_savings',
                y='title',
                color='category',
                title='Potential Cost Savings by Recommendation',
                labels={'estimated_savings': 'Estimated Savings ($)', 'title': 'Recommendation'},
                orientation='h'
            )
            
            fig.update_layout(
                xaxis_title="Estimated Savings ($)",
                yaxis_title="Recommendation",
                height=max(400, len(recommendations) * 40)
            )
            
            return fig
            
        except Exception as e:
            return self._create_error_chart(f"Error creating savings chart: {str(e)}")
    
    def create_cost_breakdown_sunburst(self, data):
        """Create sunburst chart for cost breakdown"""
        try:
            # Create hierarchical data
            hierarchy_data = data.groupby(['resource_type', 'region', 'instance_size'])['cost'].sum().reset_index()
            
            fig = px.sunburst(
                hierarchy_data,
                path=['resource_type', 'region', 'instance_size'],
                values='cost',
                title='Cost Breakdown Hierarchy'
            )
            
            fig.update_traces(
                hovertemplate='<b>%{label}</b><br>Cost: $%{value:.2f}<br>Percentage: %{percentParent}<extra></extra>'
            )
            
            return fig
            
        except Exception as e:
            return self._create_error_chart(f"Error creating sunburst chart: {str(e)}")
    
    def create_efficiency_metrics_chart(self, data):
        """Create efficiency metrics visualization"""
        try:
            if 'usage_hours' not in data.columns:
                return self._create_error_chart("Usage hours data not available")
            
            # Calculate efficiency metrics
            efficiency_data = data.groupby('resource_type').agg({
                'cost': 'sum',
                'usage_hours': 'sum'
            }).reset_index()
            
            efficiency_data['cost_per_hour'] = efficiency_data['cost'] / efficiency_data['usage_hours']
            efficiency_data['efficiency_score'] = 100 / (1 + efficiency_data['cost_per_hour'])
            
            # Create bubble chart
            fig = px.scatter(
                efficiency_data,
                x='usage_hours',
                y='cost',
                size='efficiency_score',
                color='resource_type',
                title='Resource Efficiency Analysis',
                labels={'usage_hours': 'Total Usage Hours', 'cost': 'Total Cost ($)'},
                hover_data=['cost_per_hour']
            )
            
            fig.update_layout(
                xaxis_title="Total Usage Hours",
                yaxis_title="Total Cost ($)"
            )
            
            return fig
            
        except Exception as e:
            return self._create_error_chart(f"Error creating efficiency chart: {str(e)}")
    
    def create_comparison_chart(self, current_costs, projected_costs):
        """Create before/after comparison chart"""
        try:
            comparison_data = pd.DataFrame({
                'Category': ['Current Costs', 'Projected Costs', 'Potential Savings'],
                'Amount': [current_costs, projected_costs, current_costs - projected_costs],
                'Type': ['Current', 'Projected', 'Savings']
            })
            
            fig = px.bar(
                comparison_data,
                x='Category',
                y='Amount',
                color='Type',
                title='Cost Optimization Impact',
                labels={'Amount': 'Cost ($)', 'Category': ''}
            )
            
            fig.update_layout(
                xaxis_title="",
                yaxis_title="Cost ($)"
            )
            
            return fig
            
        except Exception as e:
            return self._create_error_chart(f"Error creating comparison chart: {str(e)}")
    
    def _create_error_chart(self, error_message):
        """Create error chart when visualization fails"""
        fig = go.Figure()
        fig.add_annotation(
            text=error_message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="red")
        )
        fig.update_layout(
            title="Visualization Error",
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, showticklabels=False)
        )
        return fig
