import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from modules.data_processor import DataProcessor
from modules.usage_analyzer import UsageAnalyzer
from modules.cost_anomaly_detector import CostAnomalyDetector
from modules.optimization_engine import OptimizationEngine
from modules.visualization import CostVisualization
from utils.helpers import format_currency, calculate_percentage_change

# Configure page
st.set_page_config(
    page_title="Cloud Cost Optimization Tool",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = DataProcessor()
if 'usage_analyzer' not in st.session_state:
    st.session_state.usage_analyzer = UsageAnalyzer()
if 'anomaly_detector' not in st.session_state:
    st.session_state.anomaly_detector = CostAnomalyDetector()
if 'optimization_engine' not in st.session_state:
    st.session_state.optimization_engine = OptimizationEngine()
if 'visualization' not in st.session_state:
    st.session_state.visualization = CostVisualization()

def generate_sample_data():
    """Generate sample cloud cost data for demonstration, ensuring some resources will trigger recommendations"""
    np.random.seed(42)
    
    # Generate sample data for the last 30 days
    dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='H')
    
    resource_types = ['EC2', 'RDS', 'S3', 'Lambda', 'ELB']
    regions = ['us-east-1', 'us-west-2', 'eu-west-1', 'ap-southeast-1']
    instance_sizes = ['t2.micro', 't2.small', 't2.medium', 'm5.large', 'm5.xlarge']
    
    sample_data = []
    
    # Add some underutilized, idle, and high-cost resources explicitly
    # Underutilized EC2 instance (low usage, high cost)
    for i in range(10):
        sample_data.append({
            'timestamp': np.random.choice(dates),
            'resource_id': f"ec2-underutilized-{i}",
            'resource_type': 'EC2',
            'cost': 200 + np.random.uniform(0, 20),
            'region': 'us-east-1',
            'usage_hours': np.random.uniform(1, 4),  # low usage
            'instance_size': 'm5.xlarge'
        })
    # Idle Lambda (very low usage, low cost)
    for i in range(5):
        sample_data.append({
            'timestamp': np.random.choice(dates),
            'resource_id': f"lambda-idle-{i}",
            'resource_type': 'Lambda',
            'cost': 1 + np.random.uniform(0, 1),
            'region': 'us-west-2',
            'usage_hours': np.random.uniform(0, 0.5),
            'instance_size': 't2.micro'
        })
    # High utilization RDS (to trigger performance/scale rec)
    for i in range(5):
        sample_data.append({
            'timestamp': np.random.choice(dates),
            'resource_id': f"rds-highutil-{i}",
            'resource_type': 'RDS',
            'cost': 300 + np.random.uniform(0, 30),
            'region': 'eu-west-1',
            'usage_hours': np.random.uniform(22, 24),
            'instance_size': 'm5.large'
        })
    # Normal random data
    for i in range(480):  # Generate 480 sample records
        timestamp = np.random.choice(dates)
        resource_type = np.random.choice(resource_types)
        region = np.random.choice(regions)
        instance_size = np.random.choice(instance_sizes)
        
        # Generate realistic cost based on resource type and size
        base_cost = {
            'EC2': 0.1, 'RDS': 0.2, 'S3': 0.023, 'Lambda': 0.0001, 'ELB': 0.025
        }
        
        size_multiplier = {
            't2.micro': 1, 't2.small': 2, 't2.medium': 4, 'm5.large': 8, 'm5.xlarge': 16
        }
        
        cost = base_cost[resource_type] * size_multiplier.get(instance_size, 1) * np.random.uniform(0.5, 2.0)
        usage_hours = np.random.uniform(0, 24)
        
        sample_data.append({
            'timestamp': timestamp,
            'resource_id': f"{resource_type.lower()}-{np.random.randint(1000, 9999)}",
            'resource_type': resource_type,
            'cost': round(cost, 2),
            'region': region,
            'usage_hours': round(usage_hours, 1),
            'instance_size': instance_size
        })
    
    return pd.DataFrame(sample_data)

def map_columns_to_standard_format(data):
    """Map various column names to standard format"""
    # Create a copy of the data
    mapped_data = data.copy()
    
    # Define the mapping logic
    new_columns = {}
    
    for col in mapped_data.columns:
        col_lower = col.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('%', '')
        
        # Map specific columns from the multi-cloud dataset
        if col == 'Service_ID':
            new_columns[col] = 'resource_id'
        elif col == 'Service_Type':
            new_columns[col] = 'resource_type'
        elif col == 'Cloud_Provider':
            new_columns[col] = 'region'
        elif col == 'Edge_Node_ID':
            new_columns[col] = 'instance_size'
        # Keep other columns as-is but clean the names
        else:
            new_columns[col] = col_lower
    
    # Apply the mapping
    mapped_data = mapped_data.rename(columns=new_columns)
    
    return mapped_data

def estimate_cost_from_utilization(data):
    """Estimate cost based on utilization metrics when cost column is missing"""
    if 'cost' in data.columns:
        return data
    
    # Base costs per hour for different service types (in USD)
    base_costs = {
        'database': 0.20,
        'ai model': 0.50,
        'network': 0.10,
        'storage': 0.023,
        'compute': 0.15,
        'application': 0.12,
        'web server': 0.08,
        'load balancer': 0.025
    }
    
    # Calculate estimated cost based on available metrics
    estimated_costs = []
    
    for _, row in data.iterrows():
        service_type = str(row.get('resource_type', 'compute')).lower()
        
        # Get base cost for service type
        base_cost = base_costs.get(service_type, 0.15)
        
        # Calculate utilization factor
        utilization_factor = 1.0
        if 'cpu_utilization_' in row:
            cpu_util = float(row['cpu_utilization_']) / 100.0
            utilization_factor *= (0.5 + cpu_util)
        
        if 'memory_usage_mb' in row:
            memory_gb = float(row['memory_usage_mb']) / 1024  # Convert MB to GB
            utilization_factor *= (0.8 + memory_gb * 0.001)
        
        if 'storage_usage_gb' in row:
            storage_gb = float(row['storage_usage_gb'])
            utilization_factor *= (0.9 + storage_gb * 0.0001)
        
        # Calculate estimated hourly cost
        estimated_cost = base_cost * utilization_factor
        estimated_costs.append(round(estimated_cost, 4))
    
    data['cost'] = estimated_costs
    return data

def create_synthetic_timestamp(data):
    """Create synthetic timestamps if missing"""
    if 'timestamp' in data.columns:
        return data
    
    # Generate timestamps for the last 30 days
    start_date = datetime.now() - timedelta(days=30)
    timestamps = []
    
    for i in range(len(data)):
        # Create timestamps distributed over the last 30 days
        random_offset = np.random.uniform(0, 30 * 24 * 60 * 60)  # 30 days in seconds
        timestamp = start_date + timedelta(seconds=random_offset)
        timestamps.append(timestamp)
    
    data['timestamp'] = timestamps
    return data

def main():
    st.title("☁️ Cloud Cost Optimization Tool")
    st.markdown("Analyze cloud resource usage patterns and get AI-driven cost reduction recommendations")
    
    # Sidebar for data input and configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Data source selection
        data_source = st.selectbox(
            "Data Source",
            ["Upload CSV", "API Integration", "Database Connection"],
            help="Select how you want to input your cloud usage data"
        )
        
        # Cloud provider selection
        cloud_provider = st.selectbox(
            "Cloud Provider",
            ["AWS", "Azure", "Google Cloud", "Multi-Cloud"],
            help="Select your cloud provider for optimized analysis"
        )
        
        # Analysis period
        analysis_period = st.selectbox(
            "Analysis Period",
            ["Last 7 days", "Last 30 days", "Last 90 days", "Custom"],
            help="Select the time period for analysis"
        )
        
        if analysis_period == "Custom":
            start_date = st.date_input("Start Date")
            end_date = st.date_input("End Date")
        
        # Cost threshold for anomaly detection
        anomaly_threshold = st.slider(
            "Anomaly Detection Sensitivity",
            min_value=1.0,
            max_value=5.0,
            value=2.0,
            step=0.1,
            help="Higher values = less sensitive to anomalies"
        )
    
    # Main content area
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader(
            "Upload your cloud usage data (CSV format)",
            type=['csv'],
            help="Upload a CSV file containing your cloud resource usage data"
        )
        
        if uploaded_file is not None:
            try:
                # Process uploaded data
                raw_data = pd.read_csv(uploaded_file)
                
                # Show preview of uploaded data
                st.subheader("Data Preview")
                st.write("**Uploaded columns:**", list(raw_data.columns))
                st.dataframe(raw_data.head())
                
                # Check for required columns
                required_cols = ['timestamp', 'resource_id', 'resource_type', 'cost', 'region']
                missing_cols = [col for col in required_cols if col not in raw_data.columns]
                
                if missing_cols:
                    st.warning(f"Missing required columns: {missing_cols}")
                    st.info("🔄 Attempting to automatically map your columns...")
                    
                    # Try to map columns automatically
                    mapped_data = map_columns_to_standard_format(raw_data)
                    
                    # Check if we still have missing columns after mapping
                    mapped_missing_cols = [col for col in required_cols if col not in mapped_data.columns]
                    
                    if mapped_missing_cols:
                        st.warning(f"Still missing after auto-mapping: {mapped_missing_cols}")
                        
                        # Try to create synthetic data for missing columns
                        if 'timestamp' in mapped_missing_cols:
                            st.info("🕐 Creating synthetic timestamps...")
                            mapped_data = create_synthetic_timestamp(mapped_data)
                        
                        if 'cost' in mapped_missing_cols:
                            st.info("💰 Estimating costs from utilization metrics...")
                            mapped_data = estimate_cost_from_utilization(mapped_data)
                        
                        if 'region' in mapped_missing_cols and 'Cloud_Provider' in raw_data.columns:
                            st.info("🌍 Using Cloud_Provider as region...")
                            mapped_data['region'] = raw_data['Cloud_Provider']
                        
                        if 'usage_hours' in mapped_missing_cols:
                            st.info("⏰ Estimating usage hours from utilization metrics...")
                            # Estimate usage hours based on utilization (assuming 24-hour period)
                            if 'cpu_utilization_' in mapped_data.columns:
                                mapped_data['usage_hours'] = (mapped_data['cpu_utilization_'] / 100.0) * 24
                            else:
                                mapped_data['usage_hours'] = 12  # Default to 12 hours
                        
                        # Final check
                        final_missing_cols = [col for col in required_cols if col not in mapped_data.columns]
                        
                        if final_missing_cols:
                            st.error(f"Cannot proceed. Still missing: {final_missing_cols}")
                            
                            # Show sample data format
                            st.subheader("Expected Data Format")
                            sample_data = generate_sample_data()
                            st.dataframe(sample_data)
                            
                            # Provide download link for sample data
                            csv_sample = sample_data.to_csv(index=False)
                            st.download_button(
                                label="Download Sample Data",
                                data=csv_sample,
                                file_name="sample_cloud_data.csv",
                                mime="text/csv"
                            )
                        else:
                            st.success("✅ Successfully mapped all columns!")
                            st.subheader("Column Mappings Applied")
                            
                            # Show the mappings
                            mapping_info = []
                            
                            # Show what columns were mapped
                            if 'Service_ID' in raw_data.columns:
                                mapping_info.append("🔄 Service_ID → resource_id")
                            if 'Service_Type' in raw_data.columns:
                                mapping_info.append("🔄 Service_Type → resource_type")
                            if 'Cloud_Provider' in raw_data.columns:
                                mapping_info.append("🔄 Cloud_Provider → region")
                            
                            if 'timestamp' not in raw_data.columns and 'timestamp' in mapped_data.columns:
                                mapping_info.append("🕐 timestamp → Created synthetic timestamps")
                            if 'cost' not in raw_data.columns and 'cost' in mapped_data.columns:
                                mapping_info.append("💰 cost → Estimated from utilization metrics")
                            
                            st.info("\n".join(mapping_info))
                            
                            # Show mapped data preview
                            st.subheader("Mapped Data Preview")
                            st.write("**Final columns:**", list(mapped_data.columns))
                            st.dataframe(mapped_data.head())
                            
                            # Process the mapped data
                            try:
                                processed_data = st.session_state.data_processor.process_data(mapped_data)
                                
                                if processed_data is not None and not processed_data.empty:
                                    display_analysis(processed_data, cloud_provider, anomaly_threshold)
                                else:
                                    st.error("Unable to process the mapped data.")
                            except Exception as e:
                                st.error(f"Error during data processing: {str(e)}")
                                import traceback
                                st.code(traceback.format_exc())
                    else:
                        st.success("✅ Successfully mapped all columns!")
                        processed_data = st.session_state.data_processor.process_data(mapped_data)
                        
                        if processed_data is not None and not processed_data.empty:
                            display_analysis(processed_data, cloud_provider, anomaly_threshold)
                        else:
                            st.error("Unable to process the mapped data.")
                else:
                    processed_data = st.session_state.data_processor.process_data(raw_data)
                    
                    if processed_data is not None and not processed_data.empty:
                        display_analysis(processed_data, cloud_provider, anomaly_threshold)
                    else:
                        st.error("Unable to process the uploaded data. Please check the file format.")
                        
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                st.info("Please ensure your file is a valid CSV format.")
                
    else:
        # Show configuration instructions for API/Database integration
        st.info(f"Please configure your {data_source.lower()} settings in the sidebar and ensure your credentials are properly set.")
        
        # Show sample data structure
        st.subheader("Expected Data Structure")
        st.markdown("""
        Your cloud usage data should include the following columns:
        - **timestamp**: Date and time of the usage record
        - **resource_id**: Unique identifier for the cloud resource
        - **resource_type**: Type of resource (e.g., EC2, RDS, S3, etc.)
        - **usage_hours**: Hours of usage for the time period
        - **cost**: Cost associated with the usage
        - **region**: Cloud region where the resource is located
        - **instance_size**: Size/type of the instance (if applicable)
        """)
        
        # Show empty state with instructions
        st.warning("No data available. Please upload a CSV file or configure your data source to begin analysis.")
        
        # Add sample data option
        st.subheader("Try with Sample Data")
        st.info("Want to test the tool? Click below to generate sample cloud cost data.")
        
        if st.button("Generate Sample Data", type="primary"):
            sample_data = generate_sample_data()
            processed_data = st.session_state.data_processor.process_data(sample_data)
            
            if processed_data is not None and not processed_data.empty:
                st.success("Sample data generated successfully!")
                display_analysis(processed_data, cloud_provider, anomaly_threshold)
            else:
                st.error("Failed to generate sample data.")

def display_analysis(data, cloud_provider, anomaly_threshold):
    """Display the complete cost analysis dashboard"""
    
    # Analyze usage patterns
    usage_patterns = st.session_state.usage_analyzer.analyze_patterns(data)
    
    # Detect cost anomalies
    anomalies = st.session_state.anomaly_detector.detect_anomalies(data, threshold=anomaly_threshold)
    
    # Generate optimization recommendations
    recommendations = st.session_state.optimization_engine.generate_recommendations(
        data, usage_patterns, cloud_provider
    )
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_cost = data['cost'].sum()
        st.metric("Total Cost", format_currency(total_cost))
    
    with col2:
        potential_savings = sum([rec['estimated_savings'] for rec in recommendations])
        st.metric("Potential Savings", format_currency(potential_savings))
    
    with col3:
        savings_percentage = (potential_savings / total_cost) * 100 if total_cost > 0 else 0
        st.metric("Savings Percentage", f"{savings_percentage:.1f}%")
    
    with col4:
        num_anomalies = len(anomalies)
        st.metric("Cost Anomalies", num_anomalies)
    
    # Create tabs for different analysis views
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Cost Overview", "🔍 Usage Patterns", "⚠️ Anomalies", "💡 Recommendations"])
    
    with tab1:
        display_cost_overview(data)
    
    with tab2:
        display_usage_patterns(usage_patterns)
    
    with tab3:
        display_anomalies(anomalies)
    
    with tab4:
        display_recommendations(recommendations)

def display_cost_overview(data):
    """Display cost overview charts and metrics"""
    
    st.subheader("Cost Trends")
    
    # Daily cost trend
    daily_costs = data.groupby(data['timestamp'].dt.date)['cost'].sum().reset_index()
    daily_costs.columns = ['date', 'total_cost']
    
    fig_trend = px.line(
        daily_costs, 
        x='date', 
        y='total_cost',
        title="Daily Cost Trend",
        labels={'total_cost': 'Cost ($)', 'date': 'Date'}
    )
    st.plotly_chart(fig_trend, use_container_width=True)
    
    # Cost by resource type
    col1, col2 = st.columns(2)
    
    with col1:
        resource_costs = data.groupby('resource_type')['cost'].sum().reset_index()
        fig_pie = px.pie(
            resource_costs, 
            values='cost', 
            names='resource_type',
            title="Cost Distribution by Resource Type"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        region_costs = data.groupby('region')['cost'].sum().reset_index()
        fig_bar = px.bar(
            region_costs, 
            x='region', 
            y='cost',
            title="Cost by Region"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

def display_usage_patterns(usage_patterns):
    """Display usage pattern analysis"""
    
    st.subheader("Usage Patterns Analysis")
    
    if not usage_patterns:
        st.info("No usage patterns detected. This could indicate insufficient data or very irregular usage.")
        return
    
    # Handle case where usage_patterns is a list instead of dict
    if isinstance(usage_patterns, list):
        patterns_list = usage_patterns
        hourly_pattern = None
        
        # Look for hourly pattern in the list
        for pattern in patterns_list:
            if isinstance(pattern, dict) and pattern.get('type') == 'hourly_pattern':
                hourly_pattern = pattern.get('hourly_pattern', {})
                break
    else:
        patterns_list = []
        hourly_pattern = usage_patterns.get('hourly_pattern', {})
    
    # Display pattern summary
    st.markdown("### Key Insights")
    for pattern in patterns_list:
        if isinstance(pattern, dict):
            pattern_type = pattern.get('type', 'unknown')
            if pattern_type == 'peak_hours':
                st.info(f"**Peak Usage Hours**: {pattern.get('description', 'No description')}")
            elif pattern_type == 'underutilized':
                st.warning(f"**Underutilized Resources**: {pattern.get('description', 'No description')}")
            elif pattern_type == 'seasonal':
                st.info(f"**Seasonal Pattern**: {pattern.get('description', 'No description')}")
            elif pattern_type == 'low_hours':
                st.info(f"**Low Usage Hours**: {pattern.get('description', 'No description')}")
            elif pattern_type == 'weekday_pattern':
                st.info(f"**Weekday Pattern**: {pattern.get('description', 'No description')}")
    
    # Hourly usage pattern
    st.markdown("### Hourly Usage Pattern")
    if hourly_pattern:
        hours = list(hourly_pattern.keys())
        usage_values = list(hourly_pattern.values())
        
        fig_hourly = go.Figure()
        fig_hourly.add_trace(go.Scatter(
            x=hours,
            y=usage_values,
            mode='lines+markers',
            name='Usage',
            line=dict(color='blue', width=2)
        ))
        fig_hourly.update_layout(
            title="Average Usage by Hour of Day",
            xaxis_title="Hour of Day",
            yaxis_title="Usage Level"
        )
        st.plotly_chart(fig_hourly, use_container_width=True)
    else:
        st.info("No hourly usage pattern data available.")

def display_anomalies(anomalies):
    """Display detected cost anomalies"""
    
    st.subheader("Cost Anomaly Detection")
    
    if not anomalies:
        st.success("No significant cost anomalies detected in the current period.")
        return
    
    st.warning(f"Detected {len(anomalies)} cost anomalies that require attention.")
    
    # Display anomalies in a table
    anomaly_data = []
    for anomaly in anomalies:
        anomaly_data.append({
            'Date': anomaly.get('timestamp', 'Unknown'),
            'Resource ID': anomaly.get('resource_id', 'Unknown'),
            'Resource Type': anomaly.get('resource_type', 'Unknown'),
            'Expected Cost': format_currency(anomaly.get('expected_cost', 0)),
            'Actual Cost': format_currency(anomaly.get('actual_cost', 0)),
            'Deviation': f"{anomaly.get('deviation_percentage', 0):.1f}%",
            'Severity': anomaly.get('severity', 'Unknown')
        })
    
    anomaly_df = pd.DataFrame(anomaly_data)
    st.dataframe(anomaly_df, use_container_width=True)
    
    # Severity distribution
    severity_counts = pd.Series([a.get('severity', 'Unknown') for a in anomalies]).value_counts()
    fig_severity = px.bar(
        x=severity_counts.index,
        y=severity_counts.values,
        title="Anomaly Severity Distribution",
        labels={'x': 'Severity', 'y': 'Count'}
    )
    st.plotly_chart(fig_severity, use_container_width=True)

def display_recommendations(recommendations):
    """Display optimization recommendations"""
    
    st.subheader("Cost Optimization Recommendations")
    
    if not recommendations:
        st.info("No optimization recommendations available at this time.")
        return
    
    # Sort recommendations by potential savings
    sorted_recommendations = sorted(recommendations, key=lambda x: x.get('estimated_savings', 0), reverse=True)
    
    for i, rec in enumerate(sorted_recommendations):
        with st.expander(f"💡 {rec.get('title', 'Optimization Recommendation')} - Save {format_currency(rec.get('estimated_savings', 0))}"):
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**Description**: {rec.get('description', 'No description available')}")
                st.markdown(f"**Implementation**: {rec.get('implementation', 'No implementation details available')}")
                
                if rec.get('risks'):
                    st.markdown(f"**Risks**: {rec.get('risks')}")
                
                if rec.get('timeline'):
                    st.markdown(f"**Timeline**: {rec.get('timeline')}")
            
            with col2:
                st.metric("Estimated Savings", format_currency(rec.get('estimated_savings', 0)))
                st.metric("Implementation Effort", rec.get('effort', 'Unknown'))
                st.metric("Priority", rec.get('priority', 'Unknown'))
            
            # Implementation button
            if st.button(f"Mark as Implemented", key=f"implement_{i}"):
                st.success("Recommendation marked as implemented!")
                st.rerun()

if __name__ == "__main__":
    main()
