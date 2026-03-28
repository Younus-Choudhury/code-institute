"""
MBA Strategic Analysis - 3 Executive Graphs
Uses only allowed libraries: numpy, pandas, matplotlib, seaborn, plotly
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import warnings
import sys
warnings.filterwarnings('ignore')

print("🎓 MBA STRATEGIC ANALYSIS - KAGGLE RETAIL DATASET")
print("="*60)

# =====================================
# DATA LOADING AND PREPARATION
# =====================================

def load_retail_data():
    """Load and prepare the Kaggle retail dataset from three files"""
    print("🔄 Loading Kaggle retail datasets...")
    
    try:
        # Load the three datasets
        features = pd.read_csv('Features data set.csv')
        sales = pd.read_csv('sales data set.csv')
        stores = pd.read_csv('stores data set.csv')
        
        print("✅ Datasets loaded successfully:")
        print(f"   - Features: {features.shape[0]:,} records")
        print(f"   - Sales: {sales.shape[0]:,} records")
        print(f"   - Stores: {stores.shape[0]:,} records")
        
        # Clean column names (remove extra spaces and standardize)
        features.columns = features.columns.str.strip()
        sales.columns = sales.columns.str.strip()
        stores.columns = stores.columns.str.strip()
        
        # Rename columns to match expected names
        sales = sales.rename(columns={'Weekly_Sales': 'Weekly_Sales'})
        features = features.rename(columns={
            'IsHoliday': 'Holiday_Flag',
            'Fuel_Price': 'Fuel_Price',
            'CPI': 'CPI',
            'Unemployment': 'Unemployment'
        })
        
        # Merge datasets
        # First merge sales and features on Store and Date
        df = pd.merge(sales, features, on=['Store', 'Date'], how='left')
        
        # Then merge with stores data
        df = pd.merge(df, stores, on=['Store'], how='left')
        
        # Ensure we have required columns
        required_columns = ['Store', 'Date', 'Weekly_Sales', 'Holiday_Flag', 
                           'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 
                           'Type', 'Size']
        
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
                
    except Exception as e:
        print(f"❌ ERROR: {e}")
        print("💡 Please ensure these files exist in the current directory:")
        print("   - 'Features data set.csv'")
        print("   - 'sales data set.csv'")
        print("   - 'stores data set.csv'")
        sys.exit(1)
    
    # Clean and prepare data
    df = prepare_retail_data(df)
    print(f"✅ Dataset prepared: {df.shape[0]:,} records, {df.shape[1]} columns")
    return df

def prepare_retail_data(df):
    """Clean and prepare the retail dataset"""
    # Ensure Date column is datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    
    # Handle missing values
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        df[col] = df[col].fillna(df[col].median())
    
    # Handle categorical/store type data
    if 'Type' in df.columns:
        df['Type'] = df['Type'].fillna('C')  # Most common type
        # Ensure consistent casing
        df['Type'] = df['Type'].str.upper().str.strip()
    
    # Create additional time features
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.isocalendar().week
    df['Quarter'] = df['Date'].dt.quarter
    
    # Ensure size is consistent
    if 'Size' in df.columns:
        # Remove any negative values
        df['Size'] = df['Size'].clip(lower=10000)
    
    # Clean weekly sales data
    if 'Weekly_Sales' in df.columns:
        # Remove negative sales
        df['Weekly_Sales'] = df['Weekly_Sales'].clip(lower=0)
        # Remove extreme outliers (top 0.1%)
        q99 = df['Weekly_Sales'].quantile(0.999)
        df['Weekly_Sales'] = df['Weekly_Sales'].clip(upper=q99)
    
    # Clean economic indicators
    economic_cols = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
    for col in economic_cols:
        if col in df.columns:
            # Remove extreme values
            q01 = df[col].quantile(0.01)
            q99 = df[col].quantile(0.99)
            df[col] = df[col].clip(lower=q01, upper=q99)
    
    # Convert data types to reduce memory usage
    df['Store'] = df['Store'].astype('int32')
    df['Year'] = df['Year'].astype('int16')
    df['Month'] = df['Month'].astype('int8')
    df['Week'] = df['Week'].astype('int8')
    df['Quarter'] = df['Quarter'].astype('int8')
    
    return df

# =====================================
# GRAPH 1: HOLIDAY IMPACT ANALYSIS
# =====================================

def create_holiday_impact_chart(df):
    """Create holiday impact analysis chart using matplotlib and seaborn"""
    print("\n📈 GRAPH 1: Holiday Impact on Sales")
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create subplot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Holiday Impact Analysis on Retail Sales', fontsize=16, fontweight='bold')
    
    # 1. Average Sales by Holiday Flag
    holiday_avg = df.groupby('Holiday_Flag')['Weekly_Sales'].mean()
    colors = ['#FF6B6B', '#4ECDC4']
    bars1 = ax1.bar(['Non-Holiday', 'Holiday'], holiday_avg, color=colors, alpha=0.8)
    ax1.set_title('Average Weekly Sales: Holiday vs Non-Holiday', fontsize=12, pad=15)
    ax1.set_ylabel('Average Sales ($)', fontsize=10)
    
    # Format y-axis
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'${height/1e6:.1f}M', ha='center', va='bottom', fontweight='bold')
    
    # 2. Sales by Store Type and Holiday
    pivot_data = df.pivot_table(values='Weekly_Sales', index='Type', columns='Holiday_Flag', aggfunc='mean')
    
    x = np.arange(len(pivot_data.index))
    width = 0.35
    
    bars2_1 = ax2.bar(x - width/2, pivot_data[0], width, label='Non-Holiday', color=colors[0], alpha=0.8)
    bars2_2 = ax2.bar(x + width/2, pivot_data[1], width, label='Holiday', color=colors[1], alpha=0.8)
    
    ax2.set_title('Average Sales by Store Type', fontsize=12, pad=15)
    ax2.set_xlabel('Store Type', fontsize=10)
    ax2.set_ylabel('Average Sales ($)', fontsize=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(pivot_data.index)
    ax2.legend()
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
    
    # 3. Monthly Holiday Sales Distribution
    monthly_holiday = df[df['Holiday_Flag'] == 1].groupby('Month')['Weekly_Sales'].sum()
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    months = list(range(1, 13))
    sales_values = [monthly_holiday.get(i, 0) for i in months]
    
    bars3 = ax3.bar(months, sales_values, color='#FF6B6B', alpha=0.8)
    ax3.set_title('Holiday Sales by Month', fontsize=12, pad=15)
    ax3.set_xlabel('Month', fontsize=10)
    ax3.set_ylabel('Holiday Sales ($)', fontsize=10)
    ax3.set_xticks(months)
    ax3.set_xticklabels(month_names, rotation=45)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.0f}M'))
    
    # 4. Holiday Sales Lift Distribution
    store_holiday_avg = df.groupby(['Store', 'Holiday_Flag'])['Weekly_Sales'].mean().unstack()
    
    # Calculate lift only for stores that have both holiday and non-holiday data
    valid_stores = store_holiday_avg.dropna()
    if len(valid_stores) > 0:
        lift_values = ((valid_stores[1] - valid_stores[0]) / valid_stores[0] * 100)
        
        ax4.hist(lift_values, bins=20, color='#4ECDC4', alpha=0.8, edgecolor='black')
        ax4.set_title('Holiday Sales Lift Distribution', fontsize=12, pad=15)
        ax4.set_xlabel('Sales Lift (%)', fontsize=10)
        ax4.set_ylabel('Number of Stores', fontsize=10)
        
        # Add mean line
        mean_lift = lift_values.mean()
        ax4.axvline(mean_lift, color='red', linestyle='--', linewidth=2,
                    label=f'Avg Lift: {mean_lift:.1f}%')
        ax4.legend()
    
    plt.tight_layout()
    plt.savefig('holiday_impact_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Saved holiday_impact_analysis.png")
    
    return df.groupby('Holiday_Flag')['Weekly_Sales'].agg(['mean', 'count'])

# =====================================
# GRAPH 2: ECONOMIC FACTORS CORRELATION
# =====================================

def create_economic_correlation_chart(df):
    """Create economic factors correlation analysis using seaborn and matplotlib"""
    print("\n📈 GRAPH 2: Economic Factors Impact Analysis")
    
    # Prepare economic data
    economic_cols = ['Weekly_Sales', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
    economic_data = df[economic_cols].copy()
    
    # Create correlation matrix
    correlation_matrix = economic_data.corr()
    
    # Create subplot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Economic Factors Impact on Retail Sales', fontsize=16, fontweight='bold')
    
    # 1. Correlation Heatmap
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdYlBu_r', center=0,
                square=True, ax=ax1, cbar_kws={'shrink': 0.8}, fmt='.3f')
    ax1.set_title('Economic Factors Correlation Matrix', fontsize=12, pad=15)
    
    # 2. Sales vs Fuel Price Scatter Plot
    sample_df = df.sample(n=min(1000, len(df)), random_state=42)
    
    scatter = ax2.scatter(sample_df['Fuel_Price'], sample_df['Weekly_Sales']/1e6, 
                         c=sample_df['Temperature'], cmap='coolwarm', alpha=0.6, s=30)
    ax2.set_xlabel('Fuel Price ($)', fontsize=10)
    ax2.set_ylabel('Weekly Sales ($ Millions)', fontsize=10)
    ax2.set_title('Sales vs Fuel Price (colored by Temperature)', fontsize=12, pad=15)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Temperature (°F)', fontsize=9)
    
    # Add trend line
    z = np.polyfit(sample_df['Fuel_Price'], sample_df['Weekly_Sales']/1e6, 1)
    p = np.poly1d(z)
    fuel_sorted = np.sort(sample_df['Fuel_Price'])
    ax2.plot(fuel_sorted, p(fuel_sorted), "r--", alpha=0.8, linewidth=2)
    
    # 3. Sales vs Unemployment Rate
    # Create unemployment bins
    df['Unemployment_Bins'] = pd.cut(df['Unemployment'], bins=5, precision=1)
    unemployment_sales = df.groupby('Unemployment_Bins')['Weekly_Sales'].mean()
    
    # Plot bar chart
    x_pos = range(len(unemployment_sales))
    bars3 = ax3.bar(x_pos, unemployment_sales, color='#FF6B6B', alpha=0.8)
    ax3.set_title('Average Sales by Unemployment Rate', fontsize=12, pad=15)
    ax3.set_xlabel('Unemployment Rate Range (%)', fontsize=10)
    ax3.set_ylabel('Average Sales ($)', fontsize=10)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([str(x) for x in unemployment_sales.index], rotation=45, ha='right')
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
    
    # 4. Economic Impact Score
    # Normalize economic indicators
    df_norm = df.copy()
    
    # Min-Max normalization
    for col in ['Fuel_Price', 'Unemployment', 'CPI']:
        col_min = df_norm[col].min()
        col_max = df_norm[col].max()
        df_norm[f'{col}_norm'] = (df_norm[col] - col_min) / (col_max - col_min)
    
    # Economic stress score
    df_norm['Economic_Stress'] = (df_norm['Fuel_Price_norm'] + df_norm['Unemployment_norm']) / 2
    
    # Create stress level categories
    df_norm['Stress_Level'] = pd.cut(df_norm['Economic_Stress'], 
                                    bins=4, labels=['Low', 'Medium', 'High', 'Very High'])
    
    stress_sales = df_norm.groupby('Stress_Level')['Weekly_Sales'].mean()
    
    # Plot stress level impact
    colors_stress = ['#4ECDC4', '#45B7D1', '#FF6B6B', '#E74C3C']
    bars4 = ax4.bar(range(len(stress_sales)), stress_sales.values, 
                    color=colors_stress, alpha=0.8)
    ax4.set_title('Sales Performance by Economic Stress Level', fontsize=12, pad=15)
    ax4.set_xlabel('Economic Stress Level', fontsize=10)
    ax4.set_ylabel('Average Sales ($)', fontsize=10)
    ax4.set_xticks(range(len(stress_sales)))
    ax4.set_xticklabels(stress_sales.index)
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
    
    # Add value labels
    for i, bar in enumerate(bars4):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'${height/1e6:.1f}M', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('economic_factors_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Saved economic_factors_analysis.png")
    
    return correlation_matrix

# =====================================
# GRAPH 3: INTERACTIVE STORE PERFORMANCE DASHBOARD
# =====================================

def create_interactive_dashboard(df):
    """Create interactive store performance dashboard using plotly"""
    print("\n📈 GRAPH 3: Interactive Store Performance Dashboard")
    
    # Prepare store summary data
    store_metrics = []
    
    for store in df['Store'].unique():
        store_data = df[df['Store'] == store]
        
        metrics = {
            'Store': store,
            'Type': store_data['Type'].iloc[0],
            'Size': store_data['Size'].iloc[0],
            'Total_Sales': store_data['Weekly_Sales'].sum(),
            'Avg_Weekly_Sales': store_data['Weekly_Sales'].mean(),
            'Weeks_Count': len(store_data),
            'Holiday_Weeks': store_data['Holiday_Flag'].sum(),
            'Avg_Temperature': store_data['Temperature'].mean(),
            'Avg_Fuel_Price': store_data['Fuel_Price'].mean(),
            'Avg_Unemployment': store_data['Unemployment'].mean()
        }
        
        # Calculate derived metrics
        metrics['Sales_Per_SqFt'] = metrics['Total_Sales'] / metrics['Size']
        metrics['Holiday_Performance'] = (metrics['Holiday_Weeks'] / metrics['Weeks_Count']) * 100
        
        store_metrics.append(metrics)
    
    store_summary = pd.DataFrame(store_metrics)
    
    # Create interactive plotly figure
    fig = go.Figure()
    
    # Color mapping for store types
    colors = {'A': '#FF6B6B', 'B': '#4ECDC4', 'C': '#45B7D1'}
    
    # Add traces for each store type
    for store_type in ['A', 'B', 'C']:
        store_data = store_summary[store_summary['Type'] == store_type]
        
        if len(store_data) > 0:
            # Create detailed hover text
            hover_text = []
            for _, row in store_data.iterrows():
                text = (f"<b>Store {int(row['Store'])}</b><br>"
                       f"<b>Type:</b> {row['Type']}<br>"
                       f"<b>Size:</b> {row['Size']:,.0f} sq ft<br>"
                       f"<b>Total Sales:</b> ${row['Total_Sales']/1e6:.2f}M<br>"
                       f"<b>Avg Weekly Sales:</b> ${row['Avg_Weekly_Sales']/1e3:.0f}K<br>"
                       f"<b>Sales per Sq Ft:</b> ${row['Sales_Per_SqFt']:.2f}<br>"
                       f"<b>Holiday Weeks:</b> {row['Holiday_Weeks']:.0f}<br>"
                       f"<b>Avg Temperature:</b> {row['Avg_Temperature']:.1f}°F<br>"
                       f"<b>Avg Fuel Price:</b> ${row['Avg_Fuel_Price']:.2f}<br>"
                       f"<b>Avg Unemployment:</b> {row['Avg_Unemployment']:.1f}%")
                hover_text.append(text)
            
            # Calculate bubble sizes (normalize sales per sq ft)
            min_efficiency = store_summary['Sales_Per_SqFt'].min()
            max_efficiency = store_summary['Sales_Per_SqFt'].max()
            normalized_sizes = ((store_data['Sales_Per_SqFt'] - min_efficiency) / 
                              (max_efficiency - min_efficiency)) * 40 + 10
            
            fig.add_trace(go.Scatter(
                x=store_data['Size'] / 1000,  # Size in thousands
                y=store_data['Total_Sales'] / 1e6,  # Sales in millions
                mode='markers',
                name=f'Type {store_type} Stores',
                marker=dict(
                    size=normalized_sizes,
                    color=colors[store_type],
                    opacity=0.7,
                    line=dict(width=1, color='DarkSlateGrey'),
                    sizemode='diameter'
                ),
                text=hover_text,
                hovertemplate='%{text}<extra></extra>'
            ))
    
    # Update layout with professional styling
    fig.update_layout(
        title={
            'text': 'Interactive Store Performance Analysis<br><sub>Store Size vs Total Sales (Bubble size = Sales per Sq Ft)</sub>',
            'x': 0.5,
            'font': {'size': 20}
        },
        xaxis=dict(
            title='Store Size (Thousands of Sq Ft)',
            titlefont={'size': 14},
            tickfont={'size': 12},
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray'
        ),
        yaxis=dict(
            title='Total Sales ($ Millions)',
            titlefont={'size': 14},
            tickfont={'size': 12},
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray'
        ),
        template='plotly_white',
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        ),
        legend=dict(
            title='Store Types',
            title_font={'size': 14},
            font={'size': 12},
            x=0.02,
            y=0.98,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='LightGray',
            borderwidth=1
        ),
        height=600,
        width=1000,
        margin=dict(l=80, r=80, t=100, b=80)
    )
    
    # Add annotations
    fig.add_annotation(
        x=0.02, y=0.02,
        xref="paper", yref="paper",
        text="💡 Larger bubbles indicate higher sales efficiency (Sales per Sq Ft)",
        showarrow=False,
        font=dict(size=11, color="gray"),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="LightGray",
        borderwidth=1
    )
    
    # Show the interactive plot
    print("🖥️ Opening interactive dashboard in your browser...")
    fig.show()
    
    return store_summary

# =====================================
# EXECUTIVE SUMMARY
# =====================================

def generate_executive_summary(df, holiday_analysis, correlation_matrix, store_summary):
    """Generate comprehensive executive summary"""
    print("\n📊 EXECUTIVE SUMMARY - KAGGLE RETAIL DATASET ANALYSIS")
    print("="*65)
    
    # Key Performance Metrics
    total_sales = df['Weekly_Sales'].sum() / 1e9
    total_stores = df['Store'].nunique()
    total_weeks = df['Date'].nunique()
    avg_weekly_sales = df['Weekly_Sales'].mean() / 1e6
    
    print(f"📈 BUSINESS PERFORMANCE:")
    print(f"   💰 Total Sales: ${total_sales:.2f}B across {total_weeks} weeks")
    print(f"   🏪 Store Count: {total_stores} stores analyzed")
    print(f"   📊 Average Weekly Sales: ${avg_weekly_sales:.2f}M per store")
    
    # Holiday Impact Analysis
    holiday_lift = ((df[df['Holiday_Flag']==1]['Weekly_Sales'].mean() - 
                    df[df['Holiday_Flag']==0]['Weekly_Sales'].mean()) / 
                   df[df['Holiday_Flag']==0]['Weekly_Sales'].mean() * 100)
    
    print(f"\n🎉 HOLIDAY IMPACT:")
    print(f"   📈 Holiday Sales Lift: {holiday_lift:.1f}%")
    
    # Store Performance
    best_store = store_summary.loc[store_summary['Total_Sales'].idxmax()]
    best_efficiency = store_summary.loc[store_summary['Sales_Per_SqFt'].idxmax()]
    
    print(f"\n🏪 STORE PERFORMANCE:")
    print(f"   🥇 Top Sales Store: #{int(best_store['Store'])} (Type {best_store['Type']})")
    print(f"      - Total Sales: ${best_store['Total_Sales']/1e6:.2f}M")
    print(f"      - Store Size: {best_store['Size']:,.0f} sq ft")
    print(f"   ⚡ Most Efficient Store: #{int(best_efficiency['Store'])} (${best_efficiency['Sales_Per_SqFt']:.2f}/sq ft)")
    
    # Economic Insights
    fuel_correlation = correlation_matrix.loc['Weekly_Sales', 'Fuel_Price']
    unemployment_correlation = correlation_matrix.loc['Weekly_Sales', 'Unemployment']
    
    print(f"\n💼 ECONOMIC FACTORS:")
    print(f"   ⛽ Fuel Price Impact: {fuel_correlation:.3f} correlation")
    print(f"   💼 Unemployment Impact: {unemployment_correlation:.3f} correlation")
    
    # Strategic Recommendations
    print(f"\n🎯 STRATEGIC RECOMMENDATIONS:")
    print(f"   1. 📅 Maximize holiday marketing - {holiday_lift:.1f}% sales lift opportunity")
    print(f"   2. 🏆 Replicate top performing store strategies")
    print(f"   3. 📍 Focus on high-efficiency store formats")
    print(f"   4. ⛽ Monitor fuel price trends for demand forecasting")
    print(f"   5. 🎯 Optimize inventory for economic conditions")

# =====================================
# MAIN EXECUTION
# =====================================

def main():
    """Main function to run the Kaggle retail analysis"""
    print("🚀 Starting Kaggle Retail Dataset Analysis...")
    
    # Load and prepare data
    df = load_retail_data()
    
    # Create interactive dashboard first
    print("\n🚀 Launching interactive dashboard...")
    store_summary = create_interactive_dashboard(df)
    
    # Create new analytical charts
    print("\n📊 Generating analytical charts...")
    holiday_analysis = create_holiday_impact_chart(df)
    correlation_matrix = create_economic_correlation_chart(df)
    
    # Generate comprehensive summary
    generate_executive_summary(df, holiday_analysis, correlation_matrix, store_summary)
    
    # Display static plots
    print("\n🖼️ Displaying static analysis charts...")
    plt.show()
    
    print("\n✅ Kaggle Retail Analysis Complete!")
    print("💾 Files saved:")
    print("   - holiday_impact_analysis.png")
    print("   - economic_factors_analysis.png")
    print("\n🔍 Interactive dashboard opened in browser for detailed exploration")

if __name__ == "__main__":
    main()