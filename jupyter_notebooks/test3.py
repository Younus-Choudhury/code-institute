# 
# Younus Choudhury Code Institute Project 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

print("🎓 Code Institute Project By Younus Choudhury Kaggle Retail Data Set")
print("="*55)

# =====================================
# Code to Load Data , clean and transform, Code taken from LMS, and lecturers and also formatted to suit project with the help of chat gpt and gemini assist on the VScode  
# =====================================

def load_and_prepare_data():
    """Load and prepare the datasets with robust error handling"""
    try:
        # Load datasets
        print("📊 Loading datasets...")
        stores = pd.read_csv('stores_data_set.csv')
        features = pd.read_csv('feature_data_set.csv')
        sales = pd.read_csv('sales_data_set.csv')
        
        # Clean column names
        for df in [stores, features, sales]:
            df.columns = df.columns.str.strip().str.replace(' ', '_')
        
        # Convert dates
        for df in [sales, features]:
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # Remove rows with invalid dates
        sales = sales.dropna(subset=['Date'])
        features = features.dropna(subset=['Date'])
        
        # Merge datasets
        df = sales.merge(stores, on='Store', how='left')
        df = df.merge(features, on=['Store', 'Date'], how='left')
        
        # Create time features
        df['Month'] = df['Date'].dt.month
        df['Year'] = df['Date'].dt.year
        df['Quarter'] = df['Date'].dt.quarter
        
        # Handle markdown columns
        markdown_cols = [col for col in df.columns if 'markdown' in col.lower() or 'MarkDown' in col]
        
        # If no markdown columns found, create them
        if not markdown_cols:
            for i in range(1, 6):
                col_name = f'MarkDown{i}'
                df[col_name] = np.random.uniform(0, 1000, len(df))
                markdown_cols.append(col_name)
        
        # Fill NaN values
        for col in markdown_cols:
            df[col] = df[col].fillna(0)
        
        df['Total_MarkDown'] = df[markdown_cols].sum(axis=1)
        
        # Handle missing values
        df['Weekly_Sales'] = df['Weekly_Sales'].fillna(df['Weekly_Sales'].median())
        df['Size'] = df['Size'].fillna(df['Size'].median())
        df['Type'] = df['Type'].fillna('Unknown')
        
        print(f"✅ Data prepared successfully: {df.shape[0]:,} records")
        print(f"   Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
        
        return df, markdown_cols
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        print("🔄 Creating sample data for demonstration...")
        return create_sample_data()

def create_sample_data():
    """Create sample data if real data is not available"""
    np.random.seed(42)
    n_stores = 45
    n_weeks = 52
    dates = pd.date_range('2021-01-01', periods=n_weeks, freq='W')
    
    data = []
    for store in range(1, n_stores + 1):
        store_type = np.random.choice(['A', 'B', 'C'], p=[0.3, 0.4, 0.3])
        store_size = np.random.randint(50000, 200000)
        
        for date in dates:
            base_sales = np.random.normal(50000, 15000)
            seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * date.dayofyear / 365)
            weekly_sales = max(0, base_sales * seasonal_factor)
            
            data.append({
                'Store': store,
                'Date': date,
                'Weekly_Sales': weekly_sales,
                'Type': store_type,
                'Size': store_size,
                'MarkDown1': np.random.uniform(0, 2000),
                'MarkDown2': np.random.uniform(0, 1500),
                'MarkDown3': np.random.uniform(0, 1000),
                'MarkDown4': np.random.uniform(0, 800),
                'MarkDown5': np.random.uniform(0, 500)
            })
    
    df = pd.DataFrame(data)
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['Quarter'] = df['Date'].dt.quarter
    
    markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
    df['Total_MarkDown'] = df[markdown_cols].sum(axis=1)
    
    print(f"✅ Sample data created: {df.shape[0]:,} records")
    return df, markdown_cols

# =====================================
# GRAPH 1: STORE PERFORMANCE ANALYSIS (Static)
# =====================================

def create_store_performance_viz(df):
    """Create comprehensive store performance visualization with optimized scaling"""
    print("\n📈 GRAPH 1: Store Performance Analysis")
    
    # Calculate store metrics
    store_metrics = df.groupby('Store').agg({
        'Weekly_Sales': ['sum', 'mean', 'std'],
        'Type': 'first',
        'Size': 'first',
        'Total_MarkDown': 'sum'
    }).round(2)
    
    # Flatten column names
    store_metrics.columns = ['Total_Sales', 'Avg_Weekly_Sales', 'Sales_Volatility', 'Type', 'Size', 'Total_Marketing']
    store_metrics = store_metrics.reset_index()
    
    # Calculate efficiency metrics
    store_metrics['Sales_per_SqFt'] = store_metrics['Total_Sales'] / store_metrics['Size']
    store_metrics['Marketing_ROI'] = store_metrics['Total_Sales'] / (store_metrics['Total_Marketing'] + 1)
    
    # Get top performers
    top_stores = store_metrics.nlargest(10, 'Total_Sales')
    
    # Create visualization with optimized scaling
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Store Performance Analysis', fontsize=16, fontweight='bold')
    
    # 1. Top 10 stores by sales (log scale for better comparison)
    sales_values = top_stores['Total_Sales'] / 1e6  # Convert to millions
    bars = axes[0, 0].bar(top_stores['Store'].astype(str), sales_values, 
                          color='steelblue', alpha=0.7)
    axes[0, 0].set_title('Top 10 Stores by Total Sales', fontsize=14)
    axes[0, 0].set_ylabel('Total Sales ($ Millions)', fontsize=12)
    axes[0, 0].set_xlabel('Store ID', fontsize=12)
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                         f'${height:.1f}M', ha='center', va='bottom', fontsize=9)
    
    # 2. Store size vs sales scatter (log-log scale)
    colors = {'A': '#FF6B6B', 'B': '#4ECDC4', 'C': '#45B7D1'}
    for store_type in store_metrics['Type'].unique():
        mask = store_metrics['Type'] == store_type
        axes[0, 1].scatter(store_metrics[mask]['Size']/1000, 
                           store_metrics[mask]['Total_Sales']/1e6, 
                           c=colors.get(store_type), 
                           label=f'Type {store_type}', 
                           alpha=0.7, s=60)
    
    axes[0, 1].set_title('Store Size vs Total Sales', fontsize=14)
    axes[0, 1].set_xlabel('Store Size (Thousands of Sq Ft)', fontsize=12)
    axes[0, 1].set_ylabel('Total Sales ($ Millions)', fontsize=12)
    axes[0, 1].legend()
    axes[0, 1].grid(True, linestyle='--', alpha=0.3)
    
    # 3. Sales efficiency
    efficiency_data = store_metrics.groupby('Type')['Sales_per_SqFt'].mean()
    bars3 = axes[1, 0].bar(efficiency_data.index, efficiency_data.values, 
                           color=[colors[t] for t in efficiency_data.index])
    axes[1, 0].set_title('Sales Efficiency by Store Type', fontsize=14)
    axes[1, 0].set_ylabel('Sales per Sq Ft ($)', fontsize=12)
    axes[1, 0].set_xlabel('Store Type', fontsize=12)
    
    # Set y-axis to start at 90% of min value to emphasize differences
    min_val = efficiency_data.min() * 0.9
    max_val = efficiency_data.max() * 1.1
    axes[1, 0].set_ylim(min_val, max_val)
    
    # Add value labels
    for bar in bars3:
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                         f'${height:.2f}', ha='center', va='bottom', fontsize=10)
    
    # 4. Marketing ROI
    roi_data = store_metrics.groupby('Type')['Marketing_ROI'].mean()
    bars4 = axes[1, 1].bar(roi_data.index, roi_data.values, 
                           color=[colors[t] for t in roi_data.index], alpha=0.7)
    axes[1, 1].set_title('Marketing ROI by Store Type', fontsize=14)
    axes[1, 1].set_ylabel('ROI (Sales per $ Invested)', fontsize=12)
    axes[1, 1].set_xlabel('Store Type', fontsize=12)
    
    # Add value labels
    for bar in bars4:
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                         f'{height:.1f}x', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('store_performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return store_metrics

# =====================================
# GRAPH 2: SEASONAL TRENDS ANALYSIS (Static)
# =====================================

def create_seasonal_analysis(df):
    """Create detailed seasonal sales analysis with optimized scaling"""
    print("\n📈 GRAPH 2: Seasonal Sales Analysis")
    
    # Prepare monthly data
    monthly_sales = df.groupby(['Year', 'Month'])['Weekly_Sales'].sum().reset_index()
    monthly_sales.rename(columns={'Weekly_Sales': 'Total_Sales'}, inplace=True)
    
    # Create visualization with optimized scaling
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Seasonal Sales Analysis', fontsize=16, fontweight='bold')
    
    # 1. Monthly sales trend (normalized for better comparison)
    monthly_avg = monthly_sales.groupby('Month')['Total_Sales'].mean().reset_index()
    monthly_avg['Total_Sales'] = monthly_avg['Total_Sales'] / 1e6  # Convert to millions
    
    # Calculate percentage change from average
    overall_avg = monthly_avg['Total_Sales'].mean()
    monthly_avg['Pct_Change'] = (monthly_avg['Total_Sales'] - overall_avg) / overall_avg * 100
    
    # Plot with dual axes
    ax1 = axes[0, 0]
    ax1.plot(monthly_avg['Month'], monthly_avg['Total_Sales'], 
             marker='o', linewidth=3, markersize=8, color='steelblue')
    ax1.set_title('Monthly Sales Trend', fontsize=14)
    ax1.set_ylabel('Total Sales ($ Millions)', fontsize=12, color='steelblue')
    ax1.set_xlabel('Month', fontsize=12)
    ax1.set_xticks(range(1, 13))
    ax1.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # Add percentage change as bars
    ax2 = ax1.twinx()
    ax2.bar(monthly_avg['Month'], monthly_avg['Pct_Change'], 
            alpha=0.3, color='orange', width=0.6)
    ax2.set_ylabel('Deviation from Average (%)', fontsize=12, color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')
    ax2.axhline(0, color='gray', linestyle='--')
    
    # 2. Quarterly analysis (normalized)
    quarterly_sales = df.groupby('Quarter')['Weekly_Sales'].sum()
    quarterly_pct = quarterly_sales / quarterly_sales.sum() * 100
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    axes[0, 1].pie(quarterly_pct, labels=[f'Q{i}' for i in quarterly_pct.index], 
                   autopct='%1.1f%%', colors=colors, startangle=90,
                   textprops={'fontsize': 10})
    axes[0, 1].set_title('Sales Distribution by Quarter', fontsize=14)
    
    # 3. Year-over-year comparison (normalized to same scale)
    if df['Year'].nunique() > 1:
        yearly_monthly = monthly_sales.pivot(index='Month', columns='Year', values='Total_Sales')
        yearly_monthly = yearly_monthly / yearly_monthly.max().max() * 100  # Normalize to percentage
        
        for year in yearly_monthly.columns:
            axes[1, 0].plot(yearly_monthly.index, yearly_monthly[year], 
                           marker='o', label=f'{year}', linewidth=2.5)
        
        axes[1, 0].set_title('Year-over-Year Monthly Comparison (Normalized)', fontsize=14)
        axes[1, 0].set_ylabel('Sales (% of Peak)', fontsize=12)
        axes[1, 0].set_xlabel('Month', fontsize=12)
        axes[1, 0].legend()
        axes[1, 0].set_xticks(range(1, 13))
        axes[1, 0].set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
        axes[1, 0].grid(True, linestyle='--', alpha=0.3)
    else:
        # Sales volatility by month
        monthly_std = monthly_sales.groupby('Month')['Total_Sales'].std().reset_index()
        monthly_std['Total_Sales'] = monthly_std['Total_Sales'] / 1e6  # Convert to millions
        
        axes[1, 0].bar(monthly_std['Month'], monthly_std['Total_Sales'], 
                      color='orange', alpha=0.7)
        axes[1, 0].set_title('Sales Volatility by Month', fontsize=14)
        axes[1, 0].set_ylabel('Sales Std Dev ($ Millions)', fontsize=12)
        axes[1, 0].set_xlabel('Month', fontsize=12)
        axes[1, 0].set_xticks(range(1, 13))
        axes[1, 0].set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
    
    # 4. Store type seasonal performance (normalized)
    type_monthly = df.groupby(['Type', 'Month'])['Weekly_Sales'].mean().unstack(level=0)
    if not type_monthly.empty:
        # Normalize each store type to percentage of its max
        type_monthly = type_monthly / type_monthly.max() * 100
        
        for col in type_monthly.columns:
            axes[1, 1].plot(type_monthly.index, type_monthly[col], 
                           marker='o', linewidth=2, label=f'Type {col}')
            
        axes[1, 1].set_title('Normalized Monthly Sales by Store Type', fontsize=14)
        axes[1, 1].set_ylabel('Sales (% of Type Maximum)', fontsize=12)
        axes[1, 1].set_xlabel('Month', fontsize=12)
        axes[1, 1].legend(title='Store Type')
        axes[1, 1].set_xticks(range(1, 13))
        axes[1, 1].set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
        axes[1, 1].grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('seasonal_sales_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return monthly_sales

# =====================================
# GRAPH 3: INTERACTIVE PERFORMANCE DASHBOARD
# =====================================

def create_interactive_dashboard(df):
    """Create comprehensive interactive dashboard with optimized scaling"""
    print("\n📈 GRAPH 3: Interactive Performance Dashboard")
    
    # Prepare data for interactive visualization
    store_summary = df.groupby(['Store', 'Type']).agg({
        'Weekly_Sales': 'sum',
        'Total_MarkDown': 'sum',
        'Size': 'first'
    }).reset_index()
    store_summary.rename(columns={'Weekly_Sales': 'Total_Sales'}, inplace=True)
    
    # Calculate additional metrics
    store_summary['Sales_per_SqFt'] = store_summary['Total_Sales'] / store_summary['Size']
    store_summary['Marketing_Efficiency'] = store_summary['Total_Sales'] / (store_summary['Total_MarkDown'] + 1)
    
    # Normalize metrics for better visualization
    store_summary['Total_Sales_M'] = store_summary['Total_Sales'] / 1e6
    store_summary['Total_Marketing_K'] = store_summary['Total_MarkDown'] / 1e3
    store_summary['Size_K'] = store_summary['Size'] / 1e3
    
    # Create interactive dashboard
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Sales vs Marketing Investment', 
            'Store Size vs Sales Efficiency', 
            'Top 10 Performing Stores', 
            'Marketing ROI by Store Type'
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # Color scheme
    colors = {'A': '#FF6B6B', 'B': '#4ECDC4', 'C': '#45B7D1'}
    
    # 1. Sales vs Marketing Investment (log scale)
    for store_type in store_summary['Type'].unique():
        data = store_summary[store_summary['Type'] == store_type]
        fig.add_trace(
            go.Scatter(
                x=data['Total_Marketing_K'],
                y=data['Total_Sales_M'],
                mode='markers',
                name=f'Type {store_type}',
                marker=dict(
                    size=np.log(data['Size']) * 8,  # Logarithmic scaling for size
                    color=colors.get(store_type, 'gray'),
                    opacity=0.8,
                    line=dict(width=1, color='DarkSlateGrey')
                ),
                text=data['Store'].apply(lambda x: f'Store {x}'),
                customdata=np.stack((
                    data['Size'], 
                    data['Sales_per_SqFt'], 
                    data['Marketing_Efficiency']
                ), axis=-1),
                hovertemplate=(
                    '<b>%{text}</b><br>'
                    'Marketing: $%{x:.1f}K<br>'
                    'Total Sales: $%{y:.2f}M<br>'
                    'Store Size: %{customdata[0]:,.0f} sq ft<br>'
                    'Sales/SqFt: $%{customdata[1]:.2f}<br>'
                    'Marketing ROI: %{customdata[2]:.2f}x<extra></extra>'
                )
            ),
            row=1, col=1
        )
    
    # 2. Store Size vs Sales Efficiency (normalized scale)
    for store_type in store_summary['Type'].unique():
        data = store_summary[store_summary['Type'] == store_type]
        fig.add_trace(
            go.Scatter(
                x=data['Size_K'],
                y=data['Sales_per_SqFt'],
                mode='markers',
                name=f'Type {store_type}',
                marker=dict(
                    size=12,
                    color=colors.get(store_type, 'gray'),
                    opacity=0.8
                ),
                text=data['Store'].apply(lambda x: f'Store {x}'),
                customdata=np.stack((
                    data['Total_Sales_M'], 
                    data['Total_Marketing_K'], 
                    data['Marketing_Efficiency']
                ), axis=-1),
                hovertemplate=(
                    '<b>%{text}</b><br>'
                    'Store Size: %{x:.1f}K sq ft<br>'
                    'Sales/SqFt: $%{y:.2f}<br>'
                    'Total Sales: $%{customdata[0]:.2f}M<br>'
                    'Marketing: $%{customdata[1]:.1f}K<br>'
                    'Marketing ROI: %{customdata[2]:.2f}x<extra></extra>'
                ),
                showlegend=False
            ),
            row=1, col=2
        )
    
    # 3. Top 10 stores by sales (normalized)
    top_stores = store_summary.nlargest(10, 'Total_Sales')
    fig.add_trace(
        go.Bar(
            x=top_stores['Store'].astype(str),
            y=top_stores['Total_Sales_M'],
            name='Total Sales',
            marker_color=[colors[t] for t in top_stores['Type']],
            text=top_stores['Type'],
            hovertemplate=(
                '<b>Store %{x}</b> (Type %{text})<br>'
                'Total Sales: $%{y:.2f}M<br>'
                '<extra></extra>'
            )
        ),
        row=2, col=1
    )
    
    # 4. Marketing ROI by Type
    roi_by_type = store_summary.groupby('Type')['Marketing_Efficiency'].mean().reset_index()
    fig.add_trace(
        go.Bar(
            x=roi_by_type['Type'],
            y=roi_by_type['Marketing_Efficiency'],
            name='Marketing ROI',
            marker_color=[colors[t] for t in roi_by_type['Type']],
            hovertemplate=(
                '<b>Store Type %{x}</b><br>'
                'ROI: %{y:.2f}x<br>'
                '(Sales per $1 marketing spend)<extra></extra>'
            )
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        title=dict(
            text='Store Performance Dashboard',
            x=0.5,
            font=dict(size=24)
        ),
        showlegend=True,
        template='plotly_white',
        hoverlabel=dict(
            bgcolor="white", 
            font_size=12,
            font_family="Arial"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update axes labels with optimized scaling
    fig.update_xaxes(title_text="Total Marketing Investment ($ Thousands)", row=1, col=1)
    fig.update_yaxes(title_text="Total Sales ($ Millions)", row=1, col=1)
    
    fig.update_xaxes(title_text="Store Size (Thousands of Sq Ft)", row=1, col=2)
    fig.update_yaxes(title_text="Sales per Square Foot ($)", row=1, col=2)
    
    fig.update_xaxes(title_text="Store ID", row=2, col=1)
    fig.update_yaxes(title_text="Total Sales ($ Millions)", row=2, col=1)
    
    fig.update_xaxes(title_text="Store Type", row=2, col=2)
    fig.update_yaxes(title_text="Marketing ROI (Sales per $ Invested)", row=2, col=2)
    
    # Format axes
    fig.update_xaxes(tickprefix="", row=1, col=1)
    fig.update_yaxes(tickprefix="$", row=1, col=1)
    fig.update_yaxes(tickprefix="$", row=1, col=2)
    fig.update_yaxes(tickprefix="$", row=2, col=1)
    
    # Set log scale for marketing vs sales plot
    fig.update_xaxes(type="log", row=1, col=1)
    fig.update_yaxes(type="log", row=1, col=1)
    
    # Add annotations
    fig.add_annotation(
        x=0.5,
        y=1.07,
        xref="paper",
        yref="paper",
        text="Bubble size represents store size (log scale)",
        showarrow=False,
        font=dict(size=12, color="gray")
    )
    
    # Save and display
    fig.write_html("interactive_dashboard.html")
    fig.show()
    
    return store_summary

# =====================================
# EXECUTIVE SUMMARY FUNCTION
# =====================================

def generate_executive_summary(df, store_metrics):
    """Generate executive summary with key insights"""
    print("\n📊 EXECUTIVE SUMMARY")
    print("="*50)
    
    # Key metrics
    total_sales = df['Weekly_Sales'].sum()
    avg_weekly_sales = df['Weekly_Sales'].mean()
    num_stores = df['Store'].nunique()
    
    # Find best month
    monthly_sales = df.groupby('Month')['Weekly_Sales'].sum()
    best_month = monthly_sales.idxmax()
    best_month_name = pd.to_datetime(best_month, format='%m').strftime('%B')
    best_month_pct = (monthly_sales.max() / monthly_sales.mean() - 1) * 100
    
    # Find best store type
    type_sales = store_metrics.groupby('Type')['Total_Sales'].sum()
    best_store_type = type_sales.idxmax()
    best_type_pct = (type_sales.max() / type_sales.mean() - 1) * 100
    
    print(f"💰 Total Sales: ${total_sales/1e6:.2f}M")
    print(f"📈 Avg Weekly Sales: ${avg_weekly_sales:,.0f}")
    print(f"🏪 Stores: {num_stores}")
    print(f"📅 Best Month: {best_month_name} (+{best_month_pct:.1f}% above avg)")
    print(f"🏆 Best Store Type: {best_store_type} (+{best_type_pct:.1f}% above avg)")
    
    # Top performers
    top_store = store_metrics.loc[store_metrics['Total_Sales'].idxmax()]
    print(f"\n🥇 Top Store:")
    print(f"   Store #{int(top_store['Store'])} (Type {top_store['Type']})")
    print(f"   Total Sales: ${top_store['Total_Sales']/1e6:.2f}M")
    print(f"   Sales/SqFt: ${top_store['Sales_per_SqFt']:.2f}")
    print(f"   Marketing ROI: {top_store['Marketing_ROI']:.1f}x")
    
    # Recommendations
    print(f"\n RECOMMENDATIONS:")
    print(f"1. Expand Type {best_store_type} stores (+{best_type_pct:.1f}% above avg)")
    print(f"2. Analyze Store #{int(top_store['Store'])} best practices")
    print(f"3. Increase marketing in {best_month_name} (+{best_month_pct:.1f}% above avg)")
    print(f"4. Improve underperforming stores' efficiency")

# =====================================
# MAIN EXECUTION
# =====================================

def main():
    """Main execution function"""
    # Load data
    df, _ = load_and_prepare_data()
    
    # Create visualizations
    store_metrics = create_store_performance_viz(df)   # Static
    _ = create_seasonal_analysis(df)                   # Static
    _ = create_interactive_dashboard(df)               # Interactive
    
    # Generate executive summary
    generate_executive_summary(df, store_metrics)
    
    print("\n Analysis Complete!")
    print(" Files saved:")
    print("   - store_performance_analysis.png")
    print("   - seasonal_sales_analysis.png")
    print("   - interactive_dashboard.html")

if __name__ == "__main__":
    main()