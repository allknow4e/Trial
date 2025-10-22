import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from datetime import datetime, timedelta

# Set page configuration
st.set_page_config(
    page_title="Global Commodity Prices Analysis",
    page_icon="⛽",
    layout="wide"
)

# Title and description
st.title("⛽ Global Commodity Prices Analysis (1994-2023)")
st.markdown("""
Analysis of Petrol, Diesel, Gold, and Silver prices across India and 10 other countries over 30 years.
Explore continuous distributions and price patterns interactively!
""")

# Generate realistic commodity price data
@st.cache_data
def generate_commodity_data():
    countries = [
        'India', 'USA', 'China', 'Germany', 'Japan',
        'UK', 'Brazil', 'Russia', 'Australia', 'UAE', 'Canada'
    ]
    
    commodities = ['Petrol', 'Diesel', 'Gold', 'Silver']
    
    # Base prices in USD (1994)
    base_prices = {
        'Petrol': 0.3,    # USD per liter
        'Diesel': 0.25,   # USD per liter
        'Gold': 384.0,    # USD per ounce
        'Silver': 5.3     # USD per ounce
    }
    
    # Country-specific multipliers (cost factors)
    country_multipliers = {
        'India': {'Petrol': 1.2, 'Diesel': 1.1, 'Gold': 1.05, 'Silver': 1.08},
        'USA': {'Petrol': 1.0, 'Diesel': 0.9, 'Gold': 1.0, 'Silver': 1.0},
        'China': {'Petrol': 1.1, 'Diesel': 1.0, 'Gold': 1.02, 'Silver': 1.03},
        'Germany': {'Petrol': 1.4, 'Diesel': 1.3, 'Gold': 1.01, 'Silver': 1.02},
        'Japan': {'Petrol': 1.3, 'Diesel': 1.2, 'Gold': 1.03, 'Silver': 1.04},
        'UK': {'Petrol': 1.5, 'Diesel': 1.4, 'Gold': 1.01, 'Silver': 1.02},
        'Brazil': {'Petrol': 1.15, 'Diesel': 1.05, 'Gold': 1.06, 'Silver': 1.07},
        'Russia': {'Petrol': 0.8, 'Diesel': 0.7, 'Gold': 1.04, 'Silver': 1.05},
        'Australia': {'Petrol': 1.1, 'Diesel': 1.0, 'Gold': 1.0, 'Silver': 1.01},
        'UAE': {'Petrol': 0.5, 'Diesel': 0.4, 'Gold': 0.98, 'Silver': 0.99},
        'Canada': {'Petrol': 0.9, 'Diesel': 0.8, 'Gold': 1.01, 'Silver': 1.02}
    }
    
    # Generate monthly data from 1994 to 2023
    dates = pd.date_range('1994-01-01', '2023-12-31', freq='M')
    data = []
    
    np.random.seed(42)
    
    for country in countries:
        for commodity in commodities:
            base_price = base_prices[commodity] * country_multipliers[country][commodity]
            price = base_price
            
            for date in dates:
                # Add realistic price fluctuations
                if commodity in ['Petrol', 'Diesel']:
                    # Fuel prices are more volatile
                    monthly_change = np.random.normal(0.002, 0.08)  # 0.2% mean growth, 8% volatility
                    # Add some major events (price spikes)
                    if np.random.random() < 0.01:  # 1% chance of major event
                        monthly_change += np.random.uniform(0.1, 0.3)
                else:
                    # Precious metals are less volatile
                    monthly_change = np.random.normal(0.003, 0.05)  # 0.3% mean growth, 5% volatility
                
                price = max(0.1, price * (1 + monthly_change))
                
                # Add seasonal effects for fuels
                if commodity in ['Petrol', 'Diesel']:
                    seasonal_effect = 0.05 * np.sin(2 * np.pi * date.month / 12)
                    price = price * (1 + seasonal_effect)
                
                # Convert to local currencies (simplified)
                if country == 'India':
                    local_price = price * 75  # INR
                    currency = 'INR'
                elif country == 'Japan':
                    local_price = price * 110  # JPY
                    currency = 'JPY'
                elif country == 'UK':
                    local_price = price * 0.75  # GBP
                    currency = 'GBP'
                elif country in ['Brazil', 'Russia']:
                    local_price = price * 5  # Local currency units
                    currency = 'LCU'
                else:
                    local_price = price  # USD
                    currency = 'USD'
                
                data.append({
                    'Country': country,
                    'Commodity': commodity,
                    'Date': date,
                    'Year': date.year,
                    'Month': date.month,
                    'Price_USD': price,
                    'Price_Local': local_price,
                    'Currency': currency
                })
    
    return pd.DataFrame(data)

# Generate the dataset
commodity_data = generate_commodity_data()

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox(
    "Choose Analysis Mode",
    ["Data Overview", "Price Trends", "Distribution Calculator", 
     "Distribution Visualization", "Country Comparison", "Standard Distributions"]
)

# Page 1: Data Overview
if app_mode == "Data Overview":
    st.header("📊 Global Commodity Prices Dataset (1994-2023)")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Dataset Preview")
        st.dataframe(commodity_data.head(10), use_container_width=True)
        
        st.subheader("Data Summary by Country")
        summary_stats = commodity_data.groupby(['Country', 'Commodity']).agg({
            'Price_USD': ['mean', 'std', 'min', 'max']
        }).round(3)
        st.dataframe(summary_stats, use_container_width=True)
    
    with col2:
        st.subheader("Dataset Information")
        st.write(f"**Total Records:** {len(commodity_data):,}")
        st.write(f"**Number of Countries:** {commodity_data['Country'].nunique()}")
        st.write(f"**Commodities:** {', '.join(commodity_data['Commodity'].unique())}")
        st.write(f"**Time Period:** {commodity_data['Year'].min()} - {commodity_data['Year'].max()}")
        st.write(f"**Time Frequency:** Monthly data")
        
        st.subheader("Latest Prices (Dec 2023)")
        latest_prices = commodity_data[commodity_data['Date'] == commodity_data['Date'].max()]
        pivot_latest = latest_prices.pivot_table(
            index='Country', 
            columns='Commodity', 
            values='Price_USD'
        ).round(2)
        st.dataframe(pivot_latest, use_container_width=True)

# Page 2: Price Trends
elif app_mode == "Price Trends":
    st.header("📈 Commodity Price Trends Analysis")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        selected_countries = st.multiselect(
            "Select Countries:",
            options=commodity_data['Country'].unique(),
            default=['India', 'USA', 'Germany']
        )
        
        selected_commodities = st.multiselect(
            "Select Commodities:",
            options=commodity_data['Commodity'].unique(),
            default=['Petrol', 'Gold']
        )
        
        aggregation = st.radio(
            "Time Aggregation:",
            ["Yearly", "Monthly", "Quarterly"]
        )
        
        show_log_scale = st.checkbox("Logarithmic Scale")
    
    with col2:
        if selected_countries and selected_commodities:
            # Filter and aggregate data
            filtered_data = commodity_data[
                (commodity_data['Country'].isin(selected_countries)) &
                (commodity_data['Commodity'].isin(selected_commodities))
            ]
            
            if aggregation == "Yearly":
                trend_data = filtered_data.groupby(['Year', 'Country', 'Commodity'])['Price_USD'].mean().reset_index()
                x_col = 'Year'
            elif aggregation == "Quarterly":
                filtered_data['Quarter'] = filtered_data['Date'].dt.quarter
                filtered_data['YearQuarter'] = filtered_data['Year'].astype(str) + 'Q' + filtered_data['Quarter'].astype(str)
                trend_data = filtered_data.groupby(['YearQuarter', 'Country', 'Commodity'])['Price_USD'].mean().reset_index()
                x_col = 'YearQuarter'
            else:
                trend_data = filtered_data
                x_col = 'Date'
            
            # Create interactive plot
            fig = go.Figure()
            
            for country in selected_countries:
                for commodity in selected_commodities:
                    country_commodity_data = trend_data[
                        (trend_data['Country'] == country) & 
                        (trend_data['Commodity'] == commodity)
                    ]
                    
                    fig.add_trace(go.Scatter(
                        x=country_commodity_data[x_col],
                        y=country_commodity_data['Price_USD'],
                        name=f"{country} - {commodity}",
                        mode='lines',
                        opacity=0.8
                    ))
            
            fig.update_layout(
                title=f"{aggregation} Price Trends ({', '.join(selected_commodities)})",
                xaxis_title="Time",
                yaxis_title="Price (USD)",
                height=500,
                showlegend=True
            )
            
            if show_log_scale:
                fig.update_yaxis(type="log")
                
            st.plotly_chart(fig, use_container_width=True)
            
            # Price change analysis
            st.subheader("Price Change Analysis")
            col1, col2, col3, col4 = st.columns(4)
            
            for commodity in selected_commodities:
                with col1 if commodity == selected_commodities[0] else col2 if commodity == selected_commodities[1] else col3 if commodity == selected_commodities[2] else col4:
                    comm_data = filtered_data[filtered_data['Commodity'] == commodity]
                    first_price = comm_data[comm_data['Year'] == 1994]['Price_USD'].mean()
                    last_price = comm_data[comm_data['Year'] == 2023]['Price_USD'].mean()
                    change_pct = ((last_price - first_price) / first_price) * 100
                    
                    st.metric(
                        label=f"{commodity} Price Change (1994-2023)",
                        value=f"${last_price:.2f}",
                        delta=f"{change_pct:.1f}%"
                    )

# Page 3: Distribution Calculator
elif app_mode == "Distribution Calculator":
    st.header("🧮 Distribution Probability Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution selection
        distribution_type = st.selectbox(
            "Select Distribution:",
            ["Normal", "Exponential", "Gamma", "Uniform", "Lognormal"]
        )
        
        # Parameter inputs based on distribution
        if distribution_type == "Normal":
            mu = st.slider("Mean (μ)", 0.0, 10.0, 2.0, 0.1)
            sigma = st.slider("Standard Deviation (σ)", 0.1, 5.0, 1.0, 0.1)
            
        elif distribution_type == "Exponential":
            lambda_param = st.slider("Rate (λ)", 0.1, 5.0, 1.0, 0.1)
            
        elif distribution_type == "Gamma":
            alpha = st.slider("Shape (α)", 0.1, 10.0, 2.0, 0.1)
            beta = st.slider("Rate (β)", 0.1, 5.0, 1.0, 0.1)
            
        elif distribution_type == "Uniform":
            a = st.slider("Lower bound (a)", 0.0, 10.0, 0.0, 0.1)
            b = st.slider("Upper bound (b)", 0.0, 10.0, 5.0, 0.1)
            if b <= a:
                st.error("b must be greater than a")
                b = a + 0.1
                
        elif distribution_type == "Lognormal":
            mu_ln = st.slider("Log Mean (μ)", -2.0, 2.0, 0.0, 0.1)
            sigma_ln = st.slider("Log Std (σ)", 0.1, 2.0, 1.0, 0.1)
        
        # Probability calculation inputs
        st.subheader("Probability Calculation")
        x_value = st.number_input("X value for probability calculation:", value=2.0, min_value=0.0)
        prob_type = st.radio("Probability Type:", ["PDF", "CDF", "Survival Function (1-CDF)"])
        
        # Calculate probability
        if distribution_type == "Normal":
            if prob_type == "PDF":
                result = stats.norm.pdf(x_value, mu, sigma)
            elif prob_type == "CDF":
                result = stats.norm.cdf(x_value, mu, sigma)
            else:
                result = 1 - stats.norm.cdf(x_value, mu, sigma)
                
        elif distribution_type == "Exponential":
            if prob_type == "PDF":
                result = stats.expon.pdf(x_value, scale=1/lambda_param)
            elif prob_type == "CDF":
                result = stats.expon.cdf(x_value, scale=1/lambda_param)
            else:
                result = 1 - stats.expon.cdf(x_value, scale=1/lambda_param)
                
        elif distribution_type == "Gamma":
            if prob_type == "PDF":
                result = stats.gamma.pdf(x_value, alpha, scale=1/beta)
            elif prob_type == "CDF":
                result = stats.gamma.cdf(x_value, alpha, scale=1/beta)
            else:
                result = 1 - stats.gamma.cdf(x_value, alpha, scale=1/beta)
                
        elif distribution_type == "Uniform":
            if prob_type == "PDF":
                result = stats.uniform.pdf(x_value, a, b-a)
            elif prob_type == "CDF":
                result = stats.uniform.cdf(x_value, a, b-a)
            else:
                result = 1 - stats.uniform.cdf(x_value, a, b-a)
                
        elif distribution_type == "Lognormal":
            if prob_type == "PDF":
                result = stats.lognorm.pdf(x_value, sigma_ln, scale=np.exp(mu_ln))
            elif prob_type == "CDF":
                result = stats.lognorm.cdf(x_value, sigma_ln, scale=np.exp(mu_ln))
            else:
                result = 1 - stats.lognorm.cdf(x_value, sigma_ln, scale=np.exp(mu_ln))
    
    with col2:
        st.subheader("Probability Result")
        st.metric(
            label=f"{prob_type} Value at x = {x_value}",
            value=f"{result:.6f}"
        )
        
        # Generate distribution plot
        if distribution_type == "Normal":
            x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
            y_pdf = stats.norm.pdf(x, mu, sigma)
            y_cdf = stats.norm.cdf(x, mu, sigma)
        elif distribution_type == "Exponential":
            x = np.linspace(0, 5/lambda_param, 1000)
            y_pdf = stats.expon.pdf(x, scale=1/lambda_param)
            y_cdf = stats.expon.cdf(x, scale=1/lambda_param)
        elif distribution_type == "Gamma":
            x = np.linspace(0, 3*alpha/beta, 1000)
            y_pdf = stats.gamma.pdf(x, alpha, scale=1/beta)
            y_cdf = stats.gamma.cdf(x, alpha, scale=1/beta)
        elif distribution_type == "Uniform":
            x = np.linspace(a-1, b+1, 1000)
            y_pdf = stats.uniform.pdf(x, a, b-a)
            y_cdf = stats.uniform.cdf(x, a, b-a)
        elif distribution_type == "Lognormal":
            x = np.linspace(0.01, stats.lognorm.ppf(0.99, sigma_ln, scale=np.exp(mu_ln)), 1000)
            y_pdf = stats.lognorm.pdf(x, sigma_ln, scale=np.exp(mu_ln))
            y_cdf = stats.lognorm.cdf(x, sigma_ln, scale=np.exp(mu_ln))
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1, 
            subplot_titles=(
                f'Probability Density Function - {distribution_type}',
                f'Cumulative Distribution Function - {distribution_type}'
            )
        )
        
        fig.add_trace(go.Scatter(x=x, y=y_pdf, name='PDF', line=dict(color='blue')), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=y_cdf, name='CDF', line=dict(color='red')), row=2, col=1)
        
        # Add vertical line for the selected x value
        fig.add_vline(x=x_value, line_dash="dash", line_color="green", row=1, col=1)
        fig.add_vline(x=x_value, line_dash="dash", line_color="green", row=2, col=1)
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# Page 4: Distribution Visualization
elif app_mode == "Distribution Visualization":
    st.header("📊 Distribution Fitting to Real Commodity Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Data selection
        selected_country = st.selectbox(
            "Select Country:",
            options=commodity_data['Country'].unique(),
            index=0
        )
        
        selected_commodity = st.selectbox(
            "Select Commodity:",
            options=commodity_data['Commodity'].unique(),
            index=0
        )
        
        selected_year = st.slider(
            "Select Year Range:",
            min_value=1994,
            max_value=2023,
            value=(2010, 2023)
        )
        
        # Filter data
        filtered_data = commodity_data[
            (commodity_data['Country'] == selected_country) &
            (commodity_data['Commodity'] == selected_commodity) &
            (commodity_data['Year'] >= selected_year[0]) &
            (commodity_data['Year'] <= selected_year[1])
        ]
        
        prices = filtered_data['Price_USD'].values
        
        if len(prices) == 0:
            st.warning("No data available for the selected criteria.")
        else:
            st.subheader("Price Distribution")
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=prices, 
                name='Price Data',
                nbinsx=30,
                opacity=0.7,
                histnorm='probability density'
            ))
            fig_hist.update_layout(
                title=f"{selected_commodity} Prices in {selected_country} ({selected_year[0]}-{selected_year[1]})",
                xaxis_title="Price (USD)",
                yaxis_title="Density"
            )
            st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        if len(prices) > 0:
            # Distribution fitting
            st.subheader("Distribution Fitting")
            dist_to_fit = st.selectbox(
                "Select distribution to fit:",
                ["Normal", "Lognormal", "Exponential", "Gamma"]
            )
            
            # Fit distribution
            if dist_to_fit == "Normal":
                params = stats.norm.fit(prices)
                x_fit = np.linspace(prices.min(), prices.max(), 100)
                fitted_pdf = stats.norm.pdf(x_fit, *params)
                param_text = f"μ = {params[0]:.3f}, σ = {params[1]:.3f}"
                
            elif dist_to_fit == "Lognormal":
                params = stats.lognorm.fit(prices)
                x_fit = np.linspace(prices.min(), prices.max(), 100)
                fitted_pdf = stats.lognorm.pdf(x_fit, *params)
                param_text = f"σ = {params[0]:.3f}, scale = {params[2]:.3f}"
                
            elif dist_to_fit == "Exponential":
                params = stats.expon.fit(prices)
                x_fit = np.linspace(0, prices.max(), 100)
                fitted_pdf = stats.expon.pdf(x_fit, *params)
                param_text = f"λ = {1/params[1]:.3f}"
                
            elif dist_to_fit == "Gamma":
                params = stats.gamma.fit(prices)
                x_fit = np.linspace(0, prices.max(), 100)
                fitted_pdf = stats.gamma.pdf(x_fit, *params)
                param_text = f"α = {params[0]:.3f}, β = {1/params[2]:.3f}"
            
            # Plot fitted distribution
            fig_fit = go.Figure()
            fig_fit.add_trace(go.Histogram(
                x=prices,
                name='Actual Data',
                nbinsx=30,
                opacity=0.5,
                histnorm='probability density'
            ))
            fig_fit.add_trace(go.Scatter(
                x=x_fit,
                y=fitted_pdf,
                name=f'Fitted {dist_to_fit}',
                line=dict(color='red', width=3)
            ))
            fig_fit.update_layout(
                title=f"Data vs Fitted {dist_to_fit} Distribution",
                xaxis_title="Price (USD)",
                yaxis_title="Density"
            )
            st.plotly_chart(fig_fit, use_container_width=True)
            
            st.write(f"**Fitted Parameters:** {param_text}")
            
            # Goodness of fit test
            if st.button("Calculate Goodness of Fit"):
                if dist_to_fit == "Normal":
                    _, p_value = stats.normaltest(prices)
                    test_name = "Normality Test"
                else:
                    # KS test for other distributions
                    if dist_to_fit == "Lognormal":
                        cdf = lambda x: stats.lognorm.cdf(x, *params)
                    elif dist_to_fit == "Exponential":
                        cdf = lambda x: stats.expon.cdf(x, *params)
                    elif dist_to_fit == "Gamma":
                        cdf = lambda x: stats.gamma.cdf(x, *params)
                    
                    _, p_value = stats.kstest(prices, cdf)
                    test_name = "Kolmogorov-Smirnov Test"
                
                st.metric(f"{test_name} p-value", f"{p_value:.4f}")
                if p_value > 0.05:
                    st.success("The data appears to follow the distribution (p > 0.05)")
                else:
                    st.warning("The data may not follow this distribution (p ≤ 0.05)")

# Page 5: Country Comparison
elif app_mode == "Country Comparison":
    st.header("🌍 Cross-Country Price Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_commodity = st.selectbox(
            "Commodity for Comparison:",
            options=commodity_data['Commodity'].unique()
        )
        
        comparison_year = st.slider(
            "Select Year:",
            min_value=1994,
            max_value=2023,
            value=2023
        )
        
        # Get data for selected year and commodity
        comp_data = commodity_data[
            (commodity_data['Commodity'] == selected_commodity) &
            (commodity_data['Year'] == comparison_year)
        ]
        
        if len(comp_data) > 0:
            # Calculate average prices by country
            avg_prices = comp_data.groupby('Country')['Price_USD'].mean().sort_values()
            
            # Create bar chart
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=avg_prices.index,
                y=avg_prices.values,
                marker_color='lightblue'
            ))
            fig_bar.update_layout(
                title=f"Average {selected_commodity} Prices by Country ({comparison_year})",
                xaxis_title="Country",
                yaxis_title="Average Price (USD)",
                height=400
            )
            st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        if len(comp_data) > 0:
            # Price distribution by country
            st.subheader("Price Distributions by Country")
            
            selected_countries_comp = st.multiselect(
                "Select countries for distribution comparison:",
                options=commodity_data['Country'].unique(),
                default=['India', 'USA', 'Germany', 'UAE']
            )
            
            if selected_countries_comp:
                fig_dist = go.Figure()
                
                for country in selected_countries_comp:
                    country_data = comp_data[comp_data['Country'] == country]['Price_USD']
                    fig_dist.add_trace(go.Box(
                        y=country_data,
                        name=country,
                        boxpoints='outliers'
                    ))
                
                fig_dist.update_layout(
                    title=f"{selected_commodity} Price Distribution by Country ({comparison_year})",
                    yaxis_title="Price (USD)",
                    height=400
                )
                st.plotly_chart(fig_dist, use_container_width=True)
            
            # Statistical summary
            st.subheader("Statistical Summary")
            summary = comp_data.groupby('Country')['Price_USD'].agg(['mean', 'std', 'min', 'max']).round(3)
            st.dataframe(summary, use_container_width=True)

# Page 6: Standard Distributions
elif app_mode == "Standard Distributions":
    st.header("⭐ Standard Distributions Reference")
    
    st.markdown("""
    Standard distributions have fixed parameters and are commonly used as reference distributions 
    in statistical analysis and hypothesis testing.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Standard Normal Distribution (μ=0, σ=1)")
        x_norm = np.linspace(-4, 4, 1000)
        y_norm_pdf = stats.norm.pdf(x_norm)
        y_norm_cdf = stats.norm.cdf(x_norm)
        
        fig_std_norm = make_subplots(rows=2, cols=1)
        fig_std_norm.add_trace(go.Scatter(x=x_norm, y=y_norm_pdf, name='PDF', line=dict(color='blue')), row=1, col=1)
        fig_std_norm.add_trace(go.Scatter(x=x_norm, y=y_norm_cdf, name='CDF', line=dict(color='red')), row=2, col=1)
        fig_std_norm.update_layout(
            title="Standard Normal Distribution",
            height=500
        )
        st.plotly_chart(fig_std_norm, use_container_width=True)
        
        st.subheader("Standard Uniform Distribution (a=0, b=1)")
        x_uni = np.linspace(-0.5, 1.5, 1000)
        y_uni_pdf = stats.uniform.pdf(x_uni, 0, 1)
        y_uni_cdf = stats.uniform.cdf(x_uni, 0, 1)
        
        fig_std_uni = make_subplots(rows=2, cols=1)
        fig_std_uni.add_trace(go.Scatter(x=x_uni, y=y_uni_pdf, name='PDF', line=dict(color='green')), row=1, col=1)
        fig_std_uni.add_trace(go.Scatter(x=x_uni, y=y_uni_cdf, name='CDF', line=dict(color='orange')), row=2, col=1)
        fig_std_uni.update_layout(
            title="Standard Uniform Distribution",
            height=500
        )
        st.plotly_chart(fig_std_uni, use_container_width=True)
    
    with col2:
        st.subheader("Standard Exponential Distribution (λ=1)")
        x_exp = np.linspace(0, 5, 1000)
        y_exp_pdf = stats.expon.pdf(x_exp)
        y_exp_cdf = stats.expon.cdf(x_exp)
        
        fig_std_exp = make_subplots(rows=2, cols=1)
        fig_std_exp.add_trace(go.Scatter(x=x_exp, y=y_exp_pdf, name='PDF', line=dict(color='purple')), row=1, col=1)
        fig_std_exp.add_trace(go.Scatter(x=x_exp, y=y_exp_cdf, name='CDF', line=dict(color='brown')), row=2, col=1)
        fig_std_exp.update_layout(
            title="Standard Exponential Distribution",
            height=500
        )
        st.plotly_chart(fig_std_exp, use_container_width=True)
        
        # Distribution comparison
        st.subheader("Standard Distributions Comparison")
        fig_compare = go.Figure()
        fig_compare.add_trace(go.Scatter(x=x_norm, y=y_norm_pdf, name='Standard Normal'))
        fig_compare.add_trace(go.Scatter(x=x_uni, y=y_uni_pdf, name='Standard Uniform'))
        fig_compare.add_trace(go.Scatter(x=x_exp, y=y_exp_pdf, name='Standard Exponential'))
        
        fig_compare.update_layout(
            title="PDF Comparison of Standard Distributions",
            xaxis_title="x",
            yaxis_title="Probability Density",
            height=400
        )
        st.plotly_chart(fig_compare, use_container_width=True)
        
        # Statistical properties table
        st.subheader("Statistical Properties of Standard Distributions")
        properties_data = {
            'Distribution': ['Standard Normal', 'Standard Uniform', 'Standard Exponential'],
            'Mean': [0, 0.5, 1],
            'Variance': [1, 1/12, 1],
            'Skewness': [0, 0, 2],
            'Support': ['(-∞, ∞)', '[0, 1]', '[0, ∞)']
        }
        st.dataframe(pd.DataFrame(properties_data))

# Footer and data download
st.sidebar.markdown("---")
st.sidebar.subheader("Download Data")
csv = commodity_data.to_csv(index=False)
st.sidebar.download_button(
    label="Download Commodity Data (CSV)",
    data=csv,
    file_name="global_commodity_prices_1994_2023.csv",
    mime="text/csv"
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Data Source:** Synthetic data representing realistic global commodity price trends
**Period:** 1994-2023 (Monthly data)
**Coverage:** 11 countries, 4 commodities
""")
