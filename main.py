import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
from datetime import datetime, timedelta
import io

# Set page config
st.set_page_config(page_title="Continuous Distribution Analysis", layout="wide", page_icon="📊")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Select Page", [
    "🏠 Home",
    "📥 Data Collection",
    "📊 Distribution Parameters",
    "📈 Distribution Visualization",
    "📉 Standard Distributions"
])

# ============================================
# DATA GENERATION FUNCTIONS
# ============================================

def generate_commodity_data():
    """Generate realistic commodity price data"""
    np.random.seed(42)
    countries = ['India', 'USA', 'UK', 'Germany', 'France', 'Japan', 'China', 
                 'Brazil', 'Australia', 'Canada', 'South Africa']
    commodities = ['Petrol', 'Diesel', 'Gold', 'Silver']
    
    # Base prices (in local currency units)
    base_prices = {
        'Petrol': {'India': 95, 'USA': 1.2, 'UK': 1.5, 'Germany': 1.6, 'France': 1.5,
                   'Japan': 150, 'China': 7.5, 'Brazil': 6.5, 'Australia': 1.7, 
                   'Canada': 1.4, 'South Africa': 20},
        'Diesel': {'India': 85, 'USA': 1.1, 'UK': 1.4, 'Germany': 1.4, 'France': 1.3,
                   'Japan': 130, 'China': 6.8, 'Brazil': 5.8, 'Australia': 1.6, 
                   'Canada': 1.3, 'South Africa': 18},
        'Gold': {'India': 5500, 'USA': 1800, 'UK': 1400, 'Germany': 1650, 'France': 1650,
                 'Japan': 7500, 'China': 380, 'Brazil': 320, 'Australia': 2500, 
                 'Canada': 2300, 'South Africa': 28000},
        'Silver': {'India': 70, 'USA': 23, 'UK': 18, 'Germany': 21, 'France': 21,
                   'Japan': 95, 'China': 5, 'Brazil': 4, 'Australia': 32, 
                   'Canada': 29, 'South Africa': 350}
    }
    
    data = []
    start_year = datetime.now().year - 30
    
    for year in range(start_year, start_year + 30):
        year_factor = (year - start_year) / 30
        for commodity in commodities:
            for country in countries:
                base = base_prices[commodity][country]
                trend = base * (1 + year_factor * np.random.uniform(0.5, 1.5))
                seasonal = np.random.normal(0, base * 0.1)
                price = max(0, trend + seasonal)
                
                data.append({
                    'Year': year,
                    'Country': country,
                    'Commodity': commodity,
                    'Price': round(price, 2)
                })
    
    return pd.DataFrame(data)

def generate_rainfall_data():
    """Generate realistic rainfall data for Indian states"""
    np.random.seed(43)
    states = ['Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh',
              'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jharkhand', 
              'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur',
              'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Punjab', 'Rajasthan',
              'Sikkim', 'Tamil Nadu', 'Telangana', 'Tripura', 'Uttar Pradesh',
              'Uttarakhand', 'West Bengal']
    
    seasons = ['Winter', 'Summer', 'Monsoon', 'Post-Monsoon']
    
    # Average rainfall by state (mm)
    avg_rainfall = {
        'Meghalaya': 2500, 'Arunachal Pradesh': 2200, 'Assam': 2000, 'Kerala': 1800,
        'Karnataka': 1200, 'Maharashtra': 1100, 'Tamil Nadu': 950, 'West Bengal': 1500,
        'Odisha': 1400, 'Andhra Pradesh': 900, 'Telangana': 850, 'Gujarat': 700,
        'Madhya Pradesh': 900, 'Chhattisgarh': 1200, 'Jharkhand': 1100, 'Bihar': 1000,
        'Uttar Pradesh': 800, 'Himachal Pradesh': 1200, 'Uttarakhand': 1400,
        'Punjab': 600, 'Haryana': 500, 'Rajasthan': 400, 'Goa': 2500,
        'Manipur': 1800, 'Mizoram': 1900, 'Nagaland': 1700, 'Sikkim': 2000,
        'Tripura': 1600
    }
    
    # Season factors
    season_factors = {
        'Winter': 0.15, 'Summer': 0.20, 'Monsoon': 0.50, 'Post-Monsoon': 0.15
    }
    
    data = []
    start_year = datetime.now().year - 25
    
    for year in range(start_year, start_year + 25):
        for state in states:
            avg = avg_rainfall[state]
            for season in seasons:
                base_rainfall = avg * season_factors[season]
                variation = np.random.gamma(shape=2, scale=base_rainfall/2)
                rainfall = max(0, variation)
                
                data.append({
                    'Year': year,
                    'State': state,
                    'Season': season,
                    'Rainfall_mm': round(rainfall, 2)
                })
    
    return pd.DataFrame(data)

# ============================================
# DISTRIBUTION FUNCTIONS
# ============================================

def calculate_distribution_prob(dist_type, params, x_value):
    """Calculate probability for different distributions"""
    try:
        if dist_type == "Normal":
            mu, sigma = params['mu'], params['sigma']
            pdf = stats.norm.pdf(x_value, mu, sigma)
            cdf = stats.norm.cdf(x_value, mu, sigma)
            return pdf, cdf
        
        elif dist_type == "Exponential":
            lambda_param = params['lambda']
            pdf = stats.expon.pdf(x_value, scale=1/lambda_param)
            cdf = stats.expon.cdf(x_value, scale=1/lambda_param)
            return pdf, cdf
        
        elif dist_type == "Gamma":
            alpha, beta = params['alpha'], params['beta']
            pdf = stats.gamma.pdf(x_value, alpha, scale=1/beta)
            cdf = stats.gamma.cdf(x_value, alpha, scale=1/beta)
            return pdf, cdf
        
        elif dist_type == "Uniform":
            a, b = params['a'], params['b']
            pdf = stats.uniform.pdf(x_value, a, b-a)
            cdf = stats.uniform.cdf(x_value, a, b-a)
            return pdf, cdf
        
    except Exception as e:
        return None, None

def fit_distribution_to_data(data, dist_type):
    """Fit distribution to data and return parameters"""
    data = np.array(data).flatten()
    data = data[~np.isnan(data)]
    
    if dist_type == "Normal":
        mu, sigma = stats.norm.fit(data)
        return {'mu': mu, 'sigma': sigma}
    
    elif dist_type == "Exponential":
        loc, scale = stats.expon.fit(data)
        return {'lambda': 1/scale}
    
    elif dist_type == "Gamma":
        alpha, loc, scale = stats.gamma.fit(data, floc=0)
        return {'alpha': alpha, 'beta': 1/scale}
    
    elif dist_type == "Uniform":
        a, b = data.min(), data.max()
        return {'a': a, 'b': b}

def plot_distribution(dist_type, params, data=None, title="Distribution"):
    """Create distribution plot"""
    fig = go.Figure()
    
    # Generate x range
    if dist_type == "Normal":
        mu, sigma = params['mu'], params['sigma']
        x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
        y_pdf = stats.norm.pdf(x, mu, sigma)
        y_cdf = stats.norm.cdf(x, mu, sigma)
        
    elif dist_type == "Exponential":
        lambda_param = params['lambda']
        x = np.linspace(0, 5/lambda_param, 1000)
        y_pdf = stats.expon.pdf(x, scale=1/lambda_param)
        y_cdf = stats.expon.cdf(x, scale=1/lambda_param)
        
    elif dist_type == "Gamma":
        alpha, beta = params['alpha'], params['beta']
        x = np.linspace(0, (alpha + 3*np.sqrt(alpha))/beta, 1000)
        y_pdf = stats.gamma.pdf(x, alpha, scale=1/beta)
        y_cdf = stats.gamma.cdf(x, alpha, scale=1/beta)
        
    elif dist_type == "Uniform":
        a, b = params['a'], params['b']
        x = np.linspace(a - 0.1*(b-a), b + 0.1*(b-a), 1000)
        y_pdf = stats.uniform.pdf(x, a, b-a)
        y_cdf = stats.uniform.cdf(x, a, b-a)
    
    # Add PDF
    fig.add_trace(go.Scatter(x=x, y=y_pdf, name='PDF', line=dict(color='blue', width=2)))
    
    # Add CDF
    fig.add_trace(go.Scatter(x=x, y=y_cdf, name='CDF', line=dict(color='red', width=2)))
    
    # Add histogram if data provided
    if data is not None:
        fig.add_trace(go.Histogram(x=data, name='Data', opacity=0.5, 
                                    histnorm='probability density', nbinsx=30))
    
    fig.update_layout(
        title=title,
        xaxis_title='Value',
        yaxis_title='Probability',
        hovermode='x unified',
        height=500
    )
    
    return fig

# ============================================
# PAGE: HOME
# ============================================

if page == "🏠 Home":
    st.markdown('<h1 class="main-header">📊 Continuous Distribution Analysis System</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ## Welcome to the Continuous Distribution Analysis Application
    
    This comprehensive application provides tools for:
    
    ### 📥 Data Collection
    - **Commodity Prices**: Petrol, Diesel, Gold, Silver prices across 11 countries (30 years)
    - **Rainfall Data**: Seasonal rainfall data for all Indian states (25 years)
    
    ### 📊 Distribution Analysis
    - Fit continuous distributions to collected data
    - Calculate probability values from distribution parameters
    - Visualize distribution characteristics
    
    ### 📈 Supported Distributions
    1. **Normal Distribution** - Bell-shaped, symmetric
    2. **Exponential Distribution** - Memoryless property
    3. **Gamma Distribution** - Flexible shape parameter
    4. **Uniform Distribution** - Equal probability
    5. **Standard Distributions** - Normalized versions
    
    ### 🎯 Features
    - Interactive parameter input
    - Real-time probability calculations
    - Advanced visualizations with Plotly
    - PDF and CDF analysis
    - Statistical summaries
    
    ---
    
    **👈 Use the sidebar to navigate between pages**
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("📥 **30 Years** of Commodity Data")
    with col2:
        st.info("🌧️ **25 Years** of Rainfall Data")
    with col3:
        st.info("📊 **5 Distribution** Types")

# ============================================
# PAGE: DATA COLLECTION
# ============================================

elif page == "📥 Data Collection":
    st.markdown('<h1 class="main-header">📥 Data Collection</h1>', unsafe_allow_html=True)
    
    data_type = st.selectbox("Select Dataset", ["Commodity Prices", "Rainfall Data"])
    
    if data_type == "Commodity Prices":
        st.markdown("### 💰 Commodity Price Data (30 Years)")
        st.markdown("""
        **Data Includes:**
        - **Commodities**: Petrol, Diesel, Gold, Silver
        - **Countries**: India, USA, UK, Germany, France, Japan, China, Brazil, Australia, Canada, South Africa
        - **Period**: Last 30 years
        - **Prices**: In local currency units
        """)
        
        if st.button("🔄 Generate Commodity Data"):
            with st.spinner("Generating data..."):
                df = generate_commodity_data()
                st.session_state['commodity_data'] = df
                st.success("✅ Data generated successfully!")
        
        if 'commodity_data' in st.session_state:
            df = st.session_state['commodity_data']
            
            st.markdown("#### 📊 Data Overview")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Records", len(df))
            col2.metric("Countries", df['Country'].nunique())
            col3.metric("Commodities", df['Commodity'].nunique())
            col4.metric("Years", df['Year'].nunique())
            
            # Filters
            st.markdown("#### 🔍 Filters")
            col1, col2, col3 = st.columns(3)
            with col1:
                selected_commodity = st.multiselect("Commodity", df['Commodity'].unique(), 
                                                    default=df['Commodity'].unique())
            with col2:
                selected_country = st.multiselect("Country", df['Country'].unique(), 
                                                 default=df['Country'].unique()[:3])
            with col3:
                year_range = st.slider("Year Range", 
                                      int(df['Year'].min()), 
                                      int(df['Year'].max()),
                                      (int(df['Year'].min()), int(df['Year'].max())))
            
            # Filter data
            filtered_df = df[
                (df['Commodity'].isin(selected_commodity)) &
                (df['Country'].isin(selected_country)) &
                (df['Year'] >= year_range[0]) &
                (df['Year'] <= year_range[1])
            ]
            
            # Display data
            st.markdown("#### 📋 Filtered Data")
            st.dataframe(filtered_df, use_container_width=True, height=300)
            
            # Visualization
            st.markdown("#### 📈 Price Trends")
            fig = px.line(filtered_df, x='Year', y='Price', color='Country', 
                         facet_col='Commodity', facet_col_wrap=2,
                         title='Commodity Price Trends Over Time')
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            st.markdown("#### 📊 Statistical Summary")
            st.dataframe(filtered_df.groupby('Commodity')['Price'].describe(), 
                        use_container_width=True)
            
            # Download
            csv = filtered_df.to_csv(index=False)
            st.download_button("📥 Download CSV", csv, "commodity_data.csv", "text/csv")
    
    else:  # Rainfall Data
        st.markdown("### 🌧️ Rainfall Data (25 Years)")
        st.markdown("""
        **Data Includes:**
        - **States**: All 28 Indian states
        - **Seasons**: Winter, Summer, Monsoon, Post-Monsoon
        - **Period**: Last 25 years
        - **Measurement**: Rainfall in millimeters (mm)
        """)
        
        if st.button("🔄 Generate Rainfall Data"):
            with st.spinner("Generating data..."):
                df = generate_rainfall_data()
                st.session_state['rainfall_data'] = df
                st.success("✅ Data generated successfully!")
        
        if 'rainfall_data' in st.session_state:
            df = st.session_state['rainfall_data']
            
            st.markdown("#### 📊 Data Overview")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Records", len(df))
            col2.metric("States", df['State'].nunique())
            col3.metric("Seasons", df['Season'].nunique())
            col4.metric("Years", df['Year'].nunique())
            
            # Filters
            st.markdown("#### 🔍 Filters")
            col1, col2, col3 = st.columns(3)
            with col1:
                selected_state = st.multiselect("State", df['State'].unique(), 
                                               default=df['State'].unique()[:5])
            with col2:
                selected_season = st.multiselect("Season", df['Season'].unique(), 
                                                default=df['Season'].unique())
            with col3:
                year_range = st.slider("Year Range", 
                                      int(df['Year'].min()), 
                                      int(df['Year'].max()),
                                      (int(df['Year'].min()), int(df['Year'].max())))
            
            # Filter data
            filtered_df = df[
                (df['State'].isin(selected_state)) &
                (df['Season'].isin(selected_season)) &
                (df['Year'] >= year_range[0]) &
                (df['Year'] <= year_range[1])
            ]
            
            # Display data
            st.markdown("#### 📋 Filtered Data")
            st.dataframe(filtered_df, use_container_width=True, height=300)
            
            # Visualization
            st.markdown("#### 📈 Rainfall Patterns")
            fig = px.line(filtered_df, x='Year', y='Rainfall_mm', color='State', 
                         facet_col='Season', facet_col_wrap=2,
                         title='Rainfall Patterns Across Seasons')
            st.plotly_chart(fig, use_container_width=True)
            
            # Box plot
            st.markdown("#### 📊 Seasonal Distribution by State")
            fig2 = px.box(filtered_df, x='Season', y='Rainfall_mm', color='State',
                         title='Rainfall Distribution by Season')
            st.plotly_chart(fig2, use_container_width=True)
            
            # Statistics
            st.markdown("#### 📊 Statistical Summary")
            st.dataframe(filtered_df.groupby(['State', 'Season'])['Rainfall_mm'].describe(), 
                        use_container_width=True)
            
            # Download
            csv = filtered_df.to_csv(index=False)
            st.download_button("📥 Download CSV", csv, "rainfall_data.csv", "text/csv")

# ============================================
# PAGE: DISTRIBUTION PARAMETERS
# ============================================

elif page == "📊 Distribution Parameters":
    st.markdown('<h1 class="main-header">📊 Distribution Parameters & Probability Calculator</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    Enter distribution parameters to calculate probability values (PDF and CDF) for specific data points.
    """)
    
    # Select distribution type
    dist_type = st.selectbox("Select Distribution", 
                            ["Normal", "Exponential", "Gamma", "Uniform"])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎛️ Distribution Parameters")
        
        params = {}
        
        if dist_type == "Normal":
            st.markdown("**Normal Distribution: N(μ, σ²)**")
            params['mu'] = st.number_input("Mean (μ)", value=0.0, step=0.1)
            params['sigma'] = st.number_input("Standard Deviation (σ)", value=1.0, 
                                             min_value=0.01, step=0.1)
            st.latex(r"f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}")
        
        elif dist_type == "Exponential":
            st.markdown("**Exponential Distribution: Exp(λ)**")
            params['lambda'] = st.number_input("Rate Parameter (λ)", value=1.0, 
                                              min_value=0.01, step=0.1)
            st.latex(r"f(x) = \lambda e^{-\lambda x}, \quad x \geq 0")
        
        elif dist_type == "Gamma":
            st.markdown("**Gamma Distribution: Γ(α, β)**")
            params['alpha'] = st.number_input("Shape Parameter (α)", value=2.0, 
                                             min_value=0.01, step=0.1)
            params['beta'] = st.number_input("Rate Parameter (β)", value=1.0, 
                                            min_value=0.01, step=0.1)
            st.latex(r"f(x) = \frac{\beta^\alpha}{\Gamma(\alpha)} x^{\alpha-1} e^{-\beta x}")
        
        elif dist_type == "Uniform":
            st.markdown("**Uniform Distribution: U(a, b)**")
            params['a'] = st.number_input("Lower Bound (a)", value=0.0, step=0.1)
            params['b'] = st.number_input("Upper Bound (b)", value=1.0, step=0.1)
            if params['b'] <= params['a']:
                st.error("Upper bound must be greater than lower bound!")
            st.latex(r"f(x) = \frac{1}{b-a}, \quad a \leq x \leq b")
        
        x_value = st.number_input("Enter X value for probability calculation", 
                                 value=0.0, step=0.1)
        
        calculate_btn = st.button("🔢 Calculate Probabilities", type="primary")
    
    with col2:
        st.markdown("### 📊 Probability Results")
        
        if calculate_btn:
            pdf, cdf = calculate_distribution_prob(dist_type, params, x_value)
            
            if pdf is not None and cdf is not None:
                st.success("✅ Calculation Complete!")
                
                # Display results
                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    st.metric("PDF f(x)", f"{pdf:.6f}")
                with metric_col2:
                    st.metric("CDF F(x)", f"{cdf:.6f}")
                
                # Additional information
                st.markdown("---")
                st.markdown("#### 📝 Interpretation")
                st.info(f"""
                - **PDF (Probability Density Function)**: {pdf:.6f}
                  - Indicates the relative likelihood of the random variable at x = {x_value}
                
                - **CDF (Cumulative Distribution Function)**: {cdf:.6f}
                  - Probability that X ≤ {x_value}
                  - Probability that X > {x_value} = {1-cdf:.6f}
                """)
                
                # Theoretical properties
                st.markdown("#### 📐 Distribution Properties")
                if dist_type == "Normal":
                    mean = params['mu']
                    variance = params['sigma']**2
                    st.write(f"- Mean: {mean}")
                    st.write(f"- Variance: {variance:.4f}")
                    st.write(f"- Std Dev: {params['sigma']}")
                
                elif dist_type == "Exponential":
                    mean = 1/params['lambda']
                    variance = 1/(params['lambda']**2)
                    st.write(f"- Mean: {mean:.4f}")
                    st.write(f"- Variance: {variance:.4f}")
                    st.write(f"- Std Dev: {np.sqrt(variance):.4f}")
                
                elif dist_type == "Gamma":
                    mean = params['alpha']/params['beta']
                    variance = params['alpha']/(params['beta']**2)
                    st.write(f"- Mean: {mean:.4f}")
                    st.write(f"- Variance: {variance:.4f}")
                    st.write(f"- Std Dev: {np.sqrt(variance):.4f}")
                
                elif dist_type == "Uniform":
                    mean = (params['a'] + params['b'])/2
                    variance = ((params['b'] - params['a'])**2)/12
                    st.write(f"- Mean: {mean:.4f}")
                    st.write(f"- Variance: {variance:.4f}")
                    st.write(f"- Std Dev: {np.sqrt(variance):.4f}")
            else:
                st.error("❌ Error in calculation. Please check parameters.")
    
    # Visualization
    st.markdown("---")
    st.markdown("### 📈 Distribution Visualization")
    
    fig = plot_distribution(dist_type, params, title=f"{dist_type} Distribution")
    
    # Add vertical line at x_value
    if calculate_btn and pdf is not None:
        fig.add_vline(x=x_value, line_dash="dash", line_color="green", 
                     annotation_text=f"x = {x_value}")
    
    st.plotly_chart(fig, use_container_width=True)

# ============================================
# PAGE: DISTRIBUTION VISUALIZATION
# ============================================

elif page == "📈 Distribution Visualization":
    st.markdown('<h1 class="main-header">📈 Distribution Visualization with Data Fitting</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    Fit continuous distributions to your collected data and visualize the results.
    """)
    
    # Select dataset
    dataset = st.selectbox("Select Dataset", ["Commodity Prices", "Rainfall Data"])
    
    if dataset == "Commodity Prices":
        if 'commodity_data' not in st.session_state:
            st.warning("⚠️ Please generate commodity data first from the Data Collection page.")
            if st.button("🔄 Generate Data Now"):
                with st.spinner("Generating data..."):
                    df = generate_commodity_data()
                    st.session_state['commodity_data'] = df
                    st.success("✅ Data generated!")
                    st.rerun()
        else:
            df = st.session_state['commodity_data']
            
            col1, col2 = st.columns(2)
            with col1:
                commodity = st.selectbox("Select Commodity", df['Commodity'].unique())
            with col2:
                country = st.selectbox("Select Country", df['Country'].unique())
            
            # Filter data
            data = df[(df['Commodity'] == commodity) & (df['Country'] == country)]['Price'].values
            
            st.markdown(f"### 📊 Analyzing: {commodity} prices in {country}")
            
    else:  # Rainfall Data
        if 'rainfall_data' not in st.session_state:
            st.warning("⚠️ Please generate rainfall data first from the Data Collection page.")
            if st.button("🔄 Generate Data Now"):
                with st.spinner("Generating data..."):
                    df = generate_rainfall_data()
                    st.session_state['rainfall_data'] = df
                    st.success("✅ Data generated!")
                    st.rerun()
        else:
            df = st.session_state['rainfall_data']
            
            col1, col2 = st.columns(2)
            with col1:
                state = st.selectbox("Select State", df['State'].unique())
            with col2:
                season = st.selectbox("Select Season", df['Season'].unique())
            
            # Filter data
            data = df[(df['State'] == state) & (df['Season'] == season)]['Rainfall_mm'].values
            
            st.markdown(f"### 📊 Analyzing: {season} rainfall in {state}")
    
    if len(data) > 0:
        # Data statistics
        st.markdown("#### 📊 Data Statistics")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Count", len(data))
        col2.metric("Mean", f"{np.mean(data):.2f}")
        col3.metric("Std Dev", f"{np.std(data):.2f}")
        col4.metric("Min", f"{np.min(data):.2f}")
        col5.metric("Max", f"{np.max(data):.2f}")
        
        # Select distributions to fit
        st.markdown("#### 🎯 Select Distributions to Fit")
        dist_options = st.multiselect("Distributions", 
                                     ["Normal", "Exponential", "Gamma", "Uniform"],
                                     default=
