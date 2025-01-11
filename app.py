import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from data_fetcher import fetch_current_market_data

# Page configuration
st.set_page_config(
    page_title="Cassandra",
    page_icon="🔮",
    layout="wide"
)

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load('models/best_model.pkl')

def analyze_market(model, market_data):
    crash_prob = model.predict_proba(market_data.reshape(1, -1))[0][1]
    
    if crash_prob < 0.3:
        risk_level = "LOW"
        recommendation = "Consider maintaining full market exposure"
        color = "green"
    elif crash_prob < 0.6:
        risk_level = "MEDIUM"
        recommendation = "Consider reducing position size to 50%"
        color = "yellow"
    else:
        risk_level = "HIGH"
        recommendation = "Consider moving to cash"
        color = "red"
        
    return crash_prob, risk_level, recommendation, color

def main():
    # Main title and explanation
    st.title("🔮 Cassandra")
    
    st.markdown("""
    ### What is Cassandra?
    Named after the Greek prophet, Cassandra is your market oracle for detecting potential crashes and anomalies. 
    Using advanced machine learning, she analyzes multiple market indicators to foresee potential market downturns.
    
    ### How Cassandra Works:
    1. Feed her the current market indicators from the sidebar
    2. Let her analyze market conditions
    3. Receive prophetic insights about market risks
    """)
    
    # Load model
    model = load_model()
    
    # Create feature groups FIRST
    features = model.feature_names_in_
    market_indices = [f for f in features if any(x in f for x in ['VIX', 'DXY', 'BDIY', 'MXEU', 'MXRU', 'MXIN'])]
    rates = [f for f in features if any(x in f for x in ['USGG', 'GTTL', 'US0001M', 'GTITL', 'GTJPY', 'GTGBP'])]
    etfs = [f for f in features if any(x in f for x in ['LF98TRUU', 'LG30TRUU', 'LP01TREU'])]
    currencies = [f for f in features if any(x in f for x in ['JPY', 'ECSURPUS'])]
    
    # Define sections BEFORE the button
    sections = [
        ("Market Indices", market_indices),
        ("Interest Rates", rates),
        ("ETFs", etfs),
        ("Currency Rates", currencies)
    ]
    
    # Sidebar explanations and inputs
    st.sidebar.header("📊 Market Indicators")
    
    # NOW add the button
    if st.sidebar.button("📥 Load Current Market Data", help="Automatically fetch today's market data"):
        with st.spinner("Fetching market data..."):
            current_data = fetch_current_market_data()
            for section_name, feature_list in sections:
                for feature in feature_list:
                    if feature in current_data:
                        st.session_state[f"{section_name.lower().replace(' ', '_')}_{feature}"] = current_data[feature]
    
    # Then add the explanation text
    st.sidebar.markdown("""
    Enter the current values for these market indicators. 
    These help assess market conditions and potential risks.
    """)
    
    # Create two columns in sidebar for organized inputs
    feature_values = {}
    features = model.feature_names_in_
    
    # Create a dictionary of default values based on recent market data
    default_values = {
        # Market Indices
        'BDIY': 12126.20,  # Baltic Dry Index
        'DXY': 72.54,      # Dollar Index
        'VIX': 17.57,      # Volatility Index
        'MXEU': 53.95,     # MSCI Europe
        'MXRU': 8.06,      # MSCI Russia
        'MXIN': 51.21,     # MSCI India
        
        # Currency Rates
        'JPY': 102.70,     # Japanese Yen to USD
        
        # Interest Rates
        'USGG30YR': 4.67,  # 30-Year Treasury Rate
        'USGG2YR': 2.24,   # 2-Year Treasury Rate
        'USGG3M': 1.89,    # 3-Month Treasury Rate
        'US0001M': 2.46,   # 1-Month LIBOR
        'GTITL10YR': 4.63, # Italy 10Y
        'GTITL2YR': 4.27,  # Italy 2Y
        'GTJPY10YR': 1.67, # Japan 10Y
        'GTJPY2YR': 0.80,  # Japan 2Y
        'GTGBP30Y': 4.50,  # UK 30Y
    }
    
    for section_name, feature_list in sections:
        st.sidebar.subheader(section_name)
        for feature in feature_list:
            # Create a truly unique key by combining section and feature
            unique_key = f"{section_name.lower().replace(' ', '_')}_{feature}"
            feature_values[feature] = st.sidebar.number_input(
                f"{feature}",
                help=f"Enter current {feature} value",
                value=default_values.get(feature, 0.0),  # Use default value if available, otherwise 0.0
                format="%.4f",
                key=unique_key
            )
    
    # Analysis button with better visibility
    if st.sidebar.button("🔍 Analyze Market", use_container_width=True):
        st.markdown("---")
        st.markdown("## Analysis Results")
        
        # Prepare and analyze data
        market_data = np.array([feature_values[f] for f in model.feature_names_in_])
        crash_prob, risk_level, recommendation, color = analyze_market(model, market_data)
        
        # Display results in an organized layout
        col1, col2, col3 = st.columns([1,1,2])
        
        with col1:
            st.metric("Crash Probability", f"{crash_prob:.1%}")
        
        with col2:
            st.markdown(f"### Risk Level")
            st.markdown(f":{color}[{risk_level}]")
        
        with col3:
            st.markdown("### Recommendation")
            st.markdown(f"_{recommendation}_")
        
        # Visualization
        st.markdown("### Risk Assessment Visualization")
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.barh([0], [crash_prob], color=color)
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel("Crash Probability")
        
        # Add reference lines
        ax.axvline(x=0.3, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0.6, color='gray', linestyle='--', alpha=0.5)
        ax.text(0.15, 0.5, 'Low Risk', ha='center', va='center')
        ax.text(0.45, 0.5, 'Medium Risk', ha='center', va='center')
        ax.text(0.8, 0.5, 'High Risk', ha='center', va='center')
        
        st.pyplot(fig)

    st.write("Expected features:", model.feature_names_in_)

if __name__ == "__main__":
    main()