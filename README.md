# Cassandra - Market Anomaly Detection 🔮

Cassandra is an advanced market analysis tool that uses machine learning to detect potential market crashes and anomalies. Named after the Greek prophet, it analyzes multiple market indicators to provide risk assessments and trading recommendations.

## Features

- Real-time market data analysis
- Machine learning-based risk assessment
- Visual risk level indicators
- Automated market recommendations
- Support for multiple market indicators including:
  - Market Indices (VIX, DXY, etc.)
  - Interest Rates
  - ETFs
  - Currency Rates

## Installation

1. Clone the repository:

```bash
git clone <your-repository-url>
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

The application will open in your default web browser. You can:

1. Use the "Load Current Market Data" button to fetch real-time market indicators
2. Manually adjust market indicators in the sidebar
3. Click "Analyze Market" to get risk assessment and recommendations

## Model

The application uses a pre-trained machine learning model (`models/best_model.pkl`) that has been trained on historical market data to predict crash probabilities.

## Risk Levels

- **Low Risk** (< 30%): Consider maintaining full market exposure
- **Medium Risk** (30-60%): Consider reducing position size to 50%
- **High Risk** (> 60%): Consider moving to cash

## License

[MIT License](LICENSE)
