import yfinance as yf
import pandas as pd

def fetch_current_market_data():
    # map our features to Yahoo finance symbols
    symbols_map = {
        'VIX': '^VIX',  # Volatility Index
        'DXY': 'DX-Y.NYB',  # US Dollar Index
        'MXEU': 'IEUR',  # iShares MSCI Europe ETF as proxy
        'MXRU': 'ERUS',  # iShares MSCI Russia ETF
        'MXIN': 'INDA',  # iShares MSCI India ETF
        'USGG30YR': '^TYX',  # 30 Year Treasury Rate
        'USGG2YR': '^IRX',  # 2 Year Treasury Rate
        'JPY': 'JPY=X',       # Japanese Yen to USD exchange rate
        # Add other mappings
    }
    
    data = {}
    for our_symbol, yf_symbol in symbols_map.items():
        try:
            ticker = yf.Ticker(yf_symbol)
            current_data = ticker.history(period="1d")
            if not current_data.empty:
                data[our_symbol] = current_data['Close'].iloc[-1]
        except:
            data[our_symbol] = 0.0 # default value if fetch fails

    return data