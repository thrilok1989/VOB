import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime
import time

st.title("🔍 Data Fetch Debug Test")

# Test 1: Check Yahoo Finance directly
st.header("Test 1: Direct Yahoo Finance Test")

if st.button("Test India VIX"):
    with st.spinner("Testing India VIX..."):
        try:
            ticker = yf.Ticker("^INDIAVIX")
            st.write(f"Ticker created: {ticker}")
            
            hist = ticker.history(period="1d", interval="1m")
            st.write(f"Data fetched. Rows: {len(hist)}")
            
            if not hist.empty:
                st.success("✅ India VIX data fetched successfully!")
                st.dataframe(hist.tail())
                st.write(f"Latest VIX Value: {hist['Close'].iloc[-1]:.2f}")
            else:
                st.error("❌ Data is empty")
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

st.divider()

# Test 2: Check Nifty data
st.header("Test 2: Nifty 50 Test")

if st.button("Test Nifty 50"):
    with st.spinner("Testing Nifty 50..."):
        symbols_to_try = ["^NSEI", "NSEI", "^NSEBANK"]
        
        for symbol in symbols_to_try:
            st.write(f"Trying symbol: {symbol}")
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="5d", interval="5m")
                
                if not hist.empty:
                    st.success(f"✅ {symbol} - Fetched {len(hist)} candles")
                    st.dataframe(hist.tail())
                    break
                else:
                    st.warning(f"⚠️ {symbol} - Empty data")
                    
            except Exception as e:
                st.error(f"❌ {symbol} - Error: {str(e)}")
            
            time.sleep(2)

st.divider()

# Test 3: Check Sector Indices
st.header("Test 3: Sector Indices Test")

if st.button("Test Sector Indices"):
    sectors = {
        '^CNXIT': 'NIFTY IT',
        '^CNXBANK': 'NIFTY BANK',
        '^CNXAUTO': 'NIFTY AUTO'
    }
    
    results = []
    
    for symbol, name in sectors.items():
        st.write(f"Testing {name} ({symbol})...")
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="2d", interval="1d")
            
            if len(hist) >= 2:
                last_price = hist['Close'].iloc[-1]
                prev_price = hist['Close'].iloc[-2]
                change_pct = ((last_price - prev_price) / prev_price) * 100
                
                results.append({
                    'Sector': name,
                    'Symbol': symbol,
                    'Price': f"₹{last_price:.2f}",
                    'Change %': f"{change_pct:+.2f}%",
                    'Status': '✅'
                })
                
            else:
                results.append({
                    'Sector': name,
                    'Symbol': symbol,
                    'Price': 'N/A',
                    'Change %': 'N/A',
                    'Status': '⚠️ Empty'
                })
                
        except Exception as e:
            results.append({
                'Sector': name,
                'Symbol': symbol,
                'Price': 'N/A',
                'Change %': 'N/A',
                'Status': f'❌ {str(e)[:30]}'
            })
        
        time.sleep(1)
    
    st.dataframe(pd.DataFrame(results))

st.divider()

# System Info
st.header("System Information")

import sys
st.write(f"Python Version: {sys.version}")

try:
    import yfinance
    st.write(f"yfinance Version: {yfinance.__version__}")
except:
    st.error("yfinance version not available")

# Check internet connection
st.subheader("Internet Connection Test")
if st.button("Test Connection"):
    try:
        import socket
        socket.create_connection(("www.google.com", 80), timeout=3)
        st.success("✅ Internet connection OK")
    except OSError:
        st.error("❌ No internet connection")
    
    try:
        response = st.session_state.get('test_response')
        import requests
        response = requests.get("https://query1.finance.yahoo.com/v8/finance/chart/^GSPC", timeout=5)
        st.write(f"Yahoo Finance API Status: {response.status_code}")
        if response.status_code == 200:
            st.success("✅ Yahoo Finance API accessible")
        else:
            st.warning(f"⚠️ Yahoo Finance returned status: {response.status_code}")
    except Exception as e:
        st.error(f"❌ Cannot reach Yahoo Finance: {str(e)}")

st.divider()

# Display current time
st.write(f"Current Time (UTC): {datetime.utcnow()}")
st.write(f"Current Time (Local): {datetime.now()}")