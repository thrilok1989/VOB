import streamlit as st
import pandas as pd
import requests
import time

st.set_page_config(page_title="Nifty Option Screener", layout="wide")

# -----------------------------
# 🔄 AUTO REFRESH 1 MIN
# -----------------------------
def auto_refresh(interval_sec=60):
    if "last_refresh" not in st.session_state:
        st.session_state["last_refresh"] = time.time()
    if time.time() - st.session_state["last_refresh"] > interval_sec:
        st.session_state["last_refresh"] = time.time()
        st.rerun()

auto_refresh(60)

# -----------------------------
# 🎯 TITLE
# -----------------------------
st.title("📊 NIFTY Option Screener – Fijacapital")
st.markdown("⏰ Auto-refresh every 1 minutes | 🔄 Live NSE Option Chain Analysis")

# -----------------------------
# 📡 FETCH OPTION CHAIN
# -----------------------------
@st.cache_data(ttl=180)
def fetch_option_chain():
    url = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        s = requests.Session()
        s.get("https://www.nseindia.com", headers=headers)
        r = s.get(url, headers=headers, timeout=5)
        return r.json()
    except:
        st.warning("⚠️ NSE blocked or data unavailable.")
        return None

data = fetch_option_chain()
if data is None:
    st.stop()

raw = data["records"]["data"]
expiries = data["records"]["expiryDates"]
spot = data["records"]["underlyingValue"]

# -----------------------------
# 🗓 SELECT EXPIRY
# -----------------------------
expiry = st.selectbox("📅 Select Expiry", expiries)

# -----------------------------
# 📑 FILTER DATA FOR SELECTED EXPIRY
# -----------------------------
ce_list = []
pe_list = []

for item in raw:
    if item.get("expiryDate") == expiry:
        strike = item.get("strikePrice", 0)

        if "CE" in item:
            ce = item["CE"]
            ce_list.append({
                "strikePrice": strike,
                "OI_CE": ce.get("openInterest", 0),
                "Chg_OI_CE": ce.get("changeinOpenInterest", 0),
                "Vol_CE": ce.get("totalTradedVolume", 0)
            })

        if "PE" in item:
            pe = item["PE"]
            pe_list.append({
                "strikePrice": strike,
                "OI_PE": pe.get("openInterest", 0),
                "Chg_OI_PE": pe.get("changeinOpenInterest", 0),
                "Vol_PE": pe.get("totalTradedVolume", 0)
            })

df_ce = pd.DataFrame(ce_list)
df_pe = pd.DataFrame(pe_list)

# -----------------------------
# 🎯 ATM STRIKE
# -----------------------------
atm_strike = min(df_ce["strikePrice"], key=lambda x: abs(x - spot))

# STRIKE GAP
strike_gap = df_ce["strikePrice"].sort_values().diff().mode()[0]

# ATM ± 8 strike gaps
lower = atm_strike - (8 * strike_gap)
upper = atm_strike + (8 * strike_gap)

df_ce = df_ce[(df_ce["strikePrice"] >= lower) & (df_ce["strikePrice"] <= upper)]
df_pe = df_pe[(df_pe["strikePrice"] >= lower) & (df_pe["strikePrice"] <= upper)]

# -----------------------------
# 📉 PCR
# -----------------------------
total_ce_oi = df_ce["OI_CE"].sum()
total_pe_oi = df_pe["OI_PE"].sum()
pcr = round(total_pe_oi / total_ce_oi, 2) if total_ce_oi > 0 else 0

col1, col2 = st.columns(2)
with col1:
    st.metric("📉 Put Call Ratio", pcr)
with col2:
    if pcr > 1.2:
        st.success("🟢 Bullish")
    elif pcr < 0.8:
        st.error("🔴 Bearish")
    else:
        st.warning("🟡 Neutral")

# -----------------------------
# 🔀 MERGE CE + PE
# -----------------------------
df = pd.merge(df_ce, df_pe, on="strikePrice", how="outer").sort_values("strikePrice")

# -----------------------------
# 🔥 Winding & Unwinding
# -----------------------------
df["CE_Winding"] = df["Chg_OI_CE"].apply(lambda x: "🟢 Winding" if x > 0 else "🔴 Unwinding")
df["PE_Winding"] = df["Chg_OI_PE"].apply(lambda x: "🟢 Winding" if x > 0 else "🔴 Unwinding")

# -----------------------------
# 🖥 DISPLAY TABLE
# -----------------------------
st.markdown("### 🧾 Combined Option Chain with Winding/Unwinding")
st.caption(f"Spot: {spot} | ATM: {atm_strike} | Range: {lower} → {upper}")

st.dataframe(df, use_container_width=True)

# -----------------------------
# 🔍 BREAKOUT ZONES
# -----------------------------
st.markdown("### 🔍 Breakout Zones")

top_ce = df_ce.sort_values("Chg_OI_CE", ascending=False).head(3)
top_pe = df_pe.sort_values("Chg_OI_PE", ascending=False).head(3)

col1, col2 = st.columns(2)
with col1:
    st.subheader("🚀 CALL Breakout")
    for _, r in top_ce.iterrows():
        st.success(f"{r['strikePrice']} → OI↑ {r['Chg_OI_CE']} | Vol {r['Vol_CE']}")

with col2:
    st.subheader("🔻 PUT Breakout")
    for _, r in top_pe.iterrows():
        st.success(f"{r['strikePrice']} → OI↑ {r['Chg_OI_PE']} | Vol {r['Vol_PE']}")

# -----------------------------
# 🛑 SUPPORT / RESISTANCE
# -----------------------------
df_ce["score"] = df_ce["Chg_OI_CE"] + df_ce["Vol_CE"]
df_pe["score"] = df_pe["Chg_OI_PE"] + df_pe["Vol_PE"]

resistance = df_ce.sort_values("score", ascending=False).iloc[0]["strikePrice"]
support = df_pe.sort_values("score", ascending=False).iloc[0]["strikePrice"]

st.markdown("### 🛑📈 Support & Resistance")
col1, col2 = st.columns(2)
with col1:
    st.error(f"Support: {int(support)}")
with col2:
    st.success(f"Resistance: {int(resistance)}")

# -----------------------------
# 🤖 AUTO TRADE SUGGESTION
# -----------------------------
st.markdown("### 🤖 Auto Trade Suggestion")

if pcr < 0.8:
    st.error("🟥 BUY PUT – Bearish Market")
elif pcr > 1.2:
    st.success("🟩 BUY CALL – Bullish Market")
else:
    st.warning("🔄 WAIT – Neutral Zone")
