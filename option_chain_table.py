"""
Option Chain Table Module
========================
Displays option chain data in a tabulated format using native Streamlit components
- ATM ±5 strikes (11 total strikes)
- Calls on left, Strike in middle, Puts on right
- Uses st.dataframe for display
- Highlights Support (Green) and Resistance (Red) levels
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import pytz

# IST timezone
IST = pytz.timezone('Asia/Kolkata')

# Constants
LOT_SIZE = 50  # NIFTY lot size


def safe_float(val, default=0.0):
    """Safe float conversion"""
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return default
        return float(val)
    except:
        return default


def safe_int(val, default=0):
    """Safe int conversion"""
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return default
        return int(val)
    except:
        return default


def format_oi_lakhs(value):
    """Format OI in lakhs"""
    try:
        if pd.isna(value) or value == 0:
            return "-"
        return f"{value/100000:.2f}"
    except:
        return "-"


def format_change_oi(value):
    """Format Change in OI"""
    try:
        if pd.isna(value) or value == 0:
            return "0"
        sign = "+" if value > 0 else ""
        if abs(value) >= 100000:
            return f"{sign}{value/100000:.1f}L"
        return f"{sign}{value/1000:.1f}K"
    except:
        return "0"


def format_volume(value):
    """Format volume"""
    try:
        if pd.isna(value) or value == 0:
            return "-"
        if value >= 10000000:
            return f"{value/10000000:.1f}Cr"
        elif value >= 100000:
            return f"{value/100000:.1f}L"
        elif value >= 1000:
            return f"{value/1000:.1f}K"
        return str(int(value))
    except:
        return "-"


def format_gex(value):
    """Format GEX"""
    try:
        if pd.isna(value) or value == 0:
            return "-"
        abs_val = abs(value)
        if abs_val >= 10000000:
            return f"{value/10000000:.1f}Cr"
        elif abs_val >= 100000:
            return f"{value/100000:.1f}L"
        elif abs_val >= 1000:
            return f"{value/1000:.1f}K"
        return f"{value:.1f}"
    except:
        return "-"


def calculate_max_pain_from_df(df):
    """Calculate Max Pain strike from dataframe"""
    try:
        strikes = df["strikePrice"].values
        max_pain_strike = strikes[0]
        min_loss = float('inf')

        for strike in strikes:
            strike_row = df[df["strikePrice"] == strike]
            if strike_row.empty:
                continue

            call_loss = sum(
                max(0, s - strike) * safe_int(df[df["strikePrice"] == s].iloc[0].get("OI_CE", 0))
                for s in strikes if not df[df["strikePrice"] == s].empty
            )
            put_loss = sum(
                max(0, strike - s) * safe_int(df[df["strikePrice"] == s].iloc[0].get("OI_PE", 0))
                for s in strikes if not df[df["strikePrice"] == s].empty
            )

            total_loss = call_loss + put_loss
            if total_loss < min_loss:
                min_loss = total_loss
                max_pain_strike = strike

        return int(max_pain_strike)
    except:
        return 0


def determine_sr_strength(strike, spot, chg_oi, oi_total, sr_type, vol=0):
    """
    Determine if a support/resistance level is BUILDING, HOLDING, or BREAKING

    BUILDING STRONG:
    - Positive ChgOI (new positions being written at this level)
    - High volume activity
    - Price not too close to the level

    HOLDING:
    - Stable OI (small positive or small negative ChgOI)
    - Price approaching but hasn't breached

    BREAKING/WEAKENING:
    - Negative ChgOI (positions being unwound/squared off)
    - Price very close to or crossing the level
    - Low volume (sellers/buyers losing conviction)

    Returns: (status_emoji, status_text, strength_score)
    """
    # Calculate distance from spot
    distance = abs(strike - spot)
    distance_pct = (distance / spot) * 100 if spot > 0 else 0

    # Relative ChgOI (as % of total OI)
    chg_oi_pct = (chg_oi / max(oi_total, 1)) * 100

    strength_score = 0

    # === Analyze Change in OI ===
    if chg_oi > 0:
        # Positive ChgOI = new positions being added = BUILDING
        if chg_oi_pct > 5:  # More than 5% of OI added
            strength_score += 3
        elif chg_oi_pct > 2:
            strength_score += 2
        else:
            strength_score += 1
    elif chg_oi < 0:
        # Negative ChgOI = positions being unwound = WEAKENING
        if chg_oi_pct < -5:  # More than 5% of OI removed
            strength_score -= 3
        elif chg_oi_pct < -2:
            strength_score -= 2
        else:
            strength_score -= 1

    # === Analyze Price Proximity ===
    if sr_type == "resistance":
        if spot > strike:
            # Price has breached resistance = BROKEN
            strength_score -= 3
        elif distance_pct < 0.3:
            # Price very close (within 0.3%) = Testing
            strength_score -= 1
        elif distance_pct > 1.0:
            # Price far away = Safe
            strength_score += 1
    else:  # support
        if spot < strike:
            # Price has breached support = BROKEN
            strength_score -= 3
        elif distance_pct < 0.3:
            # Price very close (within 0.3%) = Testing
            strength_score -= 1
        elif distance_pct > 1.0:
            # Price far away = Safe
            strength_score += 1

    # === Determine Status ===
    if strength_score >= 3:
        return "🔼", "BUILDING STRONG", strength_score
    elif strength_score >= 1:
        return "📈", "BUILDING", strength_score
    elif strength_score == 0:
        return "➖", "HOLDING", strength_score
    elif strength_score >= -2:
        return "📉", "WEAKENING", strength_score
    else:
        return "⚠️", "BREAKING", strength_score


def identify_support_resistance(df, spot, atm_strike):
    """
    Identify support and resistance levels based on OI, PCR, GEX
    Also determines if each level is BUILDING, HOLDING, or BREAKING

    RESISTANCE (price expected to face selling pressure):
    - High Call OI (sellers expect price to stay below)
    - High Call writing (positive Chg OI for calls)
    - Low PCR (< 0.7) - more call activity
    - Strike above spot with high call concentration

    SUPPORT (price expected to find buying interest):
    - High Put OI (sellers expect price to stay above)
    - High Put writing (positive Chg OI for puts)
    - High PCR (> 1.3) - more put activity
    - Strike below spot with high put concentration

    Returns dict with support and resistance strikes + strength status
    """
    support_levels = []
    resistance_levels = []

    # Get max OI values for comparison
    max_ce_oi = df["OI_CE"].max() if "OI_CE" in df.columns else 0
    max_pe_oi = df["OI_PE"].max() if "OI_PE" in df.columns else 0

    for _, row in df.iterrows():
        strike = int(row["strikePrice"])
        oi_ce = safe_int(row.get("OI_CE", 0))
        oi_pe = safe_int(row.get("OI_PE", 0))
        chg_oi_ce = safe_int(row.get("Chg_OI_CE", 0))
        chg_oi_pe = safe_int(row.get("Chg_OI_PE", 0))
        vol_ce = safe_int(row.get("Vol_CE", 0))
        vol_pe = safe_int(row.get("Vol_PE", 0))
        gex_ce = safe_float(row.get("GEX_CE", 0))
        gex_pe = safe_float(row.get("GEX_PE", 0))

        # Calculate strike PCR
        pcr = oi_pe / max(oi_ce, 1)

        # Resistance criteria (above or at spot with high call activity)
        resistance_score = 0
        if strike >= spot:
            # High Call OI (top 30%)
            if oi_ce >= max_ce_oi * 0.7:
                resistance_score += 3
            elif oi_ce >= max_ce_oi * 0.5:
                resistance_score += 2

            # Call writing (positive change)
            if chg_oi_ce > 0:
                resistance_score += 2

            # Low PCR indicates call dominance
            if pcr < 0.5:
                resistance_score += 2
            elif pcr < 0.8:
                resistance_score += 1

        # Support criteria (below or at spot with high put activity)
        support_score = 0
        if strike <= spot:
            # High Put OI (top 30%)
            if oi_pe >= max_pe_oi * 0.7:
                support_score += 3
            elif oi_pe >= max_pe_oi * 0.5:
                support_score += 2

            # Put writing (positive change)
            if chg_oi_pe > 0:
                support_score += 2

            # High PCR indicates put dominance
            if pcr > 1.5:
                support_score += 2
            elif pcr > 1.2:
                support_score += 1

        # Classify based on scores
        if resistance_score >= 4:
            # Determine strength (for resistance, use call data)
            str_emoji, str_text, str_score = determine_sr_strength(
                strike, spot, chg_oi_ce, oi_ce, "resistance", vol_ce
            )
            resistance_levels.append({
                "strike": strike,
                "score": resistance_score,
                "oi_ce": oi_ce,
                "chg_oi": chg_oi_ce,
                "pcr": pcr,
                "strength_emoji": str_emoji,
                "strength_text": str_text,
                "strength_score": str_score
            })

        if support_score >= 4:
            # Determine strength (for support, use put data)
            str_emoji, str_text, str_score = determine_sr_strength(
                strike, spot, chg_oi_pe, oi_pe, "support", vol_pe
            )
            support_levels.append({
                "strike": strike,
                "score": support_score,
                "oi_pe": oi_pe,
                "chg_oi": chg_oi_pe,
                "pcr": pcr,
                "strength_emoji": str_emoji,
                "strength_text": str_text,
                "strength_score": str_score
            })

    # Sort by score (highest first) and get top levels
    resistance_levels = sorted(resistance_levels, key=lambda x: x["score"], reverse=True)
    support_levels = sorted(support_levels, key=lambda x: x["score"], reverse=True)

    # Get primary support and resistance
    primary_resistance = resistance_levels[0]["strike"] if resistance_levels else None
    primary_support = support_levels[0]["strike"] if support_levels else None

    return {
        "resistance": [r["strike"] for r in resistance_levels[:3]],  # Top 3
        "support": [s["strike"] for s in support_levels[:3]],  # Top 3
        "primary_resistance": primary_resistance,
        "primary_support": primary_support,
        "resistance_details": resistance_levels[:3],
        "support_details": support_levels[:3]
    }


def render_option_chain_table_tab(merged_df, spot, atm_strike, strike_gap, expiry, days_to_expiry, tau):
    """
    Render the option chain table using native Streamlit components
    """
    st.subheader("Option Chain Table")
    st.caption("ATM ±5 strikes | Support (Green) | Resistance (Red)")

    # Controls row
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 2])

    with col1:
        num_strikes = st.selectbox(
            "Strikes ±ATM",
            [3, 5, 7],
            index=1,
            key="oc_table_strikes_selector"
        )

    # Filter to selected strikes around ATM
    lower_bound = atm_strike - (num_strikes * strike_gap)
    upper_bound = atm_strike + (num_strikes * strike_gap)

    df_filtered = merged_df[(merged_df["strikePrice"] >= lower_bound) &
                            (merged_df["strikePrice"] <= upper_bound)].copy()
    # Sort by strike price ascending (lower strikes at top, ATM in middle, higher at bottom)
    df_filtered = df_filtered.sort_values("strikePrice", ascending=True).reset_index(drop=True)

    if df_filtered.empty:
        st.error("No data available for option chain table")
        return

    with col2:
        st.metric("ATM", f"{atm_strike:,}")

    with col3:
        total_ce_oi = df_filtered["OI_CE"].sum()
        total_pe_oi = df_filtered["OI_PE"].sum()
        pcr = total_pe_oi / max(total_ce_oi, 1)
        pcr_emoji = "🟢" if pcr > 1.0 else "🔴" if pcr < 0.8 else "🟡"
        st.metric("PCR", f"{pcr:.2f} {pcr_emoji}")

    with col4:
        max_pain = calculate_max_pain_from_df(df_filtered)
        st.metric("Max Pain", f"{max_pain:,}")

    with col5:
        st.metric("NIFTY Spot", f"₹{spot:,.2f}", delta=f"Exp: {expiry} ({days_to_expiry:.1f}d)")

    # Identify Support and Resistance
    sr_levels = identify_support_resistance(df_filtered, spot, atm_strike)

    # Display Support and Resistance Summary with Strength
    st.divider()

    sr_col1, sr_col2 = st.columns(2)

    with sr_col1:
        st.success(f"🟢 **SUPPORT LEVELS**")
        if sr_levels["support_details"]:
            for i, detail in enumerate(sr_levels["support_details"]):
                marker = "→" if i == 0 else "  "
                primary = " (Primary)" if i == 0 else ""
                strength_emoji = detail.get("strength_emoji", "")
                strength_text = detail.get("strength_text", "")
                chg_oi = detail.get("chg_oi", 0)
                chg_sign = "+" if chg_oi > 0 else ""
                st.write(f"{marker} **{detail['strike']:,}**{primary}")
                st.caption(f"   {strength_emoji} {strength_text} | ChgOI: {chg_sign}{chg_oi:,}")
        else:
            st.write("No strong support identified")

    with sr_col2:
        st.error(f"🔴 **RESISTANCE LEVELS**")
        if sr_levels["resistance_details"]:
            for i, detail in enumerate(sr_levels["resistance_details"]):
                marker = "→" if i == 0 else "  "
                primary = " (Primary)" if i == 0 else ""
                strength_emoji = detail.get("strength_emoji", "")
                strength_text = detail.get("strength_text", "")
                chg_oi = detail.get("chg_oi", 0)
                chg_sign = "+" if chg_oi > 0 else ""
                st.write(f"{marker} **{detail['strike']:,}**{primary}")
                st.caption(f"   {strength_emoji} {strength_text} | ChgOI: {chg_sign}{chg_oi:,}")
        else:
            st.write("No strong resistance identified")

    st.divider()

    # Build display dataframe
    table_data = []

    for _, row in df_filtered.iterrows():
        strike = int(row["strikePrice"])
        is_atm = strike == atm_strike
        is_support = strike in sr_levels["support"]
        is_resistance = strike in sr_levels["resistance"]
        is_primary_support = strike == sr_levels["primary_support"]
        is_primary_resistance = strike == sr_levels["primary_resistance"]

        # CE values
        ltp_ce = safe_float(row.get("LTP_CE", 0))
        oi_ce = safe_int(row.get("OI_CE", 0))
        chg_oi_ce = safe_int(row.get("Chg_OI_CE", 0))
        vol_ce = safe_int(row.get("Vol_CE", 0))
        iv_ce = safe_float(row.get("IV_CE", 0))
        delta_ce = safe_float(row.get("Delta_CE", 0))

        # PE values
        ltp_pe = safe_float(row.get("LTP_PE", 0))
        oi_pe = safe_int(row.get("OI_PE", 0))
        chg_oi_pe = safe_int(row.get("Chg_OI_PE", 0))
        vol_pe = safe_int(row.get("Vol_PE", 0))
        iv_pe = safe_float(row.get("IV_PE", 0))
        delta_pe = safe_float(row.get("Delta_PE", 0))

        # Strike PCR
        strike_pcr = oi_pe / max(oi_ce, 1)

        # Market Depth Bias for CE (Call)
        # Positive ChgOI + High Volume = Strong selling (writers adding positions)
        # Negative ChgOI = Unwinding (sellers closing) = Weak
        ce_bias = ""
        if chg_oi_ce > 0:
            if vol_ce > oi_ce * 0.05:  # High activity
                ce_bias = "🐻S"  # Strong selling (bearish for calls = bullish for market)
            else:
                ce_bias = "🐻"
        elif chg_oi_ce < 0:
            if vol_ce > oi_ce * 0.05:
                ce_bias = "🐂U"  # Unwinding (bullish for calls = bearish for market)
            else:
                ce_bias = "🐂"
        else:
            ce_bias = "➖"

        # Market Depth Bias for PE (Put)
        # Positive ChgOI = Put writing = Bullish
        # Negative ChgOI = Put unwinding = Bearish
        pe_bias = ""
        if chg_oi_pe > 0:
            if vol_pe > oi_pe * 0.05:
                pe_bias = "🐂S"  # Strong put selling = Bullish
            else:
                pe_bias = "🐂"
        elif chg_oi_pe < 0:
            if vol_pe > oi_pe * 0.05:
                pe_bias = "🐻U"  # Put unwinding = Bearish
            else:
                pe_bias = "🐻"
        else:
            pe_bias = "➖"

        # Build markers
        markers = []
        if is_atm:
            markers.append("⭐ATM")
        if is_primary_support:
            markers.append("🟢SUP")
        elif is_support:
            markers.append("🟢")
        if is_primary_resistance:
            markers.append("🔴RES")
        elif is_resistance:
            markers.append("🔴")

        marker_str = " ".join(markers)

        table_data.append({
            # CE Side
            "CE_Bias": ce_bias,
            "CE_OI": format_oi_lakhs(oi_ce),
            "CE_ChgOI": format_change_oi(chg_oi_ce),
            "CE_Vol": format_volume(vol_ce),
            "CE_IV": f"{iv_ce:.1f}" if iv_ce > 0 else "-",
            "CE_LTP": f"{ltp_ce:.2f}",
            # Strike info
            "Strike": strike,
            "PCR": round(strike_pcr, 2),
            "Signal": marker_str,
            # PE Side
            "PE_LTP": f"{ltp_pe:.2f}",
            "PE_IV": f"{iv_pe:.1f}" if iv_pe > 0 else "-",
            "PE_Vol": format_volume(vol_pe),
            "PE_ChgOI": format_change_oi(chg_oi_pe),
            "PE_OI": format_oi_lakhs(oi_pe),
            "PE_Bias": pe_bias,
            # Hidden for styling
            "_is_support": is_support,
            "_is_resistance": is_resistance,
            "_is_atm": is_atm,
        })

    display_df = pd.DataFrame(table_data)

    # Create display version without hidden columns
    display_cols = ["CE_Bias", "CE_OI", "CE_ChgOI", "CE_Vol", "CE_IV", "CE_LTP",
                    "Strike", "PCR", "Signal",
                    "PE_LTP", "PE_IV", "PE_Vol", "PE_ChgOI", "PE_OI", "PE_Bias"]

    show_df = display_df[display_cols].copy()

    # Rename columns for display - use unique names to avoid Styler error
    show_df.columns = ["CE Bias", "CE OI(L)", "CE ChgOI", "CE Vol", "CE IV%", "CE LTP",
                       "Strike", "PCR", "Signal",
                       "PE LTP", "PE IV%", "PE Vol", "PE ChgOI", "PE OI(L)", "PE Bias"]

    # Style function for highlighting
    def highlight_rows(row):
        strike = row.name
        original_row = display_df.iloc[strike]

        styles = [''] * len(row)

        if original_row["_is_resistance"]:
            styles = ['background-color: rgba(255, 0, 0, 0.2)'] * len(row)
        elif original_row["_is_support"]:
            styles = ['background-color: rgba(0, 255, 0, 0.2)'] * len(row)
        elif original_row["_is_atm"]:
            styles = ['background-color: rgba(255, 215, 0, 0.2)'] * len(row)

        return styles

    # Apply styling
    styled_df = show_df.style.apply(highlight_rows, axis=1)

    # Display header
    st.write("**CALLS (CE)** ← | → **PUTS (PE)**")

    # Display the styled dataframe
    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True,
        height=min(500, (len(show_df) + 1) * 35)
    )

    # Legend
    st.caption(
        "🟢 = Support | 🔴 = Resistance | ⭐ = ATM | OI in Lakhs | PCR = Put/Call Ratio"
    )
    st.caption(
        "**Bias:** 🐂 = Bullish | 🐻 = Bearish | S = Strong | U = Unwinding | ➖ = Neutral"
    )
    st.caption(
        "**Strength:** 🔼 BUILDING STRONG | 📈 BUILDING | ➖ HOLDING | 📉 WEAKENING | ⚠️ BREAKING"
    )

    st.divider()

    # Buy section
    render_buy_section(df_filtered, spot, expiry, atm_strike)


def render_buy_section(df, spot, expiry, atm_strike):
    """Render the buy option section using native Streamlit components"""
    st.subheader("Quick Buy Option")

    col1, col2, col3 = st.columns([2, 2, 1])

    strikes = sorted(df["strikePrice"].unique())

    with col1:
        selected_strike = st.selectbox(
            "Strike",
            strikes,
            index=strikes.index(atm_strike) if atm_strike in strikes else 0,
            key="buy_strike_selector",
            format_func=lambda x: f"{int(x):,} {'(ATM)' if x == atm_strike else ''}"
        )

    with col2:
        option_type = st.selectbox(
            "Type",
            ["CE", "PE"],
            key="buy_option_type_selector"
        )

    with col3:
        lots = st.number_input(
            "Lots",
            min_value=1,
            max_value=100,
            value=1,
            key="buy_lots_input"
        )

    # Get current data
    strike_row = df[df["strikePrice"] == selected_strike]
    if not strike_row.empty:
        ltp_col = f"LTP_{option_type}"
        current_ltp = safe_float(strike_row.iloc[0].get(ltp_col, 0))
        iv_col = f"IV_{option_type}"
        current_iv = safe_float(strike_row.iloc[0].get(iv_col, 0))
        delta_col = f"Delta_{option_type}"
        current_delta = safe_float(strike_row.iloc[0].get(delta_col, 0))
    else:
        current_ltp = 0
        current_iv = 0
        current_delta = 0

    quantity = lots * LOT_SIZE
    est_cost = current_ltp * quantity

    # Order summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("LTP", f"₹{current_ltp:.2f}")
    with col2:
        st.metric("Quantity", f"{quantity}")
    with col3:
        st.metric("Est. Cost", f"₹{est_cost:,.0f}")
    with col4:
        st.metric("IV / Delta", f"{current_iv:.1f}% / {current_delta:.2f}")

    # Buy buttons
    col1, col2, col3 = st.columns([2, 2, 4])

    with col1:
        if st.button(
            f"BUY {option_type} @ Market",
            type="primary",
            use_container_width=True,
            key="buy_market_button"
        ):
            place_buy_order(selected_strike, option_type, lots, "MARKET", spot, expiry)

    with col2:
        limit_price = st.number_input(
            "Limit Price",
            min_value=0.05,
            value=float(current_ltp) if current_ltp > 0 else 10.0,
            step=0.05,
            key="limit_price_field"
        )
        if st.button(
            f"BUY @ ₹{limit_price:.2f}",
            use_container_width=True,
            key="buy_limit_button"
        ):
            place_buy_order(selected_strike, option_type, lots, "LIMIT", spot, expiry, limit_price)

    with col3:
        st.warning("Orders via Dhan API. Ensure sufficient margin.")


def place_buy_order(strike, option_type, lots, order_type, spot, expiry, limit_price=None):
    """Place buy order using Dhan API"""
    try:
        from dhan_api import DhanAPI
        from config import LOT_SIZES

        dhan = DhanAPI()
        quantity = lots * LOT_SIZES.get("NIFTY", 50)

        # Basic SL and target
        sl_offset = 30 if option_type == "CE" else -30
        target_offset = 50 if option_type == "CE" else -50

        sl_price = spot + sl_offset if option_type == "CE" else spot - abs(sl_offset)
        target_price = spot + target_offset if option_type == "CE" else spot - abs(target_offset)

        st.info(
            f"**Order Preview:** NIFTY {int(strike)} {option_type} ({expiry}) | "
            f"Type: {order_type} | Qty: {quantity} ({lots} lots) | Direction: BUY"
            + (f" | Limit: ₹{limit_price:.2f}" if order_type == "LIMIT" and limit_price else "")
        )

        confirm_key = f"confirm_order_{strike}_{option_type}_{order_type}"
        if st.button("Confirm Order", type="primary", key=confirm_key):
            with st.spinner("Placing order..."):
                result = dhan.place_super_order(
                    index="NIFTY",
                    strike=int(strike),
                    option_type=option_type,
                    direction="BUY",
                    quantity=quantity,
                    sl_price=sl_price,
                    target_price=target_price
                )

                if result.get('success'):
                    st.success(
                        f"Order Placed! Order ID: {result.get('order_id', 'N/A')} | "
                        f"Status: {result.get('status', 'PENDING')}"
                    )
                    st.balloons()
                else:
                    st.error(f"Order Failed: {result.get('error', 'Unknown error')}")

    except Exception as e:
        st.error(f"Error placing order: {str(e)}")
