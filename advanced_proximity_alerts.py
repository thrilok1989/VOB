"""
Advanced Proximity Alert System
Monitors price proximity to VOB and HTF levels and sends Telegram notifications
Includes comprehensive market context in all notifications
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pytz
from config import IST, get_current_time_ist
from notification_rate_limiter import get_rate_limiter
from telegram_alerts import TelegramBot
import streamlit as st

class ProximityAlert:
    """Represents a single proximity alert"""

    def __init__(self, symbol: str, alert_type: str, level: float,
                 level_type: str, distance: float, timeframe: str = None):
        """
        Initialize proximity alert

        Args:
            symbol: Trading symbol (NIFTY/SENSEX)
            alert_type: 'VOB' or 'HTF'
            level: Price level (support/resistance)
            level_type: 'Bull', 'Bear', 'Support', 'Resistance'
            distance: Points away from level
            timeframe: For HTF alerts (10T, 15T)
        """
        self.symbol = symbol
        self.alert_type = alert_type
        self.level = level
        self.level_type = level_type
        self.distance = distance
        self.timeframe = timeframe
        self.timestamp = get_current_time_ist()

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'symbol': self.symbol,
            'alert_type': self.alert_type,
            'level': self.level,
            'level_type': self.level_type,
            'distance': self.distance,
            'timeframe': self.timeframe,
            'timestamp': self.timestamp.isoformat()
        }


class AdvancedProximityAlertSystem:
    """
    Advanced alert system for monitoring price proximity to key levels
    - VOB levels: 7 points threshold
    - HTF S/R levels: 5 points threshold (10min & 15min only)
    - Rate limited: 10 minutes between notifications
    """

    # Thresholds
    VOB_PROXIMITY_THRESHOLD = 7.0  # Points
    HTF_PROXIMITY_THRESHOLD = 5.0  # Points

    # Monitored HTF timeframes
    HTF_MONITORED_TIMEFRAMES = ['10T', '15T']

    def __init__(self, cooldown_minutes: int = 10):
        """
        Initialize the alert system

        Args:
            cooldown_minutes: Cooldown period between notifications
        """
        self.rate_limiter = get_rate_limiter(cooldown_minutes=cooldown_minutes)
        self.telegram = TelegramBot()

        # Track active alerts
        self.active_alerts: List[ProximityAlert] = []

    def check_vob_proximity(self, symbol: str, current_price: float,
                           vob_data: Dict) -> List[ProximityAlert]:
        """
        Check if current price is near any VOB levels (within 7 points)

        Args:
            symbol: Trading symbol
            current_price: Current market price
            vob_data: VOB data from volume_order_blocks.py

        Returns:
            List of proximity alerts detected
        """
        alerts = []

        if not vob_data:
            return alerts

        # Check bullish VOB blocks
        for block in vob_data.get('bullish_blocks', []):
            if not block.get('active', True):
                continue

            # Check proximity to upper, mid, and lower levels
            levels = {
                'upper': block['upper'],
                'mid': block['mid'],
                'lower': block['lower']
            }

            for level_name, level_value in levels.items():
                distance = abs(current_price - level_value)

                if distance <= self.VOB_PROXIMITY_THRESHOLD:
                    alert = ProximityAlert(
                        symbol=symbol,
                        alert_type='VOB',
                        level=level_value,
                        level_type=f'Bull ({level_name})',
                        distance=distance
                    )
                    alerts.append(alert)

        # Check bearish VOB blocks
        for block in vob_data.get('bearish_blocks', []):
            if not block.get('active', True):
                continue

            levels = {
                'upper': block['upper'],
                'mid': block['mid'],
                'lower': block['lower']
            }

            for level_name, level_value in levels.items():
                distance = abs(current_price - level_value)

                if distance <= self.VOB_PROXIMITY_THRESHOLD:
                    alert = ProximityAlert(
                        symbol=symbol,
                        alert_type='VOB',
                        level=level_value,
                        level_type=f'Bear ({level_name})',
                        distance=distance
                    )
                    alerts.append(alert)

        return alerts

    def check_htf_proximity(self, symbol: str, current_price: float,
                           htf_data: List[Dict]) -> List[ProximityAlert]:
        """
        Check if current price is near HTF support/resistance (within 5 points)
        Only monitors 10min and 15min timeframes

        Args:
            symbol: Trading symbol
            current_price: Current market price
            htf_data: HTF S/R data from htf_support_resistance.py

        Returns:
            List of proximity alerts detected
        """
        alerts = []

        if not htf_data:
            return alerts

        for level_data in htf_data:
            timeframe = level_data.get('timeframe', '')

            # Only monitor 10T and 15T timeframes
            if timeframe not in self.HTF_MONITORED_TIMEFRAMES:
                continue

            # Check resistance (pivot high)
            if 'pivot_high' in level_data and level_data['pivot_high'] is not None:
                resistance = level_data['pivot_high']
                distance = abs(current_price - resistance)

                if distance <= self.HTF_PROXIMITY_THRESHOLD:
                    alert = ProximityAlert(
                        symbol=symbol,
                        alert_type='HTF',
                        level=resistance,
                        level_type='Resistance',
                        distance=distance,
                        timeframe=timeframe
                    )
                    alerts.append(alert)

            # Check support (pivot low)
            if 'pivot_low' in level_data and level_data['pivot_low'] is not None:
                support = level_data['pivot_low']
                distance = abs(current_price - support)

                if distance <= self.HTF_PROXIMITY_THRESHOLD:
                    alert = ProximityAlert(
                        symbol=symbol,
                        alert_type='HTF',
                        level=support,
                        level_type='Support',
                        distance=distance,
                        timeframe=timeframe
                    )
                    alerts.append(alert)

        return alerts

    def send_proximity_alert(self, alert: ProximityAlert, current_price: float) -> bool:
        """
        Send Telegram notification for proximity alert (if rate limit allows)

        Args:
            alert: ProximityAlert object
            current_price: Current market price

        Returns:
            True if notification was sent, False if rate limited
        """
        if not self.telegram.enabled:
            return False

        # Create alert key for rate limiting
        alert_key = f"{alert.alert_type.lower()}_proximity"

        # Check rate limit
        if not self.rate_limiter.can_send_notification(
            alert_type=alert_key,
            symbol=alert.symbol,
            level=alert.level
        ):
            # Get time remaining
            seconds_remaining = self.rate_limiter.get_time_until_next_notification(
                alert_type=alert_key,
                symbol=alert.symbol,
                level=alert.level
            )

            if seconds_remaining:
                minutes_remaining = seconds_remaining // 60
                print(f"Rate limited: {alert.symbol} {alert.alert_type} @ {alert.level:.2f} "
                      f"(next notification in {minutes_remaining} min)")

            return False

        # Build notification message
        message = self._build_alert_message(alert, current_price)

        # Send Telegram notification
        try:
            success = self.telegram.send_message(message)

            if success:
                # Record notification
                self.rate_limiter.record_notification(
                    alert_type=alert_key,
                    symbol=alert.symbol,
                    level=alert.level
                )
                print(f"Sent proximity alert: {alert.symbol} {alert.alert_type} @ {alert.level:.2f}")
                return True
            else:
                print(f"Failed to send proximity alert: {alert.symbol} {alert.alert_type}")
                return False

        except Exception as e:
            print(f"Error sending proximity alert: {e}")
            return False

    def _gather_market_context(self) -> Dict:
        """
        Gather comprehensive market context from session state

        Returns:
            Dictionary containing all market biases and analyses
        """
        context = {
            'overall_sentiment': 'N/A',
            'overall_score': 0,
            'market_breadth_bias': 'N/A',
            'market_breadth_pct': 0,
            'technical_indicators_bias': 'N/A',
            'technical_indicators_score': 0,
            'pcr_analysis_bias': 'N/A',
            'pcr_analysis_score': 0,
            'nifty_atm_verdict': 'N/A',
            'option_chain_bias': 'N/A',
            'option_chain_score': 0
        }

        # Check if bias analysis results exist
        if 'bias_analysis_results' in st.session_state and st.session_state.bias_analysis_results:
            analysis = st.session_state.bias_analysis_results

            if analysis.get('success'):
                # 1. Market Breadth (Stock Performance)
                stock_data = analysis.get('stock_data', [])
                if stock_data:
                    bullish_stocks = sum(1 for s in stock_data if s.get('change_pct', 0) > 0.5)
                    total_stocks = len(stock_data)
                    breadth_pct = (bullish_stocks / total_stocks * 100) if total_stocks > 0 else 50

                    context['market_breadth_pct'] = breadth_pct
                    if breadth_pct > 60:
                        context['market_breadth_bias'] = 'BULLISH'
                    elif breadth_pct < 40:
                        context['market_breadth_bias'] = 'BEARISH'
                    else:
                        context['market_breadth_bias'] = 'NEUTRAL'

                # 2. Technical Indicators
                bias_results = analysis.get('bias_results', [])
                if bias_results:
                    bullish_count = sum(1 for r in bias_results if 'BULLISH' in r.get('bias', ''))
                    bearish_count = sum(1 for r in bias_results if 'BEARISH' in r.get('bias', ''))
                    total_count = len(bias_results)

                    if bullish_count > bearish_count:
                        context['technical_indicators_bias'] = 'BULLISH'
                    elif bearish_count > bullish_count:
                        context['technical_indicators_bias'] = 'BEARISH'
                    else:
                        context['technical_indicators_bias'] = 'NEUTRAL'

                    # Calculate weighted score
                    total_score = sum(r.get('score', 0) * r.get('weight', 1) for r in bias_results)
                    total_weight = sum(r.get('weight', 1) for r in bias_results)
                    context['technical_indicators_score'] = total_score / total_weight if total_weight > 0 else 0

        # 3. PCR Analysis
        if 'overall_option_data' in st.session_state and st.session_state.overall_option_data:
            option_data = st.session_state.overall_option_data

            # Calculate PCR for main indices
            main_indices = ['NIFTY', 'SENSEX']
            bullish_instruments = 0
            bearish_instruments = 0
            total_score = 0
            instruments_analyzed = 0

            for instrument in main_indices:
                if instrument not in option_data:
                    continue

                data = option_data[instrument]
                if not data.get('success'):
                    continue

                # Calculate PCR for Total OI
                total_ce_oi = data.get('total_ce_oi', 0)
                total_pe_oi = data.get('total_pe_oi', 0)
                pcr_oi = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 1

                # Determine bias
                if pcr_oi > 1.2:
                    score = min(50, (pcr_oi - 1) * 50)
                    bullish_instruments += 1
                elif pcr_oi < 0.8:
                    score = -min(50, (1 - pcr_oi) * 50)
                    bearish_instruments += 1
                else:
                    score = 0

                total_score += score
                instruments_analyzed += 1

            if instruments_analyzed > 0:
                overall_score = total_score / instruments_analyzed
                context['pcr_analysis_score'] = overall_score

                if overall_score > 10:
                    context['pcr_analysis_bias'] = 'BULLISH'
                elif overall_score < -10:
                    context['pcr_analysis_bias'] = 'BEARISH'
                else:
                    context['pcr_analysis_bias'] = 'NEUTRAL'

        # 4. NIFTY ATM Zone Summary
        if 'NIFTY_atm_zone_bias' in st.session_state:
            df_atm = st.session_state['NIFTY_atm_zone_bias']
            atm_row = df_atm[df_atm["Zone"] == "ATM"]

            if not atm_row.empty:
                context['nifty_atm_verdict'] = atm_row.iloc[0].get('Verdict', 'N/A')

        # 5. Option Chain ATM Zone Analysis (Overall)
        instruments = ['NIFTY', 'SENSEX', 'FINNIFTY', 'MIDCPNIFTY']
        bullish_instruments = 0
        bearish_instruments = 0
        total_score = 0
        instruments_analyzed = 0

        for instrument in instruments:
            atm_key = f'{instrument}_atm_zone_bias'
            if atm_key not in st.session_state:
                continue

            df_atm = st.session_state[atm_key]
            atm_row = df_atm[df_atm["Zone"] == "ATM"]

            if atm_row.empty:
                continue

            verdict = str(atm_row.iloc[0].get('Verdict', 'Neutral')).upper()

            # Calculate score based on verdict
            if 'STRONG BULLISH' in verdict:
                score = 75
                bullish_instruments += 1
            elif 'BULLISH' in verdict:
                score = 40
                bullish_instruments += 1
            elif 'STRONG BEARISH' in verdict:
                score = -75
                bearish_instruments += 1
            elif 'BEARISH' in verdict:
                score = -40
                bearish_instruments += 1
            else:
                score = 0

            total_score += score
            instruments_analyzed += 1

        if instruments_analyzed > 0:
            overall_score = total_score / instruments_analyzed
            context['option_chain_score'] = overall_score

            if overall_score > 30:
                context['option_chain_bias'] = 'BULLISH'
            elif overall_score < -30:
                context['option_chain_bias'] = 'BEARISH'
            else:
                context['option_chain_bias'] = 'NEUTRAL'

        # 6. Overall Market Sentiment (from overall_market_sentiment.py)
        # This is a calculated metric from all the above
        # Calculate simple average of all biases
        biases = [
            context['market_breadth_bias'],
            context['technical_indicators_bias'],
            context['pcr_analysis_bias'],
            context['option_chain_bias']
        ]

        bullish_count = sum(1 for b in biases if b == 'BULLISH')
        bearish_count = sum(1 for b in biases if b == 'BEARISH')

        if bullish_count > bearish_count and bullish_count >= 2:
            context['overall_sentiment'] = 'BULLISH'
        elif bearish_count > bullish_count and bearish_count >= 2:
            context['overall_sentiment'] = 'BEARISH'
        else:
            context['overall_sentiment'] = 'NEUTRAL'

        # Calculate overall score
        scores = [
            context['technical_indicators_score'],
            context['pcr_analysis_score'],
            context['option_chain_score']
        ]
        context['overall_score'] = sum(s for s in scores if isinstance(s, (int, float))) / len([s for s in scores if isinstance(s, (int, float))]) if scores else 0

        return context

    def _build_alert_message(self, alert: ProximityAlert, current_price: float) -> str:
        """
        Build formatted Telegram message for proximity alert with comprehensive market context

        Args:
            alert: ProximityAlert object
            current_price: Current market price

        Returns:
            Formatted HTML message
        """
        # Determine emoji based on alert type
        if alert.alert_type == 'VOB':
            if 'Bull' in alert.level_type:
                emoji = "🟢"
                direction = "BULLISH VOB"
            else:
                emoji = "🔴"
                direction = "BEARISH VOB"
        else:  # HTF
            if alert.level_type == 'Support':
                emoji = "🟢"
                direction = "HTF SUPPORT"
            else:
                emoji = "🔴"
                direction = "HTF RESISTANCE"

        # Gather market context
        market_context = self._gather_market_context()

        # Build message with comprehensive market context
        lines = [
            f"{emoji} <b>PROXIMITY ALERT</b> {emoji}",
            "",
            f"<b>Symbol:</b> {alert.symbol}",
            f"<b>Type:</b> {direction}",
            f"<b>Level:</b> {alert.level:.2f}",
            f"<b>Current Price:</b> {current_price:.2f}",
            f"<b>Distance:</b> {alert.distance:.2f} points",
        ]

        if alert.timeframe:
            # Convert timeframe to readable format
            tf_readable = alert.timeframe.replace('T', 'm')
            lines.append(f"<b>Timeframe:</b> {tf_readable}")

        if alert.alert_type == 'VOB':
            lines.append(f"<b>Level Type:</b> {alert.level_type}")

        # Add comprehensive market context
        lines.extend([
            "",
            "━━━━━━━━━━━━━━━━━━",
            "<b>📊 MARKET CONTEXT</b>",
            "━━━━━━━━━━━━━━━━━━",
            "",
            f"<b>Overall Sentiment:</b> {self._format_bias(market_context['overall_sentiment'])}",
            f"<b>Overall Score:</b> {market_context['overall_score']:.1f}",
            "",
            f"<b>📈 Enhanced Market Analysis:</b>",
            f"  • Bias: {self._format_bias(market_context['technical_indicators_bias'])}",
            f"  • Score: {market_context['technical_indicators_score']:.1f}",
            "",
            f"<b>🔍 Market Breadth:</b>",
            f"  • Bias: {self._format_bias(market_context['market_breadth_bias'])}",
            f"  • Breadth: {market_context['market_breadth_pct']:.1f}%",
            "",
            f"<b>📊 Technical Indicators:</b>",
            f"  • Bias: {self._format_bias(market_context['technical_indicators_bias'])}",
            f"  • Score: {market_context['technical_indicators_score']:.1f}",
            "",
            f"<b>📉 PCR Analysis:</b>",
            f"  • Bias: {self._format_bias(market_context['pcr_analysis_bias'])}",
            f"  • Score: {market_context['pcr_analysis_score']:.1f}",
            "",
            f"<b>🎯 NIFTY ATM Zone:</b>",
            f"  • Verdict: {market_context['nifty_atm_verdict']}",
            "",
            f"<b>🔗 Option Chain Analysis:</b>",
            f"  • Bias: {self._format_bias(market_context['option_chain_bias'])}",
            f"  • Score: {market_context['option_chain_score']:.1f}",
            "",
            f"<i>Time: {alert.timestamp.strftime('%I:%M:%S %p')}</i>"
        ])

        return "\n".join(lines)

    def _format_bias(self, bias: str) -> str:
        """
        Format bias with appropriate emoji

        Args:
            bias: Bias string (BULLISH, BEARISH, NEUTRAL, N/A)

        Returns:
            Formatted bias string with emoji
        """
        bias_upper = str(bias).upper()

        if 'BULLISH' in bias_upper:
            return "🐂 BULLISH"
        elif 'BEARISH' in bias_upper:
            return "🐻 BEARISH"
        elif 'NEUTRAL' in bias_upper:
            return "⚖️ NEUTRAL"
        else:
            return "❓ N/A"

    def process_market_data(self, symbol: str, current_price: float,
                           vob_data: Dict, htf_data: List[Dict]) -> Tuple[List[ProximityAlert], int]:
        """
        Process market data and check for proximity alerts

        Args:
            symbol: Trading symbol
            current_price: Current market price
            vob_data: VOB block data
            htf_data: HTF support/resistance data

        Returns:
            Tuple of (all_alerts, notifications_sent)
        """
        all_alerts = []
        notifications_sent = 0

        # Check VOB proximity
        vob_alerts = self.check_vob_proximity(symbol, current_price, vob_data)
        all_alerts.extend(vob_alerts)

        # Check HTF proximity
        htf_alerts = self.check_htf_proximity(symbol, current_price, htf_data)
        all_alerts.extend(htf_alerts)

        # Send notifications (rate limited)
        for alert in all_alerts:
            if self.send_proximity_alert(alert, current_price):
                notifications_sent += 1

        return all_alerts, notifications_sent

    def get_alert_summary(self, alerts: List[ProximityAlert]) -> str:
        """
        Get summary of active alerts

        Args:
            alerts: List of proximity alerts

        Returns:
            Summary string
        """
        if not alerts:
            return "No proximity alerts"

        vob_count = sum(1 for a in alerts if a.alert_type == 'VOB')
        htf_count = sum(1 for a in alerts if a.alert_type == 'HTF')

        return f"Active: {len(alerts)} alerts (VOB: {vob_count}, HTF: {htf_count})"

    def clear_old_rate_limit_entries(self, days: int = 7):
        """
        Clear old rate limit entries

        Args:
            days: Clear entries older than this many days
        """
        self.rate_limiter.clear_old_entries(days_old=days)


# Global instance
_proximity_alert_system = None

def get_proximity_alert_system(cooldown_minutes: int = 10) -> AdvancedProximityAlertSystem:
    """
    Get or create global proximity alert system

    Args:
        cooldown_minutes: Cooldown period between notifications

    Returns:
        AdvancedProximityAlertSystem instance
    """
    global _proximity_alert_system
    if _proximity_alert_system is None:
        _proximity_alert_system = AdvancedProximityAlertSystem(cooldown_minutes=cooldown_minutes)
    return _proximity_alert_system
