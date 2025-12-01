import requests
import streamlit as st
from config import get_dhan_credentials, LOT_SIZES
from datetime import datetime

class DhanAPI:
    def __init__(self):
        """Initialize DhanHQ API"""
        creds = get_dhan_credentials()
        if not creds:
            raise Exception("DhanHQ credentials not found")
        
        self.client_id = creds['client_id']
        self.access_token = creds['access_token']
        self.base_url = "https://api.dhan.co/v2"
        
        self.headers = {
            'Content-Type': 'application/json',
            'access-token': self.access_token
        }
    
    def test_connection(self):
        """Test API connection"""
        try:
            url = f"{self.base_url}/fundlimit"
            response = requests.get(url, headers=self.headers, timeout=10)
            return response.status_code == 200
        except:
            return False
    
    def place_super_order(self, index: str, strike: int, option_type: str, 
                         direction: str, quantity: int, sl_price: float, 
                         target_price: float):
        """
        Place Super Order (Entry + SL + Target)
        
        Args:
            index: "NIFTY" or "SENSEX"
            strike: Strike price
            option_type: "CE" or "PE"
            direction: "BUY" or "SELL"
            quantity: Total quantity (lot_size × lots)
            sl_price: Stop loss price (index level)
            target_price: Target price (index level)
        
        Returns:
            dict with order response
        """
        
        # Get security ID
        security_id = self._get_security_id(index, strike, option_type)
        
        # Determine exchange segment
        exchange_segment = "NSE_FNO" if index == "NIFTY" else "BSE_FNO"
        
        order_data = {
            "dhanClientId": self.client_id,
            "transactionType": direction,
            "exchangeSegment": exchange_segment,
            "productType": "INTRADAY",
            "orderType": "MARKET",
            "validity": "DAY",
            "securityId": security_id,
            "quantity": quantity,
            "price": 0,  # Market order
            "stopLossPrice": sl_price,
            "targetPrice": target_price,
            "trailingJump": 0
        }
        
        try:
            url = f"{self.base_url}/super/orders"
            response = requests.post(url, json=order_data, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'success': True,
                    'order_id': result.get('orderId'),
                    'status': result.get('orderStatus'),
                    'message': 'Super Order placed successfully'
                }
            else:
                return {
                    'success': False,
                    'error': response.text,
                    'message': 'Order placement failed'
                }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': 'API request failed'
            }
    
    def get_positions(self):
        """Get all open positions"""
        try:
            url = f"{self.base_url}/positions"
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                return {
                    'success': True,
                    'positions': response.json()
                }
            else:
                return {
                    'success': False,
                    'error': response.text
                }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_order_status(self, order_id: str):
        """Get order status"""
        try:
            url = f"{self.base_url}/orders/{order_id}"
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                return {
                    'success': True,
                    'order': response.json()
                }
            else:
                return {
                    'success': False,
                    'error': response.text
                }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def exit_position(self, order_id: str):
        """Exit position"""
        try:
            url = f"{self.base_url}/super/orders/{order_id}/ENTRY_LEG"
            response = requests.delete(url, headers=self.headers, timeout=10)
            
            return {
                'success': response.status_code == 200,
                'message': 'Position exited' if response.status_code == 200 else 'Exit failed'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _get_security_id(self, index: str, strike: int, option_type: str):
        """
        Get DhanHQ security ID
        
        TODO: Download instrument master CSV and implement proper lookup
        https://images.dhan.co/api-data/api-scrip-master.csv
        
        For now, returns placeholder
        """
        # Placeholder - you'll need to implement proper mapping
        return f"{index}_{strike}_{option_type}"


def check_dhan_connection():
    """Check DhanHQ connection"""
    try:
        from dhan_data_fetcher import test_dhan_connection
        return test_dhan_connection()
    except Exception as e:
        st.error(f"❌ DhanHQ Connection Failed: {e}")
        return False
