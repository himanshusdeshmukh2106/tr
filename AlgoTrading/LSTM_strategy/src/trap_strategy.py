"""
EMA Trap Strategy Implementation
Intraday trading strategy based on false breakouts (traps) around 21 EMA
"""
import numpy as np
import pandas as pd
from datetime import datetime, time
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config.config import TRAP_STRATEGY_CONFIG


class TrapStrategy:
    def __init__(self, config=None):
        self.config = config or TRAP_STRATEGY_CONFIG
        self.position = None  # 'LONG', 'SHORT', or None
        self.entry_price = None
        self.stop_loss = None
        self.target = None
        self.trailing_stop = None
        
    def calculate_ema(self, prices, period=21):
        """Calculate Exponential Moving Average"""
        return prices.ewm(span=period, adjust=False).mean()
    
    def calculate_adx(self, high, low, close, period=14):
        """Calculate Average Directional Index"""
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed indicators
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * pd.Series(plus_dm).rolling(window=period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(window=period).mean() / atr
        
        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    def is_in_entry_window(self, timestamp):
        """Check if current time is within entry windows"""
        current_time = timestamp.time()
        
        for window in self.config["entry_windows"]:
            start = datetime.strptime(window["start"], "%H:%M").time()
            end = datetime.strptime(window["end"], "%H:%M").time()
            if start <= current_time <= end:
                return True
        return False
    
    def check_candle_body_size(self, open_price, close_price, current_price):
        """Check if candle body is within size limit"""
        body_size = abs(close_price - open_price)
        body_pct = (body_size / current_price) * 100
        return body_pct <= self.config["max_candle_body_pct"]
    
    def check_adx_range(self, adx_value):
        """Check if ADX is within valid range"""
        return self.config["adx_min"] <= adx_value <= self.config["adx_max"]
    
    def detect_upside_trap(self, df, current_idx):
        """
        Detect upside trap:
        1. Price crosses above 21 EMA
        2. Then closes below 21 EMA (trap)
        3. Signal: GO SHORT
        """
        lookback = self.config["trap_lookback"]
        if current_idx < lookback:
            return False
        
        ema = df['EMA_21'].iloc[current_idx]
        current_close = df['Close'].iloc[current_idx]
        
        # Check if current candle closed below EMA
        if current_close >= ema:
            return False
        
        # Look back to find if price was above EMA recently
        for i in range(1, lookback + 1):
            past_idx = current_idx - i
            past_close = df['Close'].iloc[past_idx]
            past_ema = df['EMA_21'].iloc[past_idx]
            
            # Found a candle that closed above EMA
            if past_close > past_ema:
                # Verify it's a significant cross
                distance_pct = ((past_close - past_ema) / past_ema) * 100
                if distance_pct >= self.config["min_trap_distance_pct"]:
                    return True
        
        return False
    
    def detect_downside_trap(self, df, current_idx):
        """
        Detect downside trap:
        1. Price crosses below 21 EMA
        2. Then closes above 21 EMA (trap)
        3. Signal: GO LONG
        """
        lookback = self.config["trap_lookback"]
        if current_idx < lookback:
            return False
        
        ema = df['EMA_21'].iloc[current_idx]
        current_close = df['Close'].iloc[current_idx]
        
        # Check if current candle closed above EMA
        if current_close <= ema:
            return False
        
        # Look back to find if price was below EMA recently
        for i in range(1, lookback + 1):
            past_idx = current_idx - i
            past_close = df['Close'].iloc[past_idx]
            past_ema = df['EMA_21'].iloc[past_idx]
            
            # Found a candle that closed below EMA
            if past_close < past_ema:
                # Verify it's a significant cross
                distance_pct = ((past_ema - past_close) / past_ema) * 100
                if distance_pct >= self.config["min_trap_distance_pct"]:
                    return True
        
        return False
    
    def generate_signal(self, df, idx):
        """
        Generate trading signal based on trap strategy
        Returns: 'LONG', 'SHORT', or None
        """
        timestamp = df.index[idx]
        
        # Check if in entry window
        if not self.is_in_entry_window(timestamp):
            return None
        
        # Get current values
        open_price = df['Open'].iloc[idx]
        close_price = df['Close'].iloc[idx]
        current_price = close_price
        adx_value = df['ADX'].iloc[idx]
        
        # Check entry conditions
        if not self.check_candle_body_size(open_price, close_price, current_price):
            return None
        
        if not self.check_adx_range(adx_value):
            return None
        
        # Check for traps
        if self.detect_downside_trap(df, idx):
            return 'LONG'
        elif self.detect_upside_trap(df, idx):
            return 'SHORT'
        
        return None
    
    def enter_position(self, signal, price):
        """Enter a position"""
        self.position = signal
        self.entry_price = price
        
        if signal == 'LONG':
            self.stop_loss = price * (1 - self.config["stop_loss_pct"] / 100)
            self.target = price * (1 + self.config["target_pct"] / 100)
        else:  # SHORT
            self.stop_loss = price * (1 + self.config["stop_loss_pct"] / 100)
            self.target = price * (1 - self.config["target_pct"] / 100)
        
        self.trailing_stop = None
    
    def update_trailing_stop(self, current_price):
        """Update trailing stop if profit threshold reached"""
        if self.position == 'LONG':
            profit_pct = ((current_price - self.entry_price) / self.entry_price) * 100
            if profit_pct >= 0.5:  # After 0.5% profit, activate trailing stop
                new_trailing = current_price * (1 - self.config["trailing_stop_pct"] / 100)
                if self.trailing_stop is None or new_trailing > self.trailing_stop:
                    self.trailing_stop = new_trailing
        
        elif self.position == 'SHORT':
            profit_pct = ((self.entry_price - current_price) / self.entry_price) * 100
            if profit_pct >= 0.5:
                new_trailing = current_price * (1 + self.config["trailing_stop_pct"] / 100)
                if self.trailing_stop is None or new_trailing < self.trailing_stop:
                    self.trailing_stop = new_trailing
    
    def check_exit(self, current_price, timestamp):
        """
        Check if position should be exited
        Returns: (should_exit, reason)
        """
        if self.position is None:
            return False, None
        
        # Check exit time
        exit_time = datetime.strptime(self.config["exit_time"], "%H:%M").time()
        if timestamp.time() >= exit_time:
            return True, "TIME_EXIT"
        
        # Check stops and targets
        if self.position == 'LONG':
            if current_price <= self.stop_loss:
                return True, "STOP_LOSS"
            if current_price >= self.target:
                return True, "TARGET"
            if self.trailing_stop and current_price <= self.trailing_stop:
                return True, "TRAILING_STOP"
        
        else:  # SHORT
            if current_price >= self.stop_loss:
                return True, "STOP_LOSS"
            if current_price <= self.target:
                return True, "TARGET"
            if self.trailing_stop and current_price >= self.trailing_stop:
                return True, "TRAILING_STOP"
        
        return False, None
    
    def exit_position(self):
        """Exit current position"""
        self.position = None
        self.entry_price = None
        self.stop_loss = None
        self.target = None
        self.trailing_stop = None
