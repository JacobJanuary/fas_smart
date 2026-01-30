"""
Technical indicator calculations for rolling window data.

All functions are optimized for NumPy arrays.
Uses EMA caching for efficient sequential updates.
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class IndicatorResult:
    """Container for calculated indicators"""
    rsi: Optional[float] = None
    macd_line: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    atr: Optional[float] = None
    volume_zscore: Optional[float] = None
    price_change_pct: Optional[float] = None
    buy_ratio: Optional[float] = None
    normalized_imbalance: Optional[float] = None


def calculate_ema(values: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate Exponential Moving Average.
    
    Args:
        values: Array of values
        period: EMA period
    
    Returns:
        Array of EMA values (same length as input)
    """
    if len(values) < period:
        return np.full(len(values), np.nan)
    
    alpha = 2 / (period + 1)
    ema = np.zeros(len(values))
    
    # Initialize with SMA
    ema[period - 1] = np.mean(values[:period])
    
    # Calculate EMA
    for i in range(period, len(values)):
        ema[i] = alpha * values[i] + (1 - alpha) * ema[i - 1]
    
    # Fill initial values with NaN
    ema[:period - 1] = np.nan
    
    return ema


def calculate_rsi(
    closes: np.ndarray,
    period: int = 14,
    ema_gain: Optional[float] = None,
    ema_loss: Optional[float] = None,
) -> Tuple[float, float, float]:
    """
    Calculate RSI with optional EMA caching for sequential updates.
    
    Args:
        closes: Array of close prices
        period: RSI period
        ema_gain: Previous EMA of gains (for caching)
        ema_loss: Previous EMA of losses (for caching)
    
    Returns:
        Tuple of (rsi, new_ema_gain, new_ema_loss)
    """
    if len(closes) < period + 1:
        return (None, None, None)
    
    # Calculate price changes
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    alpha = 1 / period  # Wilder's smoothing
    
    if ema_gain is None or ema_loss is None:
        # Initial calculation: SMA for first period, then EMA
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        # Apply EMA for remaining values
        for i in range(period, len(gains)):
            avg_gain = alpha * gains[i] + (1 - alpha) * avg_gain
            avg_loss = alpha * losses[i] + (1 - alpha) * avg_loss
    else:
        # Incremental update using cached EMA values
        avg_gain = alpha * gains[-1] + (1 - alpha) * ema_gain
        avg_loss = alpha * losses[-1] + (1 - alpha) * ema_loss
    
    if avg_loss == 0:
        rsi = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
    
    return (rsi, avg_gain, avg_loss)


def calculate_macd(
    closes: np.ndarray,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> Tuple[float, float, float]:
    """
    Calculate MACD indicator.
    
    Args:
        closes: Array of close prices
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal line period
    
    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    if len(closes) < slow_period + signal_period:
        return (None, None, None)
    
    ema_fast = calculate_ema(closes, fast_period)
    ema_slow = calculate_ema(closes, slow_period)
    
    # MACD line
    macd_line = ema_fast - ema_slow
    
    # Filter out NaN values for signal calculation
    valid_macd = macd_line[~np.isnan(macd_line)]
    
    if len(valid_macd) < signal_period:
        return (None, None, None)
    
    # Signal line (EMA of MACD)
    signal = calculate_ema(valid_macd, signal_period)
    
    macd_value = float(macd_line[-1])
    signal_value = float(signal[-1])
    histogram = macd_value - signal_value
    
    return (macd_value, signal_value, histogram)


def calculate_atr(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 14,
) -> Optional[float]:
    """
    Calculate Average True Range.
    
    Args:
        highs: Array of high prices
        lows: Array of low prices
        closes: Array of close prices
        period: ATR period
    
    Returns:
        ATR value
    """
    if len(closes) < period + 1:
        return None
    
    # True Range components
    high_low = highs[1:] - lows[1:]
    high_close = np.abs(highs[1:] - closes[:-1])
    low_close = np.abs(lows[1:] - closes[:-1])
    
    # True Range = max of the three
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    
    # ATR = EMA of True Range
    atr = calculate_ema(tr, period)
    
    return float(atr[-1]) if not np.isnan(atr[-1]) else None


def calculate_volume_zscore(
    volumes: np.ndarray,
    period: int = 20,
) -> Optional[float]:
    """
    Calculate Volume Z-Score (standard deviations from mean).
    
    Args:
        volumes: Array of volume values
        period: Lookback period for mean/std calculation
    
    Returns:
        Z-Score of latest volume
    """
    if len(volumes) < period:
        return None
    
    lookback = volumes[-period:]
    mean = np.mean(lookback)
    std = np.std(lookback)
    
    if std == 0:
        return 0.0
    
    zscore = (volumes[-1] - mean) / std
    return float(zscore)


def calculate_price_change(
    closes: np.ndarray,
) -> Optional[float]:
    """Calculate percentage price change from previous candle"""
    if len(closes) < 2:
        return None
    
    prev = closes[-2]
    current = closes[-1]
    
    if prev == 0:
        return 0.0
    
    return float((current - prev) / prev * 100)


def calculate_buy_ratio(volume: float, buy_volume: float) -> Optional[float]:
    """Calculate buy/sell ratio"""
    if volume == 0:
        return 0.5
    return buy_volume / volume


def calculate_normalized_imbalance(volume: float, buy_volume: float) -> Optional[float]:
    """
    Calculate normalized imbalance: (2*buy - total) / total
    Range: -1 (all sells) to +1 (all buys)
    """
    if volume == 0:
        return 0.0
    return (2 * buy_volume - volume) / volume


def calculate_all_indicators(pair_data, window: int = 15) -> IndicatorResult:
    """
    Calculate all indicators for a trading pair.
    
    Args:
        pair_data: PairData instance from storage
        window: Rolling window size in minutes
    
    Returns:
        IndicatorResult with all calculated values
    """
    result = IndicatorResult()
    
    # Get rolling window data
    agg = pair_data.aggregate_rolling_window(window)
    if agg is None:
        return result
    
    result.buy_ratio = calculate_buy_ratio(agg['volume'], agg['buy_volume'])
    result.normalized_imbalance = calculate_normalized_imbalance(agg['volume'], agg['buy_volume'])
    
    # Get historical data for indicators
    # RSI needs 14+1 rolling windows (15 periods)
    # MACD needs 26+9 rolling windows (35 periods)
    closes = pair_data.get_close_prices(50)  # Get enough history
    
    if len(closes) >= 15:
        # RSI
        rsi, new_gain, new_loss = calculate_rsi(
            closes, 14,
            pair_data.ema_rsi_gain,
            pair_data.ema_rsi_loss
        )
        result.rsi = rsi
        pair_data.ema_rsi_gain = new_gain
        pair_data.ema_rsi_loss = new_loss
        
        # Price change
        result.price_change_pct = calculate_price_change(closes)
    
    if len(closes) >= 35:
        # MACD
        macd, signal, hist = calculate_macd(closes)
        result.macd_line = macd
        result.macd_signal = signal
        result.macd_histogram = hist
    
    # Volume Z-Score
    volumes = pair_data.get_volumes(20)
    if len(volumes) >= 20:
        result.volume_zscore = calculate_volume_zscore(volumes)
    
    # ATR (need OHLC data)
    candles = pair_data.get_last_n_candles(15)
    if len(candles) >= 15:
        result.atr = calculate_atr(
            candles['high'],
            candles['low'],
            candles['close'],
        )
    
    return result
