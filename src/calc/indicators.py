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
    volume_zscore: Optional[float] = None
    price_change_pct: Optional[float] = None
    normalized_imbalance: Optional[float] = None
    # New fields for FAS V2 parity
    cvd_cumulative: Optional[float] = None
    prev_cvd_cumulative: Optional[float] = None
    smoothed_imbalance: Optional[float] = None
    oi_delta_pct: Optional[float] = None


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
    
    result.normalized_imbalance = calculate_normalized_imbalance(agg['volume'], agg['buy_volume'])
    
    # CVD and Smoothed Imbalance from PairData (updated on each candle)
    result.cvd_cumulative = pair_data.cvd_cumulative
    result.prev_cvd_cumulative = pair_data.prev_cvd_cumulative
    result.smoothed_imbalance = pair_data.smoothed_imbalance
    
    # OI Delta % calculation
    if pair_data.prev_open_interest > 0 and pair_data.latest_open_interest > 0:
        result.oi_delta_pct = (
            (pair_data.latest_open_interest - pair_data.prev_open_interest) 
            / pair_data.prev_open_interest * 100
        )
    
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
    
    return result


def calculate_indicator_score(indicators: IndicatorResult, pair_data) -> float:
    """
    Calculate indicator score matching FAS V2 calculate_indicator_scores_batch_v2.
    
    Components:
    - Volume Z-Score: ±35 (|zscore| > 3) or ±20 (|zscore| > 2)
    - CVD Component: ±20 (based on CVD direction)
    - Smoothed Imbalance: ±15 or ±7 (based on imbalance level)
    - MACD Crossover: ±10 (histogram sign change)
    
    Returns:
        indicator_score (float)
    """
    score = 0.0
    
    # 1. Volume Z-Score component
    if indicators.volume_zscore is not None and indicators.price_change_pct is not None:
        zscore = abs(indicators.volume_zscore)
        direction = 1 if indicators.price_change_pct >= 0 else -1
        
        if zscore > 3.0:
            score += 35 * direction
        elif zscore > 2.0:
            score += 20 * direction
    
    # 2. CVD component
    if indicators.cvd_cumulative is not None and indicators.prev_cvd_cumulative is not None:
        if indicators.cvd_cumulative > indicators.prev_cvd_cumulative:
            score += 20  # Bullish
        elif indicators.cvd_cumulative < indicators.prev_cvd_cumulative:
            score -= 20  # Bearish
    
    # 3. Smoothed Imbalance component
    if indicators.smoothed_imbalance is not None:
        imb = indicators.smoothed_imbalance
        if imb > 0.3:
            score += 15
        elif imb > 0.1:
            score += 7
        elif imb < -0.3:
            score -= 15
        elif imb < -0.1:
            score -= 7
    
    # 4. MACD Crossover component (histogram sign change)
    if indicators.macd_histogram is not None and pair_data.prev_macd_histogram is not None:
        prev = pair_data.prev_macd_histogram
        curr = indicators.macd_histogram
        
        if prev < 0 and curr > 0:
            score += 10  # Bullish crossover
        elif prev > 0 and curr < 0:
            score -= 10  # Bearish crossover
    
    # 5. RSI component (FAS V2 parity)
    if indicators.rsi is not None:
        if indicators.rsi > 60:
            score += 5  # Bullish momentum
        elif indicators.rsi < 40:
            score -= 5  # Bearish momentum
    
    return score
