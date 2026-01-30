"""
Market Regime Calculator.

Determines market regime (BULL/BEAR/NEUTRAL) based on BTC price changes
and applies score adjustments accordingly.

Ported from FAS V2 calculate_market_regime().
"""

from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class MarketRegime:
    """Current market regime state."""
    regime: str  # 'BULL', 'BEAR', 'NEUTRAL'
    strength: float  # Magnitude of regime (used for adjustment)
    btc_change_1h: float  # BTC % change over 1 hour
    btc_change_4h: float  # BTC % change over 4 hours
    btc_change_24h: float  # BTC % change over 24 hours


def calculate_market_regime(btc_pair_data) -> MarketRegime:
    """
    Calculate market regime based on BTC price changes.
    
    Uses LAG logic from FAS V2:
    - 1h = 4 candles (15m * 4)
    - 4h = 16 candles (15m * 16)  
    - 24h = 96 candles (15m * 96)
    
    Args:
        btc_pair_data: PairData for BTCUSDT
        
    Returns:
        MarketRegime with current state
    """
    # Need 97 candles for 24h + current
    closes = btc_pair_data.get_close_prices(97)
    
    if len(closes) < 17:  # Minimum for 4h calculation
        return MarketRegime('NEUTRAL', 0.0, 0.0, 0.0, 0.0)
    
    current = closes[-1]
    
    # Calculate price changes
    btc_1h = 0.0
    btc_4h = 0.0
    btc_24h = 0.0
    
    if len(closes) >= 5 and closes[-5] > 0:
        btc_1h = (current - closes[-5]) / closes[-5] * 100
    
    if len(closes) >= 17 and closes[-17] > 0:
        btc_4h = (current - closes[-17]) / closes[-17] * 100
    
    if len(closes) >= 97 and closes[-97] > 0:
        btc_24h = (current - closes[-97]) / closes[-97] * 100
    
    # Determine regime using FAS V2 thresholds
    regime = 'NEUTRAL'
    
    # BEAR conditions (checked first for priority)
    if btc_1h <= -0.5 or btc_4h <= -1.0:
        regime = 'BEAR'
    elif btc_4h <= -0.5 or btc_24h <= -2.0:
        regime = 'BEAR'
    
    # BULL conditions
    elif btc_1h >= 0.5 or btc_4h >= 1.0:
        regime = 'BULL'
    elif btc_4h >= 0.5 or btc_24h >= 2.0:
        regime = 'BULL'
    
    # Strength = |btc_4h| (simplified from FAS V2)
    strength = abs(btc_4h)
    
    return MarketRegime(
        regime=regime,
        strength=strength,
        btc_change_1h=round(btc_1h, 2),
        btc_change_4h=round(btc_4h, 2),
        btc_change_24h=round(btc_24h, 2)
    )


def adjust_score_for_regime(score: float, regime: MarketRegime) -> float:
    """
    Adjust indicator score based on market regime.
    
    FAS V2 formula:
    - BULL: positive scores × (1 + strength×0.2), negative × (1 - strength×0.1)
    - BEAR: negative scores × (1 + strength×0.2), positive × (1 - strength×0.1)
    
    Args:
        score: Raw indicator score
        regime: Current market regime
        
    Returns:
        Adjusted score
    """
    if regime.regime == 'NEUTRAL' or regime.strength == 0:
        return score
    
    # Cap strength effect at reasonable levels
    strength = min(regime.strength, 5.0)  # Max 5% move
    
    if regime.regime == 'BULL':
        if score > 0:
            # Amplify bullish signals
            return score * (1 + strength * 0.2)
        else:
            # Dampen bearish signals
            return score * (1 - strength * 0.1)
    
    elif regime.regime == 'BEAR':
        if score < 0:
            # Amplify bearish signals
            return score * (1 + strength * 0.2)
        else:
            # Dampen bullish signals
            return score * (1 - strength * 0.1)
    
    return score
