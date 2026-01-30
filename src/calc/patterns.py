"""
Pattern detection for trading signals.

Ported from fas_v2.signal_detect_patterns SQL function.
All 11 patterns from the original system.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum

logger = logging.getLogger(__name__)


class PatternType(str, Enum):
    VOLUME_ANOMALY = "VOLUME_ANOMALY"
    OI_EXPLOSION = "OI_EXPLOSION"
    OI_DIVERGENCE = "OI_DIVERGENCE"
    OI_COLLAPSE = "OI_COLLAPSE"           # NEW: FAS V2
    CVD_PRICE_DIVERGENCE = "CVD_PRICE_DIVERGENCE"
    FUNDING_EXTREME = "FUNDING_EXTREME"
    FUNDING_FLIP = "FUNDING_FLIP"
    RSI_EXTREME = "RSI_EXTREME"
    RSI_DIVERGENCE = "RSI_DIVERGENCE"
    MACD_CROSSOVER = "MACD_CROSSOVER"
    MACD_DIVERGENCE = "MACD_DIVERGENCE"
    LIQUIDATION_CASCADE = "LIQUIDATION_CASCADE"
    # NEW: FAS V2 patterns
    ACCUMULATION = "ACCUMULATION"
    DISTRIBUTION = "DISTRIBUTION"
    MOMENTUM_EXHAUSTION = "MOMENTUM_EXHAUSTION"
    SQUEEZE_IGNITION = "SQUEEZE_IGNITION"


@dataclass
class Pattern:
    """Detected pattern with score and details"""
    pattern_type: PatternType
    score: float  # Positive = bullish, Negative = bearish
    confidence: float  # 0-100
    details: dict = field(default_factory=dict)


@dataclass
class PatternThresholds:
    """Configurable thresholds for pattern detection"""
    # Volume
    volume_zscore_threshold: float = 2.0
    
    # Open Interest
    oi_explosion_threshold: float = 7.0   # % change (FAS V2: raised from 5 to 7)
    oi_collapse_threshold: float = -7.0   # % change (negative)
    oi_divergence_threshold: float = 3.0
    
    # CVD
    cvd_divergence_threshold: float = 5.0
    price_change_min_cvd: float = 1.0  # %
    
    # Funding
    funding_extreme_threshold: float = 0.001  # 0.1%
    funding_flip_prev_threshold: float = 0.0005
    
    # RSI
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    rsi_extreme_overbought: float = 80.0
    rsi_extreme_oversold: float = 20.0
    
    # Liquidations
    liq_cascade_threshold: float = 50.0  # USD notional
    
    # Accumulation/Distribution (sideways range)
    sideways_price_range: float = 1.0  # % max price change for sideways
    accum_cvd_threshold: float = 0.1   # Positive CVD change
    accum_volume_threshold: float = 1.5  # Volume Z-score min


class PatternDetector:
    """
    Detects trading patterns based on calculated indicators.
    """
    
    def __init__(self, thresholds: PatternThresholds = None):
        self.thresholds = thresholds or PatternThresholds()
    
    def detect_all(
        self,
        indicators,  # IndicatorResult
        pair_data,   # PairData
        prev_indicators: Optional[dict] = None,
    ) -> List[Pattern]:
        """
        Detect all patterns for a trading pair.
        
        Args:
            indicators: Current calculated indicators
            pair_data: PairData with history
            prev_indicators: Previous period indicators (for divergence detection)
        
        Returns:
            List of detected patterns
        """
        patterns = []
        
        # 1. Volume Anomaly
        p = self._detect_volume_anomaly(indicators)
        if p:
            patterns.append(p)
        
        # 2. OI Explosion
        p = self._detect_oi_explosion(indicators)
        if p:
            patterns.append(p)
        
        # 3. OI Collapse (NEW)
        p = self._detect_oi_collapse(indicators)
        if p:
            patterns.append(p)
        
        # 4. CVD-Price Divergence
        p = self._detect_cvd_divergence(indicators)
        if p:
            patterns.append(p)
        
        # 5. Funding Extreme
        p = self._detect_funding_extreme(pair_data)
        if p:
            patterns.append(p)
        
        # 6. RSI Extreme
        p = self._detect_rsi_extreme(indicators)
        if p:
            patterns.append(p)
        
        # 7. MACD Crossover
        if prev_indicators:
            p = self._detect_macd_crossover(indicators, prev_indicators)
            if p:
                patterns.append(p)
        
        # 8. Liquidation Cascade
        p = self._detect_liquidation_cascade(pair_data)
        if p:
            patterns.append(p)
        
        # 9. Accumulation (NEW)
        p = self._detect_accumulation(indicators, pair_data)
        if p:
            patterns.append(p)
        
        # 10. Distribution (NEW)
        p = self._detect_distribution(indicators, pair_data)
        if p:
            patterns.append(p)
        
        # 11. Momentum Exhaustion (NEW)
        p = self._detect_momentum_exhaustion(indicators, pair_data)
        if p:
            patterns.append(p)
        
        return patterns
    
    def _detect_volume_anomaly(self, indicators) -> Optional[Pattern]:
        """Detect unusual volume spike"""
        if indicators.volume_zscore is None:
            return None
        
        zscore = indicators.volume_zscore
        threshold = self.thresholds.volume_zscore_threshold
        
        if abs(zscore) <= threshold:
            return None
        
        # Score based on direction
        score = 30.0 if zscore > 0 else -30.0
        confidence = min(50 + abs(zscore) * 10, 95)
        
        return Pattern(
            pattern_type=PatternType.VOLUME_ANOMALY,
            score=score,
            confidence=confidence,
            details={
                'volume_zscore': zscore,
                'threshold': threshold,
                'direction': 'HIGH' if zscore > 0 else 'LOW',
            }
        )
    
    def _detect_oi_explosion(self, indicators) -> Optional[Pattern]:
        """Detect sudden OI increase (>7% change)"""
        if indicators.oi_delta_pct is None:
            return None
        
        threshold = self.thresholds.oi_explosion_threshold
        if indicators.oi_delta_pct < threshold:
            return None
        
        score = 50.0  # Strong bullish signal
        confidence = min(70 + indicators.oi_delta_pct * 2, 95)
        
        return Pattern(
            pattern_type=PatternType.OI_EXPLOSION,
            score=score,
            confidence=confidence,
            details={
                'oi_delta_pct': indicators.oi_delta_pct,
                'threshold': threshold,
            }
        )
    
    def _detect_oi_collapse(self, indicators) -> Optional[Pattern]:
        """Detect sudden OI decrease (<-7% change)"""
        if indicators.oi_delta_pct is None:
            return None
        
        threshold = self.thresholds.oi_collapse_threshold
        if indicators.oi_delta_pct > threshold:  # threshold is negative
            return None
        
        score = -50.0  # Strong bearish signal
        confidence = min(70 + abs(indicators.oi_delta_pct) * 2, 90)
        
        return Pattern(
            pattern_type=PatternType.OI_COLLAPSE,
            score=score,
            confidence=confidence,
            details={
                'oi_delta_pct': indicators.oi_delta_pct,
                'threshold': threshold,
            }
        )
    
    def _detect_cvd_divergence(self, indicators) -> Optional[Pattern]:
        """Detect CVD-Price divergence"""
        if indicators.price_change_pct is None or indicators.normalized_imbalance is None:
            return None
        
        price_change = indicators.price_change_pct
        imbalance = indicators.normalized_imbalance * 100  # Scale to %
        threshold = self.thresholds.cvd_divergence_threshold
        min_price = self.thresholds.price_change_min_cvd
        
        # Check for divergence
        divergence = abs(price_change - imbalance)
        if divergence <= threshold:
            return None
        
        if abs(price_change) < min_price:
            return None
        
        # Signs must be opposite
        if (price_change > 0 and imbalance > 0) or (price_change < 0 and imbalance < 0):
            return None
        
        # Bearish: price up but CVD down
        # Bullish: price down but CVD up
        if price_change > 0 and imbalance < 0:
            score = -20.0  # Bearish signal
        else:
            score = 20.0   # Bullish signal
        
        confidence = min(60 + divergence * 10, 85)
        
        return Pattern(
            pattern_type=PatternType.CVD_PRICE_DIVERGENCE,
            score=score,
            confidence=confidence,
            details={
                'price_change': price_change,
                'cvd_imbalance': imbalance,
                'divergence': divergence,
            }
        )
    
    def _detect_funding_extreme(self, pair_data) -> Optional[Pattern]:
        """Detect extreme funding rate"""
        rate = pair_data.latest_funding_rate
        threshold = self.thresholds.funding_extreme_threshold
        
        if abs(rate) <= threshold:
            return None
        
        # High positive = many longs = bearish
        # High negative = many shorts = bullish
        score = -20.0 if rate > 0 else 20.0
        confidence = min(50 + abs(rate) / threshold * 20, 85)
        
        return Pattern(
            pattern_type=PatternType.FUNDING_EXTREME,
            score=score,
            confidence=confidence,
            details={
                'funding_rate': rate,
                'threshold': threshold,
                'direction': 'HIGH_LONGS' if rate > 0 else 'HIGH_SHORTS',
            }
        )
    
    def _detect_rsi_extreme(self, indicators) -> Optional[Pattern]:
        """Detect RSI at extreme levels"""
        if indicators.rsi is None:
            return None
        
        rsi = indicators.rsi
        
        if rsi >= self.thresholds.rsi_extreme_overbought:
            # Extremely overbought = bearish
            score = -25.0
            confidence = min(60 + (rsi - 80) * 2, 90)
            direction = 'OVERBOUGHT'
        elif rsi <= self.thresholds.rsi_extreme_oversold:
            # Extremely oversold = bullish
            score = 25.0
            confidence = min(60 + (20 - rsi) * 2, 90)
            direction = 'OVERSOLD'
        else:
            return None
        
        return Pattern(
            pattern_type=PatternType.RSI_EXTREME,
            score=score,
            confidence=confidence,
            details={
                'rsi': rsi,
                'direction': direction,
            }
        )
    
    def _detect_macd_crossover(self, indicators, prev_indicators) -> Optional[Pattern]:
        """Detect MACD line crossing signal line"""
        if indicators.macd_line is None or indicators.macd_signal is None:
            return None
        
        prev_macd = prev_indicators.get('macd_line')
        prev_signal = prev_indicators.get('macd_signal')
        
        if prev_macd is None or prev_signal is None:
            return None
        
        curr_diff = indicators.macd_line - indicators.macd_signal
        prev_diff = prev_macd - prev_signal
        
        # Check for crossover
        if prev_diff <= 0 and curr_diff > 0:
            # Bullish crossover
            score = 20.0
            direction = 'BULLISH'
        elif prev_diff >= 0 and curr_diff < 0:
            # Bearish crossover
            score = -20.0
            direction = 'BEARISH'
        else:
            return None
        
        confidence = min(60 + abs(curr_diff) * 100, 85)
        
        return Pattern(
            pattern_type=PatternType.MACD_CROSSOVER,
            score=score,
            confidence=confidence,
            details={
                'macd_line': indicators.macd_line,
                'signal_line': indicators.macd_signal,
                'histogram': indicators.macd_histogram,
                'direction': direction,
            }
        )
    
    def _detect_liquidation_cascade(self, pair_data) -> Optional[Pattern]:
        """Detect significant liquidation activity"""
        liq_long = pair_data.liq_long_current
        liq_short = pair_data.liq_short_current
        threshold = self.thresholds.liq_cascade_threshold
        
        total = liq_long + liq_short
        if total < threshold:
            return None
        
        # More long liq = bearish, more short liq = bullish
        if liq_long > liq_short:
            score = -15.0 * (liq_long / total)
            direction = 'LONG_LIQUIDATIONS'
        else:
            score = 15.0 * (liq_short / total)
            direction = 'SHORT_LIQUIDATIONS'
        
        confidence = min(50 + total / threshold * 10, 80)
        
        return Pattern(
            pattern_type=PatternType.LIQUIDATION_CASCADE,
            score=score,
            confidence=confidence,
            details={
                'long_liquidated': liq_long,
                'short_liquidated': liq_short,
                'total': total,
                'direction': direction,
            }
        )
    
    def _detect_accumulation(self, indicators, pair_data) -> Optional[Pattern]:
        """
        Detect accumulation pattern: 
        - Sideways price action (small price change)
        - Elevated volume
        - Positive CVD (more buying)
        """
        if indicators.price_change_pct is None or indicators.volume_zscore is None:
            return None
        if indicators.smoothed_imbalance is None:
            return None
        
        # Check for sideways (low price change)
        if abs(indicators.price_change_pct) > self.thresholds.sideways_price_range:
            return None
        
        # Check for elevated volume
        if indicators.volume_zscore < self.thresholds.accum_volume_threshold:
            return None
        
        # Check for positive order flow (accumulation)
        if indicators.smoothed_imbalance < self.thresholds.accum_cvd_threshold:
            return None
        
        score = 20.0  # Bullish
        confidence = min(60 + indicators.volume_zscore * 5, 85)
        
        return Pattern(
            pattern_type=PatternType.ACCUMULATION,
            score=score,
            confidence=confidence,
            details={
                'price_change_pct': indicators.price_change_pct,
                'volume_zscore': indicators.volume_zscore,
                'smoothed_imbalance': indicators.smoothed_imbalance,
            }
        )
    
    def _detect_distribution(self, indicators, pair_data) -> Optional[Pattern]:
        """
        Detect distribution pattern: 
        - Sideways price action (small price change)
        - Elevated volume
        - Negative CVD (more selling)
        """
        if indicators.price_change_pct is None or indicators.volume_zscore is None:
            return None
        if indicators.smoothed_imbalance is None:
            return None
        
        # Check for sideways (low price change)
        if abs(indicators.price_change_pct) > self.thresholds.sideways_price_range:
            return None
        
        # Check for elevated volume
        if indicators.volume_zscore < self.thresholds.accum_volume_threshold:
            return None
        
        # Check for negative order flow (distribution)
        if indicators.smoothed_imbalance > -self.thresholds.accum_cvd_threshold:
            return None
        
        score = -20.0  # Bearish
        confidence = min(60 + indicators.volume_zscore * 5, 85)
        
        return Pattern(
            pattern_type=PatternType.DISTRIBUTION,
            score=score,
            confidence=confidence,
            details={
                'price_change_pct': indicators.price_change_pct,
                'volume_zscore': indicators.volume_zscore,
                'smoothed_imbalance': indicators.smoothed_imbalance,
            }
        )
    
    def _detect_momentum_exhaustion(self, indicators, pair_data) -> Optional[Pattern]:
        """
        Detect momentum exhaustion:
        - RSI at extreme (overbought/oversold)
        - MACD showing divergence (histogram decreasing magnitude)
        """
        if indicators.rsi is None or indicators.macd_histogram is None:
            return None
        if pair_data.prev_macd_histogram is None:
            return None
        
        rsi = indicators.rsi
        macd_now = abs(indicators.macd_histogram)
        macd_prev = abs(pair_data.prev_macd_histogram)
        
        # Overbought + momentum fading = bearish exhaustion
        if rsi > self.thresholds.rsi_overbought and macd_now < macd_prev * 0.8:
            score = -30.0
            confidence = min(60 + (rsi - 70) * 2, 85)
            direction = 'BEARISH_EXHAUSTION'
        # Oversold + momentum fading = bullish exhaustion (reversal signal)
        elif rsi < self.thresholds.rsi_oversold and macd_now < macd_prev * 0.8:
            score = 30.0
            confidence = min(60 + (30 - rsi) * 2, 85)
            direction = 'BULLISH_EXHAUSTION'
        else:
            return None
        
        return Pattern(
            pattern_type=PatternType.MOMENTUM_EXHAUSTION,
            score=score,
            confidence=confidence,
            details={
                'rsi': rsi,
                'macd_histogram': indicators.macd_histogram,
                'prev_macd_histogram': pair_data.prev_macd_histogram,
                'direction': direction,
            }
        )


def calculate_total_score(patterns: List[Pattern]) -> tuple[float, str, float]:
    """
    Calculate total score from all detected patterns.
    
    Returns:
        Tuple of (total_score, direction, avg_confidence)
    """
    if not patterns:
        return (0.0, 'NEUTRAL', 0.0)
    
    total_score = sum(p.score for p in patterns)
    avg_confidence = sum(p.confidence for p in patterns) / len(patterns)
    
    if total_score > 10:
        direction = 'LONG'
    elif total_score < -10:
        direction = 'SHORT'
    else:
        direction = 'NEUTRAL'
    
    return (total_score, direction, avg_confidence)
