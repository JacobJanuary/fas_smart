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
    SMART_MONEY_DIVERGENCE = "SMART_MONEY_DIVERGENCE"


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
    # Volume (default for TIER_2)
    volume_zscore_threshold: float = 3.0
    
    # Open Interest (default for TIER_2)
    oi_explosion_threshold: float = 4.0
    oi_collapse_threshold: float = -10.0  # FAS V2: -10%
    oi_divergence_threshold: float = 3.0
    
    # Liquidations (default for TIER_2)
    liq_ratio_threshold: float = 0.025
    
    # CVD (FAS V2 optimized values)
    cvd_divergence_threshold: float = 3.0  # FAS V2: 3.0
    price_change_min_cvd: float = 2.0  # FAS V2: 2.0%
    
    # Funding
    funding_extreme_threshold: float = 0.001  # 0.1%
    funding_flip_prev_threshold: float = 0.0005
    
    # RSI
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    rsi_extreme_overbought: float = 70.0  # FAS V2: 70
    rsi_extreme_oversold: float = 30.0    # FAS V2: 30
    
    # Accumulation/Distribution (FAS V2 Phase 1-2 optimized)
    sideways_price_range: float = 0.8  # FAS V2: ±0.8% max price change
    accum_cvd_threshold: float = 0.35  # FAS V2: normalized_imbalance > 0.35
    accum_volume_threshold: float = 2.0  # FAS V2: buy > sell * 2.0
    
    @classmethod
    def for_tier(cls, tier: str) -> 'PatternThresholds':
        """
        Get tier-adjusted thresholds (FAS V2 parity).
        
        TIER_1 (BTC, ETH): Lower thresholds - high liquidity, less noise
        TIER_2 (mid-caps): Medium thresholds
        TIER_3 (low-caps): Higher thresholds - more volatility
        """
        if tier == 'TIER_1':
            return cls(
                volume_zscore_threshold=2.5,
                oi_explosion_threshold=3.0,
                liq_ratio_threshold=0.02,
            )
        elif tier == 'TIER_2':
            return cls(
                volume_zscore_threshold=3.0,
                oi_explosion_threshold=4.0,
                liq_ratio_threshold=0.025,
            )
        else:  # TIER_3+
            return cls(
                volume_zscore_threshold=3.5,
                oi_explosion_threshold=5.0,
                liq_ratio_threshold=0.03,
            )


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
        p = self._detect_liquidation_cascade(pair_data, indicators)
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
        
        # 12. Smart Money Divergence (NEW: FAS V2)
        p = self._detect_smart_money_divergence(indicators)
        if p:
            patterns.append(p)
        
        # 13. OI Divergence
        if prev_indicators:
            p = self._detect_oi_divergence(indicators, prev_indicators)
            if p:
                patterns.append(p)
        
        # 14. Funding Flip
        prev_funding = prev_indicators.get('funding_rate') if prev_indicators else None
        p = self._detect_funding_flip(pair_data, prev_funding)
        if p:
            patterns.append(p)
        
        # 15. RSI Divergence
        if prev_indicators:
            p = self._detect_rsi_divergence(indicators, prev_indicators)
            if p:
                patterns.append(p)
        
        # 16. MACD Divergence
        if prev_indicators:
            p = self._detect_macd_divergence(indicators, prev_indicators)
            if p:
                patterns.append(p)
        
        # 17. Squeeze Ignition
        p = self._detect_squeeze_ignition(indicators, pair_data)
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
        """
        Detect extreme funding rate (FAS V2 FUNDING_DIVERGENCE logic).
        
        Multi-level thresholds:
        - funding < -0.001: Strong short squeeze (+70)
        - funding < -0.0005: Moderate short squeeze (+50)
        - funding > +0.001: Long squeeze potential (-50)
        """
        rate = pair_data.latest_funding_rate
        
        # Strong short squeeze (very bullish)
        if rate < -0.001:
            return Pattern(
                pattern_type=PatternType.FUNDING_EXTREME,
                score=70.0,
                confidence=90,
                details={
                    'funding_rate': rate,
                    'squeeze_type': 'STRONG_SHORT_SQUEEZE',
                    'direction': 'BULLISH',
                }
            )
        
        # Moderate short squeeze
        if rate < -0.0005:
            return Pattern(
                pattern_type=PatternType.FUNDING_EXTREME,
                score=50.0,
                confidence=80,
                details={
                    'funding_rate': rate,
                    'squeeze_type': 'SHORT_SQUEEZE',
                    'direction': 'BULLISH',
                }
            )
        
        # Long squeeze potential (bearish)
        if rate > 0.001:
            return Pattern(
                pattern_type=PatternType.FUNDING_EXTREME,
                score=-50.0,
                confidence=80,
                details={
                    'funding_rate': rate,
                    'squeeze_type': 'LONG_SQUEEZE_POTENTIAL',
                    'direction': 'BEARISH',
                }
            )
        
        return None
    
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
    
    def _detect_liquidation_cascade(self, pair_data, indicators) -> Optional[Pattern]:
        """Detect significant liquidation activity (FAS V2 parity)"""
        liq_long = pair_data.liq_long_current
        liq_short = pair_data.liq_short_current
        
        total_liq = liq_long + liq_short
        if total_liq <= 0:
            return None
        
        # Get current volume from indicators
        candles = pair_data.get_last_n_candles(1)
        if len(candles) == 0:
            return None
        
        volume = float(candles['volume'][0]) if candles['volume'][0] > 0 else 1.0
        liq_ratio = total_liq / volume
        
        # FAS V2: threshold is 3% of volume
        if liq_ratio < self.thresholds.liq_ratio_threshold:
            return None
        
        # FAS V2 scoring: 80 if >10%, 60 if >5%, else 40
        if liq_ratio > 0.10:
            base_score = 80.0
        elif liq_ratio > 0.05:
            base_score = 60.0
        else:
            base_score = 40.0
        
        # More long liq = bearish, more short liq = bullish
        if liq_long > liq_short:
            score = -base_score
            direction = 'LONG'
        else:
            score = base_score
            direction = 'SHORT'
        
        confidence = min(70 + liq_ratio * 100, 95)
        
        return Pattern(
            pattern_type=PatternType.LIQUIDATION_CASCADE,
            score=score,
            confidence=confidence,
            details={
                'liquidations_long': liq_long,
                'liquidations_short': liq_short,
                'total_liquidations': total_liq,
                'liquidation_ratio': round(liq_ratio, 4),
                'volume': volume,
                'dominant_side': direction,
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
        Detect momentum exhaustion (FAS V2 logic):
        - RSI at extreme (>70 or <30)
        - Price stalled (small change)
        - Low volume (negative z-score)
        """
        if indicators.rsi is None or indicators.volume_zscore is None:
            return None
        
        rsi = indicators.rsi
        price_change = indicators.price_change_pct if indicators.price_change_pct else 0
        volume_zscore = indicators.volume_zscore
        
        # FAS V2 conditions
        overbought_exhaustion = (
            rsi > 70 and 
            price_change < 0.5 and 
            volume_zscore < -1
        )
        
        oversold_exhaustion = (
            rsi < 30 and 
            price_change > -0.5 and 
            volume_zscore < -1
        )
        
        if overbought_exhaustion:
            score = -40.0  # Bearish reversal
            direction = 'OVERBOUGHT'
            confidence = min(70 + (rsi - 70), 90)
        elif oversold_exhaustion:
            score = 40.0   # Bullish reversal
            direction = 'OVERSOLD'
            confidence = min(70 + (30 - rsi), 90)
        else:
            return None
        
        return Pattern(
            pattern_type=PatternType.MOMENTUM_EXHAUSTION,
            score=score,
            confidence=confidence,
            details={
                'rsi': rsi,
                'price_change': price_change,
                'volume_zscore': volume_zscore,
                'exhaustion_type': direction,
            }
        )

    def _detect_smart_money_divergence(self, indicators) -> Optional[Pattern]:
        """
        Detect Smart Money Divergence (FAS V2 parity):
        - OI increasing while price decreasing → smart money accumulating = BULLISH
        - OI decreasing while price increasing → smart money distributing = BEARISH
        """
        if indicators.oi_delta_pct is None or indicators.price_change_pct is None:
            return None
        
        oi_delta = indicators.oi_delta_pct
        price_change = indicators.price_change_pct
        
        # OI up, price down → Smart money accumulating = BULLISH (FAS V2)
        if oi_delta > 3.0 and price_change < -1.0:
            return Pattern(
                pattern_type=PatternType.SMART_MONEY_DIVERGENCE,
                score=40.0,  # BULLISH - accumulation
                confidence=min(65 + abs(oi_delta) * 3, 85),
                details={
                    'oi_delta_pct': oi_delta,
                    'price_change_pct': price_change,
                    'divergence_type': 'BULLISH',
                }
            )
        
        # OI down, price up → Smart money distributing = BEARISH (FAS V2)
        if oi_delta < -3.0 and price_change > 1.0:
            return Pattern(
                pattern_type=PatternType.SMART_MONEY_DIVERGENCE,
                score=-40.0,  # BEARISH - distribution
                confidence=min(65 + abs(oi_delta) * 3, 85),
                details={
                    'oi_delta_pct': oi_delta,
                    'price_change_pct': price_change,
                    'divergence_type': 'BEARISH',
                }
            )
        
        return None
    
    def _detect_oi_divergence(self, indicators, prev_indicators) -> Optional[Pattern]:
        """
        Detect OI-Price Divergence:
        - OI rising while price falling = bearish divergence
        - OI falling while price rising = bullish divergence
        """
        if prev_indicators is None:
            return None
        if indicators.oi_delta_pct is None or indicators.price_change_pct is None:
            return None
        
        oi_delta = indicators.oi_delta_pct
        price_change = indicators.price_change_pct
        
        # OI up + price down = bearish
        if oi_delta > self.thresholds.oi_divergence_threshold and price_change < -0.5:
            return Pattern(
                pattern_type=PatternType.OI_DIVERGENCE,
                score=-25.0,
                confidence=min(85, 65 + abs(oi_delta)),
                details={'oi_delta': oi_delta, 'price_change': price_change, 'direction': 'BEARISH'}
            )
        
        # OI down + price up = bullish divergence
        if oi_delta < -self.thresholds.oi_divergence_threshold and price_change > 0.5:
            return Pattern(
                pattern_type=PatternType.OI_DIVERGENCE,
                score=25.0,
                confidence=min(85, 65 + abs(oi_delta)),
                details={'oi_delta': oi_delta, 'price_change': price_change, 'direction': 'BULLISH'}
            )
        
        return None
    
    def _detect_funding_flip(self, pair_data, prev_funding: Optional[float] = None) -> Optional[Pattern]:
        """
        Detect Funding Rate Flip (sign change):
        - Negative to Positive = bullish
        - Positive to Negative = bearish
        """
        current_funding = pair_data.latest_funding_rate
        if current_funding is None or prev_funding is None:
            return None
        
        # Positive to Negative flip (bearish)
        if prev_funding > 0.0001 and current_funding < -0.0001:
            return Pattern(
                pattern_type=PatternType.FUNDING_FLIP,
                score=-20.0,
                confidence=75,
                details={'prev_funding': prev_funding, 'current_funding': current_funding, 'direction': 'BEARISH'}
            )
        
        # Negative to Positive flip (bullish)
        if prev_funding < -0.0001 and current_funding > 0.0001:
            return Pattern(
                pattern_type=PatternType.FUNDING_FLIP,
                score=20.0,
                confidence=75,
                details={'prev_funding': prev_funding, 'current_funding': current_funding, 'direction': 'BULLISH'}
            )
        
        return None
    
    def _detect_rsi_divergence(self, indicators, prev_indicators) -> Optional[Pattern]:
        """
        Detect RSI Divergence:
        - Price higher high + RSI lower high = bearish divergence
        - Price lower low + RSI higher low = bullish divergence
        """
        if prev_indicators is None:
            return None
        if indicators.rsi is None or indicators.price_change_pct is None:
            return None
        
        prev_rsi = prev_indicators.get('rsi')
        if prev_rsi is None:
            return None
        
        current_rsi = indicators.rsi
        price_change = indicators.price_change_pct
        
        # Price up but RSI down = bearish divergence
        if price_change > 1.0 and current_rsi < prev_rsi - 5:
            return Pattern(
                pattern_type=PatternType.RSI_DIVERGENCE,
                score=-20.0,
                confidence=70,
                details={'rsi': current_rsi, 'prev_rsi': prev_rsi, 'price_change': price_change, 'type': 'BEARISH'}
            )
        
        # Price down but RSI up = bullish divergence
        if price_change < -1.0 and current_rsi > prev_rsi + 5:
            return Pattern(
                pattern_type=PatternType.RSI_DIVERGENCE,
                score=20.0,
                confidence=70,
                details={'rsi': current_rsi, 'prev_rsi': prev_rsi, 'price_change': price_change, 'type': 'BULLISH'}
            )
        
        return None
    
    def _detect_macd_divergence(self, indicators, prev_indicators) -> Optional[Pattern]:
        """
        Detect MACD Divergence:
        - Price higher high + MACD histogram lower = bearish
        - Price lower low + MACD histogram higher = bullish
        """
        if prev_indicators is None:
            return None
        if indicators.macd_histogram is None or indicators.price_change_pct is None:
            return None
        
        prev_hist = prev_indicators.get('macd_histogram')
        if prev_hist is None:
            return None
        
        current_hist = indicators.macd_histogram
        price_change = indicators.price_change_pct
        
        # Price up but histogram weakening = bearish divergence
        if price_change > 1.0 and current_hist < prev_hist and current_hist > 0:
            return Pattern(
                pattern_type=PatternType.MACD_DIVERGENCE,
                score=-15.0,
                confidence=65,
                details={'histogram': current_hist, 'prev_histogram': prev_hist, 'type': 'BEARISH'}
            )
        
        # Price down but histogram strengthening = bullish divergence  
        if price_change < -1.0 and current_hist > prev_hist and current_hist < 0:
            return Pattern(
                pattern_type=PatternType.MACD_DIVERGENCE,
                score=15.0,
                confidence=65,
                details={'histogram': current_hist, 'prev_histogram': prev_hist, 'type': 'BULLISH'}
            )
        
        return None
    
    def _detect_squeeze_ignition(self, indicators, pair_data) -> Optional[Pattern]:
        """
        Detect Squeeze Ignition (FAS V2):
        - Price change > 2%
        - Volume zscore > 3
        - Funding rate < -0.0003 (shorts squeezed)
        - OI delta > 2%
        """
        if indicators.volume_zscore is None or indicators.price_change_pct is None:
            return None
        
        volume_zscore = indicators.volume_zscore
        price_change = indicators.price_change_pct
        funding_rate = pair_data.latest_funding_rate
        oi_delta = indicators.oi_delta_pct if indicators.oi_delta_pct else 0
        
        # FAS V2 conditions (short squeeze)
        if (price_change > 2.0 and 
            volume_zscore > 3.0 and 
            funding_rate < -0.0003 and 
            oi_delta > 2.0):
            
            confidence = min(70 + abs(funding_rate * 10000), 90)
            
            return Pattern(
                pattern_type=PatternType.SQUEEZE_IGNITION,
                score=50.0,  # Always bullish (short squeeze)
                confidence=confidence,
                details={
                    'price_change': price_change,
                    'volume_zscore': volume_zscore,
                    'funding_rate': funding_rate,
                    'oi_delta': oi_delta,
                    'squeeze_type': 'SHORT_SQUEEZE'
                }
            )
        
        return None

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


def detect_htf_patterns(
    detector: PatternDetector,
    pair_data,
    htf_indicators: dict  # {'1h': IndicatorResult, '4h': ..., '1d': ...}
) -> List[Pattern]:
    """
    Detect patterns from higher timeframe indicators.
    Only returns high-impact patterns (|score| >= 10) matching FAS V2 logic.
    
    Args:
        detector: PatternDetector instance
        pair_data: PairData with HTF candles
        htf_indicators: Dict mapping timeframe to IndicatorResult
    
    Returns:
        List of high-impact patterns from all HTFs
    """
    htf_patterns = []
    
    for timeframe, indicators in htf_indicators.items():
        if indicators is None:
            continue
        
        # Detect patterns using existing methods
        patterns = detector.detect_all(indicators, pair_data)
        
        # Add timeframe tag and filter by score >= 10 (FAS V2 rule)
        for p in patterns:
            if abs(p.score) >= 10:
                p.details['timeframe'] = timeframe
                htf_patterns.append(p)
    
    return htf_patterns


def calculate_multi_tf_score(
    base_patterns: List[Pattern],  # 15m patterns
    htf_patterns: List[Pattern],   # 1h/4h/1d patterns with |score| >= 10
    base_indicator_score: float = 0.0
) -> tuple[float, str, float]:
    """
    Calculate combined score from 15m patterns + HTF patterns.
    Matches FAS V2 calculate_realtime_scores_v2 logic.
    
    total_score = pattern_score + indicator_score
    pattern_score = sum(15m patterns) + sum(HTF patterns with |score| >= 10)
    
    Returns:
        Tuple of (total_score, direction, avg_confidence)
    """
    all_patterns = base_patterns + htf_patterns
    
    if not all_patterns:
        return (base_indicator_score, 'NEUTRAL', 0.0)
    
    pattern_score = sum(p.score for p in all_patterns)
    total_score = pattern_score + base_indicator_score
    avg_confidence = sum(p.confidence for p in all_patterns) / len(all_patterns)
    
    if total_score > 10:
        direction = 'LONG'
    elif total_score < -10:
        direction = 'SHORT'
    else:
        direction = 'NEUTRAL'
    
    return (total_score, direction, avg_confidence)
