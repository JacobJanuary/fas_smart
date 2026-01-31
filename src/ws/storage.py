"""
In-memory storage for real-time market data.

Features:
- NumPy-based ring buffer for efficient storage
- 7-day history per trading pair
- Thread-safe access
- Rolling window aggregation
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


# NumPy structured dtype for 1m candle
CANDLE_DTYPE = np.dtype([
    ('timestamp', 'i8'),    # Unix ms
    ('open', 'f8'),
    ('high', 'f8'),
    ('low', 'f8'),
    ('close', 'f8'),
    ('volume', 'f8'),
    ('buy_volume', 'f8'),
    ('funding_rate', 'f8'),
])

# Buffer sizes per timeframe
DEFAULT_BUFFER_SIZE = 10080  # 7 days = 10080 minutes
HTF_BUFFER_SIZE = 50  # Higher timeframes: 35 (MACD) + padding

# Supported higher timeframes
HIGHER_TIMEFRAMES = ['1h', '4h', '1d']


@dataclass
class PairData:
    """Data storage for a single trading pair"""
    pair_id: int
    symbol: str
    
    # Ring buffer for 1m candles
    candles: np.ndarray = field(default=None)
    write_idx: int = 0
    candle_count: int = 0
    
    # Latest funding rate (updated via markPrice stream)
    latest_funding_rate: float = 0.0
    latest_open_interest: float = 0.0  # Updated via REST
    prev_open_interest: float = 0.0    # For OI delta calculation
    
    # Liquidity tier (cached, updated periodically)
    tier: str = 'TIER_2'  # Default to TIER_2
    
    # Liquidations accumulator for current minute
    liq_long_current: float = 0.0
    liq_short_current: float = 0.0
    liq_last_reset: int = 0  # Timestamp of last reset
    
    # EMA caches for efficient calculation
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None
    ema_signal: Optional[float] = None
    ema_rsi_gain: Optional[float] = None
    ema_rsi_loss: Optional[float] = None
    
    # CVD and Imbalance tracking (for FAS V2 parity)
    cvd_cumulative: float = 0.0           # Running CVD sum
    prev_cvd_cumulative: float = 0.0      # For delta comparison
    smoothed_imbalance: float = 0.0       # (SMA3 + SMA6) / 2 of normalized imbalance
    imbalance_history: list = field(default_factory=list)  # Last 6 values for SMA

    
    # Previous indicators for crossover detection
    prev_macd_histogram: Optional[float] = None
    prev_rsi: Optional[float] = None
    prev_price_change: Optional[float] = None
    
    # Higher timeframe candle buffers (for multi-TF pattern detection)
    candles_1h: np.ndarray = field(default=None)
    candles_4h: np.ndarray = field(default=None)
    candles_1d: np.ndarray = field(default=None)
    
    # HTF write indices and counts
    htf_write_idx: Dict[str, int] = field(default_factory=dict)
    htf_candle_count: Dict[str, int] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.candles is None:
            self.candles = np.zeros(DEFAULT_BUFFER_SIZE, dtype=CANDLE_DTYPE)
        # Initialize HTF buffers
        if self.candles_1h is None:
            self.candles_1h = np.zeros(HTF_BUFFER_SIZE, dtype=CANDLE_DTYPE)
            self.candles_4h = np.zeros(HTF_BUFFER_SIZE, dtype=CANDLE_DTYPE)
            self.candles_1d = np.zeros(HTF_BUFFER_SIZE, dtype=CANDLE_DTYPE)
            self.htf_write_idx = {'1h': 0, '4h': 0, '1d': 0}
            self.htf_candle_count = {'1h': 0, '4h': 0, '1d': 0}
    
    def add_candle(self, timestamp: int, o: float, h: float, l: float, 
                   c: float, volume: float, buy_volume: float):
        """Add a new candle to the ring buffer and update derived metrics"""
        self.candles[self.write_idx] = (
            timestamp, o, h, l, c, volume, buy_volume, self.latest_funding_rate
        )
        self.write_idx = (self.write_idx + 1) % DEFAULT_BUFFER_SIZE
        self.candle_count = min(self.candle_count + 1, DEFAULT_BUFFER_SIZE)
        
        # Update CVD cumulative: buy_volume - sell_volume (in notional terms)
        # CVD delta = (buy_volume - sell_volume) = 2*buy_volume - total_volume
        if volume > 0:
            cvd_delta = (2 * buy_volume - volume) * c  # In notional (USD)
            self.prev_cvd_cumulative = self.cvd_cumulative
            self.cvd_cumulative += cvd_delta
            
            # Update smoothed imbalance: (SMA3 + SMA6) / 2 (FAS V2 parity)
            norm_imbalance = (2 * buy_volume - volume) / volume
            self.imbalance_history.append(norm_imbalance)
            if len(self.imbalance_history) > 6:
                self.imbalance_history.pop(0)
            
            # Calculate SMA3 and SMA6
            if len(self.imbalance_history) >= 3:
                sma3 = sum(self.imbalance_history[-3:]) / 3
                sma6 = sum(self.imbalance_history[-6:]) / min(6, len(self.imbalance_history))
                self.smoothed_imbalance = (sma3 + sma6) / 2
    
    def add_htf_candle(self, timeframe: str, timestamp: int, o: float, h: float, 
                       l: float, c: float, volume: float, buy_volume: float,
                       funding_rate: float = 0.0):
        """Add a candle to higher timeframe buffer (1h, 4h, 1d)
        
        Args:
            funding_rate: Latest funding rate at this candle time (FAS V2 parity)
        """
        if timeframe == '1h':
            buffer = self.candles_1h
        elif timeframe == '4h':
            buffer = self.candles_4h
        elif timeframe == '1d':
            buffer = self.candles_1d
        else:
            return
        
        idx = self.htf_write_idx.get(timeframe, 0)
        buffer[idx] = (timestamp, o, h, l, c, volume, buy_volume, funding_rate)
        self.htf_write_idx[timeframe] = (idx + 1) % HTF_BUFFER_SIZE
        self.htf_candle_count[timeframe] = min(
            self.htf_candle_count.get(timeframe, 0) + 1, HTF_BUFFER_SIZE
        )
    
    def get_htf_candles(self, timeframe: str, n: int = 24) -> np.ndarray:
        """Get last N candles from higher timeframe buffer"""
        if timeframe == '1h':
            buffer = self.candles_1h
        elif timeframe == '4h':
            buffer = self.candles_4h
        elif timeframe == '1d':
            buffer = self.candles_1d
        else:
            return np.array([])
        
        count = min(n, self.htf_candle_count.get(timeframe, 0))
        if count == 0:
            return np.array([])
        
        idx = self.htf_write_idx.get(timeframe, 0)
        indices = [(idx - 1 - i) % HTF_BUFFER_SIZE for i in range(count)]
        return buffer[indices[::-1]]
    
    def get_last_n_candles(self, n: int) -> np.ndarray:
        """Get the last N candles in chronological order"""
        if n > self.candle_count:
            n = self.candle_count
        
        if n == 0:
            return np.array([], dtype=CANDLE_DTYPE)
        
        # Calculate start position
        if self.candle_count < DEFAULT_BUFFER_SIZE:
            # Buffer not full yet
            start = max(0, self.write_idx - n)
            return self.candles[start:self.write_idx]
        else:
            # Buffer is full, handle wrap-around
            end = self.write_idx
            start = (end - n) % DEFAULT_BUFFER_SIZE
            
            if start < end:
                return self.candles[start:end]
            else:
                return np.concatenate([
                    self.candles[start:],
                    self.candles[:end]
                ])
    
    def aggregate_rolling_window(self, window_minutes: int = 15) -> Optional[dict]:
        """
        Aggregate the last N 1-minute candles into a single OHLCV.
        
        Returns dict with: open, high, low, close, volume, buy_volume
        """
        candles = self.get_last_n_candles(window_minutes)
        
        if len(candles) < window_minutes:
            return None  # Not enough data
        
        return {
            'open': float(candles[0]['open']),
            'high': float(np.max(candles['high'])),
            'low': float(np.min(candles['low'])),
            'close': float(candles[-1]['close']),
            'volume': float(np.sum(candles['volume'])),
            'buy_volume': float(np.sum(candles['buy_volume'])),
            'funding_rate': float(candles[-1]['funding_rate']),
        }
    
    def get_close_prices(self, n: int) -> np.ndarray:
        """Get array of close prices for the last N candles"""
        candles = self.get_last_n_candles(n)
        return candles['close'].astype(np.float64)
    
    def get_volumes(self, n: int) -> np.ndarray:
        """Get array of volumes for the last N candles"""
        candles = self.get_last_n_candles(n)
        return candles['volume'].astype(np.float64)
    
    def get_high_low_close(self, n: int) -> tuple:
        """Get arrays of high, low, close prices for the last N candles"""
        candles = self.get_last_n_candles(n)
        return (
            candles['high'].astype(np.float64),
            candles['low'].astype(np.float64),
            candles['close'].astype(np.float64)
        )


class DataStore:
    """
    Central storage for all trading pair data.
    Thread-safe singleton.
    """
    
    _instance = None
    _lock = asyncio.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.pairs: Dict[str, PairData] = {}  # symbol -> PairData
        self.pair_id_map: Dict[int, str] = {}  # pair_id -> symbol
        self._initialized = True
        
        logger.info("DataStore initialized")
    
    def register_pair(self, pair_id: int, symbol: str):
        """Register a trading pair for data collection"""
        if symbol not in self.pairs:
            self.pairs[symbol] = PairData(pair_id=pair_id, symbol=symbol)
            self.pair_id_map[pair_id] = symbol
            logger.debug(f"Registered pair: {symbol} (id={pair_id})")
    
    def register_pairs_from_db(self, pairs: list[tuple[int, str]]):
        """Register multiple pairs from database query result"""
        for pair_id, symbol in pairs:
            self.register_pair(pair_id, symbol)
        logger.info(f"Registered {len(pairs)} trading pairs")
    
    async def add_candle(self, candle) -> bool:
        """
        Add a closed candle to storage.
        
        Args:
            candle: Candle dataclass from handlers.py
        
        Returns:
            True if added successfully
        """
        symbol = candle.symbol
        if symbol not in self.pairs:
            logger.warning(f"Unknown symbol in candle: {symbol}")
            return False
        
        pair_data = self.pairs[symbol]
        pair_data.add_candle(
            timestamp=candle.timestamp,
            o=candle.open,
            h=candle.high,
            l=candle.low,
            c=candle.close,
            volume=candle.volume,
            buy_volume=candle.buy_volume,
        )
        return True
    
    async def add_htf_candle(self, candle, timeframe: str) -> bool:
        """Add a closed candle to higher timeframe storage (1h, 4h, 1d)"""
        symbol = candle.symbol
        if symbol not in self.pairs:
            return False
        
        pair_data = self.pairs[symbol]
        pair_data.add_htf_candle(
            timeframe=timeframe,
            timestamp=candle.timestamp,
            o=candle.open,
            h=candle.high,
            l=candle.low,
            c=candle.close,
            volume=candle.volume,
            buy_volume=candle.buy_volume,
            funding_rate=pair_data.latest_funding_rate,  # FAS V2 parity
        )
        return True
    
    async def add_liquidation(self, liquidation) -> bool:
        """Add liquidation to current 15m candle accumulator (FAS V2 parity)"""
        symbol = liquidation.symbol
        if symbol not in self.pairs:
            return False
        
        pair_data = self.pairs[symbol]
        
        # Reset accumulator if 15m candle changed (FAS V2 uses 15m aggregated data)
        current_15m = liquidation.timestamp // 900000 * 900000  # 900000ms = 15 minutes
        if current_15m != pair_data.liq_last_reset:
            pair_data.liq_long_current = 0.0
            pair_data.liq_short_current = 0.0
            pair_data.liq_last_reset = current_15m
        
        if liquidation.side == 'L':
            pair_data.liq_long_current += liquidation.quantity
        else:
            pair_data.liq_short_current += liquidation.quantity
        
        return True
    
    async def update_funding_rate(self, mark_price) -> bool:
        """Update latest funding rate for a symbol"""
        symbol = mark_price.symbol
        if symbol not in self.pairs:
            return False
        
        self.pairs[symbol].latest_funding_rate = mark_price.funding_rate
        return True
    
    async def update_open_interest(self, symbol: str, oi: float) -> bool:
        """Update open interest (from REST API)"""
        if symbol not in self.pairs:
            return False
        
        self.pairs[symbol].latest_open_interest = oi
        return True
    
    def get_pair(self, symbol: str) -> Optional[PairData]:
        """Get PairData for a symbol"""
        return self.pairs.get(symbol)
    
    def get_pair_by_id(self, pair_id: int) -> Optional[PairData]:
        """Get PairData by pair_id"""
        symbol = self.pair_id_map.get(pair_id)
        if symbol:
            return self.pairs.get(symbol)
        return None
    
    def get_all_symbols(self) -> list[str]:
        """Get list of all registered symbols"""
        return list(self.pairs.keys())
    
    def get_stats(self) -> dict:
        """Get storage statistics"""
        total_candles = sum(p.candle_count for p in self.pairs.values())
        return {
            'total_pairs': len(self.pairs),
            'total_candles': total_candles,
            'avg_candles_per_pair': total_candles / len(self.pairs) if self.pairs else 0,
            'memory_mb': (len(self.pairs) * DEFAULT_BUFFER_SIZE * CANDLE_DTYPE.itemsize) / 1024 / 1024,
        }
