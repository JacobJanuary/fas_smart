"""
Message handlers for Binance WebSocket streams.

Parses raw messages and updates in-memory storage.
"""

import logging
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Candle:
    """Parsed 1-minute candle data"""
    symbol: str
    timestamp: int  # Close time in ms
    open: float
    high: float
    low: float
    close: float
    volume: float
    buy_volume: float  # Taker buy volume
    is_closed: bool


@dataclass
class Liquidation:
    """Parsed liquidation event"""
    symbol: str
    timestamp: int
    side: str  # 'L' = long liquidated, 'S' = short liquidated
    quantity: float
    price: float


@dataclass
class MarkPrice:
    """Parsed mark price with funding rate"""
    symbol: str
    timestamp: int
    mark_price: float
    index_price: float
    funding_rate: float
    next_funding_time: int


def parse_kline(data: dict) -> Optional[Candle]:
    """
    Parse kline (candlestick) message.
    
    Input format:
    {
        "e": "kline",
        "E": 1638747660000,    # Event time
        "s": "BTCUSDT",         # Symbol
        "k": {
            "t": 1638747600000, # Kline start time
            "T": 1638747659999, # Kline close time
            "s": "BTCUSDT",
            "i": "1m",           # Interval
            "o": "48000.00",     # Open
            "c": "48050.00",     # Close
            "h": "48100.00",     # High
            "l": "47950.00",     # Low
            "v": "1000.5",       # Volume
            "V": "600.3",        # Taker buy volume
            "x": false           # Is closed?
        }
    }
    """
    try:
        k = data.get('k', {})
        return Candle(
            symbol=data.get('s', ''),
            timestamp=k.get('T', 0),  # Close time
            open=float(k.get('o', 0)),
            high=float(k.get('h', 0)),
            low=float(k.get('l', 0)),
            close=float(k.get('c', 0)),
            volume=float(k.get('v', 0)),
            buy_volume=float(k.get('V', 0)),
            is_closed=k.get('x', False),
        )
    except (ValueError, KeyError, TypeError) as e:
        logger.error(f"Failed to parse kline: {e}, data: {data}")
        return None


def parse_force_order(data: dict) -> Optional[Liquidation]:
    """
    Parse force order (liquidation) message.
    
    Input format:
    {
        "e": "forceOrder",
        "E": 1638747660000,
        "o": {
            "s": "BTCUSDT",
            "S": "SELL",           # Side: SELL = long liquidated
            "q": "0.500",          # Quantity
            "p": "48000.00",       # Price
            "ap": "47990.00",      # Average price
            "T": 1638747659999     # Trade time
        }
    }
    """
    try:
        o = data.get('o', {})
        side = o.get('S', '')
        
        return Liquidation(
            symbol=o.get('s', ''),
            timestamp=o.get('T', 0),
            side='L' if side == 'SELL' else 'S',  # SELL = long liquidated
            quantity=float(o.get('q', 0)),
            price=float(o.get('ap', 0) or o.get('p', 0)),  # Prefer average price
        )
    except (ValueError, KeyError, TypeError) as e:
        logger.error(f"Failed to parse force order: {e}, data: {data}")
        return None


def parse_mark_price(data: dict) -> Optional[MarkPrice]:
    """
    Parse mark price message.
    
    Input format:
    {
        "e": "markPriceUpdate",
        "E": 1638747660000,
        "s": "BTCUSDT",
        "p": "48000.00",           # Mark price
        "i": "48010.00",           # Index price
        "r": "0.00010000",         # Funding rate
        "T": 1638748800000         # Next funding time
    }
    """
    try:
        return MarkPrice(
            symbol=data.get('s', ''),
            timestamp=data.get('E', 0),
            mark_price=float(data.get('p', 0)),
            index_price=float(data.get('i', 0)),
            funding_rate=float(data.get('r', 0)),
            next_funding_time=data.get('T', 0),
        )
    except (ValueError, KeyError, TypeError) as e:
        logger.error(f"Failed to parse mark price: {e}, data: {data}")
        return None


class MessageRouter:
    """
    Routes incoming WebSocket messages to appropriate handlers
    and updates the data store.
    """
    
    def __init__(self, data_store):
        """
        Args:
            data_store: DataStore instance for storing parsed data
        """
        self.data_store = data_store
        
        # Statistics
        self.stats = {
            'klines_received': 0,
            'klines_closed': 0,
            'liquidations_received': 0,
            'mark_prices_received': 0,
            'errors': 0,
        }
    
    async def handle_message(self, connection_id: int, stream_name: str, data: dict):
        """
        Route message to appropriate handler based on stream type.
        
        Args:
            connection_id: ID of the WebSocket connection
            stream_name: Full stream name (e.g., 'btcusdt@kline_1m')
            data: Parsed message payload
        """
        try:
            if '@kline_' in stream_name:
                await self._handle_kline(data)
            elif '@forceOrder' in stream_name:
                await self._handle_force_order(data)
            elif '@markPrice' in stream_name:
                await self._handle_mark_price(data)
            else:
                logger.debug(f"Unknown stream type: {stream_name}")
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Error handling message from {stream_name}: {e}")
    
    async def _handle_kline(self, data: dict):
        """Handle kline message"""
        candle = parse_kline(data)
        if candle:
            self.stats['klines_received'] += 1
            
            # Only store closed candles
            if candle.is_closed:
                self.stats['klines_closed'] += 1
                await self.data_store.add_candle(candle)
                logger.debug(f"Stored closed candle: {candle.symbol} @ {candle.timestamp}")
    
    async def _handle_force_order(self, data: dict):
        """Handle liquidation message"""
        liquidation = parse_force_order(data)
        if liquidation:
            self.stats['liquidations_received'] += 1
            await self.data_store.add_liquidation(liquidation)
    
    async def _handle_mark_price(self, data: dict):
        """Handle mark price message"""
        mark_price = parse_mark_price(data)
        if mark_price:
            self.stats['mark_prices_received'] += 1
            await self.data_store.update_funding_rate(mark_price)
    
    def get_stats(self) -> dict:
        """Get handler statistics"""
        return dict(self.stats)
