"""
Binance Futures WebSocket Connection Manager.

Features:
- Multi-connection pool (200 streams per connection)
- Exponential backoff reconnection with jitter
- Heartbeat monitoring (zombie detection)
- Graceful shutdown
"""

import asyncio
import logging
import random
from dataclasses import dataclass, field
from typing import Callable, Optional
from datetime import datetime

import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException
import orjson

logger = logging.getLogger(__name__)


@dataclass
class ConnectionConfig:
    """Configuration for a single WebSocket connection"""
    base_url: str = "wss://fstream.binance.com/stream"
    max_streams_per_connection: int = 200
    reconnect_min_delay: float = 1.0
    reconnect_max_delay: float = 60.0
    heartbeat_timeout: float = 60.0  # Force reconnect if no message for this long
    ping_interval: float = 20.0


@dataclass
class StreamConfig:
    """Configuration for a stream subscription"""
    symbol: str
    stream_type: str  # 'kline_1m', 'forceOrder', 'markPrice'
    
    @property
    def stream_name(self) -> str:
        """Returns the full stream name for Binance"""
        symbol_lower = self.symbol.lower()
        if self.stream_type == 'kline_1m':
            return f"{symbol_lower}@kline_1m"
        elif self.stream_type == 'forceOrder':
            return f"{symbol_lower}@forceOrder"
        elif self.stream_type == 'markPrice':
            return f"{symbol_lower}@markPrice"
        else:
            raise ValueError(f"Unknown stream type: {self.stream_type}")


class BinanceWSConnection:
    """
    Single WebSocket connection to Binance.
    Handles reconnection with exponential backoff.
    """
    
    def __init__(
        self,
        connection_id: int,
        streams: list[StreamConfig],
        message_callback: Callable,
        config: ConnectionConfig = None,
    ):
        self.connection_id = connection_id
        self.streams = streams
        self.message_callback = message_callback
        self.config = config or ConnectionConfig()
        
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._running = False
        self._last_message_time: Optional[datetime] = None
        self._reconnect_count = 0
        self._task: Optional[asyncio.Task] = None
    
    @property
    def url(self) -> str:
        """Build WebSocket URL with all stream subscriptions"""
        stream_names = [s.stream_name for s in self.streams]
        streams_param = "/".join(stream_names)
        return f"{self.config.base_url}?streams={streams_param}"
    
    async def start(self):
        """Start the connection in a background task"""
        self._running = True
        self._task = asyncio.create_task(self._run_forever())
        logger.info(f"[WS-{self.connection_id}] Started with {len(self.streams)} streams")
    
    async def stop(self):
        """Gracefully stop the connection"""
        self._running = False
        if self._ws:
            await self._ws.close()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info(f"[WS-{self.connection_id}] Stopped")
    
    async def _run_forever(self):
        """Main connection loop with reconnection logic"""
        while self._running:
            try:
                await self._connect_and_listen()
            except asyncio.CancelledError:
                break
            except Exception as e:
                if self._running:
                    delay = self._get_reconnect_delay()
                    logger.warning(
                        f"[WS-{self.connection_id}] Connection error: {e}. "
                        f"Reconnecting in {delay:.1f}s (attempt {self._reconnect_count})"
                    )
                    await asyncio.sleep(delay)
    
    async def _connect_and_listen(self):
        """Establish connection and listen for messages"""
        logger.info(f"[WS-{self.connection_id}] Connecting to {len(self.streams)} streams...")
        
        async with websockets.connect(
            self.url,
            ping_interval=self.config.ping_interval,
            ping_timeout=30,
            close_timeout=10,
        ) as ws:
            self._ws = ws
            self._reconnect_count = 0  # Reset on successful connection
            self._last_message_time = datetime.utcnow()
            
            logger.info(f"[WS-{self.connection_id}] Connected successfully")
            
            # Start heartbeat monitor
            heartbeat_task = asyncio.create_task(self._heartbeat_monitor())
            
            try:
                async for message in ws:
                    self._last_message_time = datetime.utcnow()
                    await self._handle_message(message)
            finally:
                heartbeat_task.cancel()
                try:
                    await heartbeat_task
                except asyncio.CancelledError:
                    pass
    
    async def _handle_message(self, raw_message: str):
        """Parse and route message to callback"""
        try:
            data = orjson.loads(raw_message)
            
            # Combined stream format: {"stream": "btcusdt@kline_1m", "data": {...}}
            if "stream" in data and "data" in data:
                stream_name = data["stream"]
                payload = data["data"]
                
                await self.message_callback(
                    connection_id=self.connection_id,
                    stream_name=stream_name,
                    data=payload,
                )
            else:
                # Direct message (subscription confirmations, etc.)
                logger.debug(f"[WS-{self.connection_id}] Non-stream message: {data}")
                
        except orjson.JSONDecodeError as e:
            logger.error(f"[WS-{self.connection_id}] JSON parse error: {e}")
    
    async def _heartbeat_monitor(self):
        """Monitor for zombie connections"""
        while self._running:
            await asyncio.sleep(10)  # Check every 10 seconds
            
            if self._last_message_time:
                elapsed = (datetime.utcnow() - self._last_message_time).total_seconds()
                if elapsed > self.config.heartbeat_timeout:
                    logger.warning(
                        f"[WS-{self.connection_id}] No message for {elapsed:.0f}s. "
                        f"Forcing reconnect..."
                    )
                    if self._ws:
                        await self._ws.close()
                    break
    
    def _get_reconnect_delay(self) -> float:
        """Calculate reconnect delay with exponential backoff + jitter"""
        self._reconnect_count += 1
        
        # Exponential backoff: 1, 2, 4, 8, 16, 32, max 60
        delay = min(
            self.config.reconnect_min_delay * (2 ** (self._reconnect_count - 1)),
            self.config.reconnect_max_delay
        )
        
        # Add jitter (±25%)
        jitter = delay * 0.25 * (2 * random.random() - 1)
        return delay + jitter


class BinanceWSManager:
    """
    Manages multiple WebSocket connections to Binance Futures.
    Distributes streams across connections respecting the 200 streams/connection limit.
    """
    
    def __init__(
        self,
        symbols: list[str],
        stream_types: list[str],
        message_callback: Callable,
        config: ConnectionConfig = None,
    ):
        """
        Args:
            symbols: List of trading pair symbols (e.g., ['BTCUSDT', 'ETHUSDT'])
            stream_types: List of stream types to subscribe (e.g., ['kline_1m', 'forceOrder'])
            message_callback: Async callback for received messages
            config: Connection configuration
        """
        self.symbols = symbols
        self.stream_types = stream_types
        self.message_callback = message_callback
        self.config = config or ConnectionConfig()
        
        self.connections: list[BinanceWSConnection] = []
        self._running = False
    
    def _build_connections(self) -> list[BinanceWSConnection]:
        """Create connection objects with distributed streams"""
        # Build all streams
        all_streams: list[StreamConfig] = []
        for symbol in self.symbols:
            for stream_type in self.stream_types:
                all_streams.append(StreamConfig(symbol=symbol, stream_type=stream_type))
        
        logger.info(f"Total streams to subscribe: {len(all_streams)}")
        
        # Split into chunks of max_streams_per_connection
        connections = []
        chunk_size = self.config.max_streams_per_connection
        
        for i, chunk_start in enumerate(range(0, len(all_streams), chunk_size)):
            chunk = all_streams[chunk_start:chunk_start + chunk_size]
            conn = BinanceWSConnection(
                connection_id=i,
                streams=chunk,
                message_callback=self.message_callback,
                config=self.config,
            )
            connections.append(conn)
        
        logger.info(f"Created {len(connections)} WebSocket connections")
        return connections
    
    async def start(self):
        """Start all WebSocket connections"""
        self._running = True
        self.connections = self._build_connections()
        
        # Start all connections
        for conn in self.connections:
            await conn.start()
        
        logger.info(f"BinanceWSManager started with {len(self.connections)} connections")
    
    async def stop(self):
        """Stop all WebSocket connections gracefully"""
        self._running = False
        
        # Stop all connections concurrently
        await asyncio.gather(*[conn.stop() for conn in self.connections])
        
        logger.info("BinanceWSManager stopped")
    
    async def add_symbols(self, symbols: list[str]):
        """Add new symbols to WebSocket subscriptions dynamically."""
        if not symbols:
            return
        
        # Build new streams for the symbols
        new_streams: list[StreamConfig] = []
        for symbol in symbols:
            for stream_type in self.stream_types:
                new_streams.append(StreamConfig(symbol=symbol, stream_type=stream_type))
        
        logger.info(f"Adding {len(new_streams)} new streams for {len(symbols)} symbols...")
        
        # For simplicity, create a new connection for new streams
        # (Could optimize later to fill existing connections)
        if new_streams:
            conn_id = len(self.connections)
            conn = BinanceWSConnection(
                connection_id=conn_id,
                streams=new_streams,
                message_callback=self.message_callback,
                config=self.config,
            )
            self.connections.append(conn)
            await conn.start()
            
            self.symbols.extend(symbols)
            logger.info(f"Added connection WS-{conn_id} with {len(new_streams)} streams")
    
    @property
    def stats(self) -> dict:
        """Get connection statistics"""
        return {
            "total_connections": len(self.connections),
            "total_streams": sum(len(c.streams) for c in self.connections),
            "is_running": self._running,
        }
