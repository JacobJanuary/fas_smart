"""
FAS Smart Main Service.

Orchestrates WebSocket connections, data collection,
indicator calculation, and signal generation.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

from config import config
from db import get_cursor
from ws.connection import BinanceWSManager, ConnectionConfig
from ws.handlers import MessageRouter
from ws.storage import DataStore
from ws.rest_poller import OpenInterestPoller
from calc.indicators import calculate_all_indicators
from calc.patterns import PatternDetector, calculate_total_score

logger = logging.getLogger(__name__)


class FASService:
    """
    Main service that orchestrates real-time signal generation.
    
    Flow:
    1. Load trading pairs from database
    2. Start WebSocket connections for klines, liquidations, markPrice
    3. Start OI REST poller
    4. Every minute: calculate indicators, detect patterns, save signals
    """
    
    def __init__(self):
        self.data_store = DataStore()
        self.message_router = MessageRouter(self.data_store)
        self.pattern_detector = PatternDetector()
        
        self.ws_manager: Optional[BinanceWSManager] = None
        self.oi_poller: Optional[OpenInterestPoller] = None
        
        self._running = False
        self._calculation_task: Optional[asyncio.Task] = None
        
        # Previous indicators for each pair (for pattern detection)
        self._prev_indicators: dict = {}
    
    async def start(self):
        """Start the service"""
        logger.info("Starting FAS Smart Service...")
        
        # 1. Load pairs from database
        pairs = self._load_pairs()
        if not pairs:
            logger.error("No trading pairs found in database!")
            return
        
        # 2. Register pairs in data store
        self.data_store.register_pairs_from_db(pairs)
        symbols = [symbol for _, symbol in pairs]
        
        # 3. Start WebSocket manager
        ws_config = ConnectionConfig(
            max_streams_per_connection=200,
            heartbeat_timeout=60.0,
        )
        
        self.ws_manager = BinanceWSManager(
            symbols=symbols,
            stream_types=['kline_1m', 'forceOrder', 'markPrice'],
            message_callback=self.message_router.handle_message,
            config=ws_config,
        )
        await self.ws_manager.start()
        
        # 4. Start OI poller
        self.oi_poller = OpenInterestPoller(
            data_store=self.data_store,
            poll_interval=30.0,
        )
        await self.oi_poller.start()
        
        # 5. Start calculation loop
        self._running = True
        self._calculation_task = asyncio.create_task(self._calculation_loop())
        
        logger.info(f"FAS Smart Service started with {len(symbols)} pairs")
    
    async def stop(self):
        """Stop the service gracefully"""
        logger.info("Stopping FAS Smart Service...")
        self._running = False
        
        if self._calculation_task:
            self._calculation_task.cancel()
            try:
                await self._calculation_task
            except asyncio.CancelledError:
                pass
        
        if self.oi_poller:
            await self.oi_poller.stop()
        
        if self.ws_manager:
            await self.ws_manager.stop()
        
        logger.info("FAS Smart Service stopped")
    
    def _load_pairs(self) -> list[tuple[int, str]]:
        """Load active trading pairs from database"""
        with get_cursor() as cur:
            cur.execute("""
                SELECT id, symbol 
                FROM fas_smart.trading_pairs 
                WHERE is_active = true
                ORDER BY avg_volume_24h DESC
            """)
            return [(row[0], row[1]) for row in cur.fetchall()]
    
    async def _calculation_loop(self):
        """Main calculation loop - runs every minute"""
        logger.info("Calculation loop started")
        
        while self._running:
            try:
                # Wait until the next minute boundary
                await self._wait_for_minute()
                
                if not self._running:
                    break
                
                # Run calculations
                start_time = datetime.now(timezone.utc)
                signals = await self._process_all_pairs()
                elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
                
                logger.info(
                    f"Calculation complete: {len(signals)} signals generated "
                    f"in {elapsed:.2f}s"
                )
                
                # Log stats periodically
                self._log_stats()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Calculation loop error: {e}", exc_info=True)
                await asyncio.sleep(5)
    
    async def _wait_for_minute(self):
        """Wait until the start of the next minute"""
        now = datetime.now(timezone.utc)
        seconds_to_wait = 60 - now.second - now.microsecond / 1_000_000
        
        # Add small buffer to ensure candle is closed
        seconds_to_wait += 1.0
        
        if seconds_to_wait > 0:
            await asyncio.sleep(seconds_to_wait)
    
    async def _process_all_pairs(self) -> list[dict]:
        """Process all pairs and generate signals"""
        signals = []
        
        for symbol in self.data_store.get_all_symbols():
            try:
                signal = await self._process_pair(symbol)
                if signal:
                    signals.append(signal)
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
        
        # Save signals to database
        if signals:
            self._save_signals(signals)
        
        return signals
    
    async def _process_pair(self, symbol: str) -> Optional[dict]:
        """Process a single trading pair"""
        pair_data = self.data_store.get_pair(symbol)
        if not pair_data:
            return None
        
        # Check if we have enough data
        if pair_data.candle_count < config.ROLLING_WINDOW_MINUTES:
            return None
        
        # Calculate indicators
        indicators = calculate_all_indicators(pair_data, config.ROLLING_WINDOW_MINUTES)
        
        # Get previous indicators for pattern detection
        prev = self._prev_indicators.get(symbol, {})
        
        # Detect patterns
        patterns = self.pattern_detector.detect_all(indicators, pair_data, prev)
        
        # Calculate total score
        total_score, direction, confidence = calculate_total_score(patterns)
        
        # Store current indicators for next iteration
        self._prev_indicators[symbol] = {
            'macd_line': indicators.macd_line,
            'macd_signal': indicators.macd_signal,
            'rsi': indicators.rsi,
        }
        
        # Only generate signal if score is significant
        if abs(total_score) < 10:
            return None
        
        # Get current price
        candles = pair_data.get_last_n_candles(1)
        entry_price = float(candles[-1]['close']) if len(candles) > 0 else 0.0
        
        return {
            'trading_pair_id': pair_data.pair_id,
            'symbol': symbol,
            'timestamp': datetime.now(timezone.utc),
            'total_score': total_score,
            'direction': direction,
            'confidence': confidence,
            'entry_price': entry_price,
            'patterns': [
                {
                    'type': p.pattern_type.value,
                    'score': p.score,
                    'confidence': p.confidence,
                    'details': p.details,
                }
                for p in patterns
            ],
            'indicators': {
                'rsi': indicators.rsi,
                'macd_histogram': indicators.macd_histogram,
                'volume_zscore': indicators.volume_zscore,
                'atr': indicators.atr,
            },
        }
    
    def _save_signals(self, signals: list[dict]):
        """Save signals to database"""
        with get_cursor() as cur:
            for signal in signals:
                try:
                    # Insert signal
                    cur.execute("""
                        INSERT INTO fas_smart.signals 
                        (trading_pair_id, timestamp, total_score, direction, 
                         confidence, entry_price, details)
                        VALUES (%s, %s, %s, %s::fas_smart.signal_direction, 
                                %s, %s, %s::jsonb)
                        RETURNING id
                    """, (
                        signal['trading_pair_id'],
                        signal['timestamp'],
                        signal['total_score'],
                        signal['direction'],
                        signal['confidence'],
                        signal['entry_price'],
                        str(signal),  # JSON details
                    ))
                    
                    signal_id = cur.fetchone()[0]
                    
                    # Insert patterns
                    for pattern in signal['patterns']:
                        cur.execute("""
                            INSERT INTO fas_smart.signal_patterns
                            (signal_id, pattern_type, timeframe, score_impact, 
                             confidence, details)
                            VALUES (%s, %s::fas_smart.pattern_type, '15m', %s, %s, %s::jsonb)
                        """, (
                            signal_id,
                            pattern['type'],
                            pattern['score'],
                            pattern['confidence'],
                            str(pattern['details']),
                        ))
                    
                    # Update last_signal_at on trading_pair
                    cur.execute("""
                        UPDATE fas_smart.trading_pairs
                        SET last_signal_at = NOW()
                        WHERE id = %s
                    """, (signal['trading_pair_id'],))
                    
                    logger.info(
                        f"Signal saved: {signal['symbol']} "
                        f"score={signal['total_score']:.1f} "
                        f"dir={signal['direction']}"
                    )
                    
                except Exception as e:
                    logger.error(f"Failed to save signal for {signal['symbol']}: {e}")
    
    def _log_stats(self):
        """Log service statistics"""
        ws_stats = self.ws_manager.stats if self.ws_manager else {}
        router_stats = self.message_router.get_stats()
        storage_stats = self.data_store.get_stats()
        oi_stats = self.oi_poller.get_stats() if self.oi_poller else {}
        
        logger.debug(
            f"Stats - WS: {ws_stats}, Router: {router_stats}, "
            f"Storage: {storage_stats}, OI: {oi_stats}"
        )


async def run_service():
    """Entry point for running the service"""
    service = FASService()
    
    try:
        await service.start()
        
        # Run forever (until interrupted)
        while True:
            await asyncio.sleep(60)
            
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        await service.stop()


if __name__ == "__main__":
    asyncio.run(run_service())
