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
from calc.indicators import calculate_all_indicators, calculate_indicator_score
from calc.patterns import PatternDetector, calculate_total_score
from calc.market_regime import calculate_market_regime, adjust_score_for_regime

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
                
                # Save latest candles to DB for monitoring
                self._save_candles_to_db()
                
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
        """Wait until the start of the next minute + buffer for candle delivery"""
        now = datetime.now(timezone.utc)
        seconds_to_wait = 60 - now.second - now.microsecond / 1_000_000
        
        # Add buffer to ensure all candles are received via WebSocket
        # Binance sends closed candles at :00.000, but network latency can delay them
        seconds_to_wait += 5.0
        
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
            # Log first pair's progress every cycle
            if symbol == 'BTCUSDT':
                logger.debug(f"BTCUSDT waiting: {pair_data.candle_count}/{config.ROLLING_WINDOW_MINUTES} candles")
            return None
        
        # Calculate indicators
        indicators = calculate_all_indicators(pair_data, config.ROLLING_WINDOW_MINUTES)
        
        # Get previous indicators for pattern detection
        prev = self._prev_indicators.get(symbol, {})
        
        # Detect patterns
        patterns = self.pattern_detector.detect_all(indicators, pair_data, prev)
        
        # Calculate pattern score
        pattern_score, direction, confidence = calculate_total_score(patterns)
        
        # Calculate indicator score (FAS V2 parity)
        indicator_score = calculate_indicator_score(indicators, pair_data)
        
        # Apply market regime adjustment (FAS V2 parity)
        btc_data = self.data_store.get_pair_data('BTCUSDT')
        if btc_data and btc_data.candle_count >= 17:
            regime = calculate_market_regime(btc_data)
            indicator_score = adjust_score_for_regime(indicator_score, regime)
            # Log regime for BTC
            if symbol == 'BTCUSDT':
                logger.debug(f"Market regime: {regime.regime} (str={regime.strength:.2f}, btc_4h={regime.btc_change_4h:.2f}%)")
        
        # Total score = pattern_score + indicator_score (matching FAS V2)
        total_score = pattern_score + indicator_score
        
        # Adjust direction based on total score
        if total_score > 10:
            direction = 'LONG'
        elif total_score < -10:
            direction = 'SHORT'
        else:
            direction = 'NEUTRAL'
        
        # Log patterns for debugging
        if symbol == 'BTCUSDT':
            logger.debug(f"BTCUSDT: patterns={len(patterns)}, p_score={pattern_score:.1f}, i_score={indicator_score:.1f}, total={total_score:.1f}")
        
        # Store current indicators for next iteration
        self._prev_indicators[symbol] = {
            'macd_line': indicators.macd_line,
            'macd_signal': indicators.macd_signal,
            'rsi': indicators.rsi,
        }
        
        # Update pair_data prev values for crossover detection
        pair_data.prev_macd_histogram = indicators.macd_histogram
        pair_data.prev_rsi = indicators.rsi
        pair_data.prev_price_change = indicators.price_change_pct
        
        # Update prev OI for next delta calculation
        if pair_data.latest_open_interest > 0:
            pair_data.prev_open_interest = pair_data.latest_open_interest
        
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
        import json
        
        with get_cursor() as cur:
            for signal in signals:
                try:
                    # Calculate pattern and indicator scores
                    pattern_score = sum(p['score'] for p in signal['patterns'])
                    indicator_score = signal['total_score'] - pattern_score
                    
                    # Insert signal (total_score is generated column)
                    cur.execute("""
                        INSERT INTO fas_smart.signals 
                        (pair_id, ts, pattern_score, indicator_score, direction, 
                         confidence, entry_price, patterns_json, indicators_json)
                        VALUES (%s, %s, %s, %s, %s::fas_smart.signal_direction, 
                                %s, %s, %s::jsonb, %s::jsonb)
                        ON CONFLICT (pair_id, ts) DO UPDATE SET
                            pattern_score = EXCLUDED.pattern_score,
                            indicator_score = EXCLUDED.indicator_score,
                            direction = EXCLUDED.direction,
                            confidence = EXCLUDED.confidence
                        RETURNING id
                    """, (
                        signal['trading_pair_id'],
                        signal['timestamp'],
                        pattern_score,
                        indicator_score,
                        signal['direction'],
                        signal['confidence'],
                        signal['entry_price'],
                        json.dumps(signal['patterns']),
                        json.dumps(signal['indicators']),
                    ))
                    
                    signal_id = cur.fetchone()[0]
                    
                    # Insert patterns
                    for pattern in signal['patterns']:
                        cur.execute("""
                            INSERT INTO fas_smart.signal_patterns
                            (signal_id, pattern, timeframe, score_impact, 
                             confidence, details)
                            VALUES (%s, %s::fas_smart.pattern_type, '15m'::fas_smart.timeframe_enum, %s, %s, %s::jsonb)
                            ON CONFLICT (signal_id, pattern, timeframe) DO NOTHING
                        """, (
                            signal_id,
                            pattern['type'],
                            pattern['score'],
                            pattern['confidence'],
                            json.dumps(pattern['details']),
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
        
        logger.info(
            f"Stats - Klines: {router_stats.get('klines_closed', 0)}, "
            f"OI: {oi_stats.get('polls_success', 0)}, "
            f"Mem: {storage_stats.get('memory_mb', 0):.1f}MB"
        )
    
    def _save_candles_to_db(self):
        """Save latest candle for each pair to database for monitoring"""
        saved = 0
        
        with get_cursor() as cur:
            for symbol in self.data_store.get_all_symbols():
                pair_data = self.data_store.get_pair(symbol)
                if not pair_data or pair_data.candle_count == 0:
                    continue
                
                # Get the latest candle
                candles = pair_data.get_last_n_candles(1)
                if len(candles) == 0:
                    continue
                
                c = candles[0]
                try:
                    cur.execute("""
                        INSERT INTO fas_smart.candles_1m 
                        (pair_id, ts, o, h, l, c, v, bv, oi, fr)
                        VALUES (%s, to_timestamp(%s/1000.0), %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (pair_id, ts) DO NOTHING
                    """, (
                        pair_data.pair_id,
                        int(c['timestamp']),
                        float(c['open']),
                        float(c['high']),
                        float(c['low']),
                        float(c['close']),
                        float(c['volume']),
                        float(c['buy_volume']),
                        pair_data.latest_open_interest,
                        pair_data.latest_funding_rate,
                    ))
                    saved += 1
                except Exception as e:
                    logger.debug(f"Failed to save candle for {symbol}: {e}")
        
        if saved > 0:
            logger.debug(f"Saved {saved} candles to DB")


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
