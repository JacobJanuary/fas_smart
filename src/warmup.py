"""
Startup warmup module for FAS Smart.

Handles:
1. Gap detection and filling from Binance REST API
2. Loading historical data from DB to memory
"""

import asyncio
import logging
import os
import random
import ipaddress
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import aiohttp

from db import get_cursor
from ws.storage import DataStore, PairData
from config import config

logger = logging.getLogger(__name__)


def generate_ipv6_address() -> Optional[str]:
    """
    Generate a random IPv6 address from configured prefix.
    Returns None if IPv6 is not configured.
    """
    if not config.IPV6.ENABLED or not config.IPV6.PREFIX:
        return None
    
    try:
        # Parse the prefix
        network = ipaddress.IPv6Network(
            f"{config.IPV6.PREFIX}/{config.IPV6.PREFIX_LENGTH}", 
            strict=False
        )
        
        # Generate random suffix
        suffix_bits = 128 - config.IPV6.PREFIX_LENGTH
        random_suffix = random.getrandbits(suffix_bits)
        
        # Combine prefix and suffix
        address = network.network_address + random_suffix
        return str(address)
        
    except Exception as e:
        logger.warning(f"Failed to generate IPv6 address: {e}")
        return None


class WarmupManager:
    """
    Manages startup warmup process.
    
    Runs in background after service starts to:
    1. Fill gaps in candles_1m table from Binance API
    2. Load historical candles from DB into PairData memory
    """
    
    BINANCE_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"
    BINANCE_TICKERS_URL = "https://fapi.binance.com/fapi/v1/ticker/24hr"
    MAX_KLINES_PER_REQUEST = 1000
    GAP_THRESHOLD_MINUTES = 1  # Detect gaps >= 1 minute
    MAX_RESTORE_HOURS = int(os.getenv('RESTORE_HOURS', '168'))  # Default 7 days (168 hours)
    MAX_RESTORE_MINUTES = MAX_RESTORE_HOURS * 60
    PROXY_PARALLEL_REQUESTS = int(os.getenv('PROXY_PARALLEL', '30'))
    PROXY_RETRY_ATTEMPTS = 5  # More attempts for rate limit recovery
    WARMUP_CANDLES = 100  # Load last 100 candles for indicators
    
    def __init__(self, data_store: DataStore):
        self.data_store = data_store
        self.warmup_complete = False
        
    async def run_warmup(self) -> bool:
        """
        Main warmup sequence.
        
        Returns:
            True if warmup successful
        """
        try:
            logger.info("Starting warmup sequence...")
            
            # Step 0: Fetch 24h tickers to determine liquidity tiers
            await self.update_tiers()
            
            # Step 1: Detect and fill gaps for 1m candles
            await self.fill_gaps()
            
            # Step 2: Fill higher timeframe candles (1h, 4h, 1d)
            await self.fill_htf_gaps()
            
            # Step 3: Load history from DB to memory
            await self.load_history()
            
            self.warmup_complete = True
            logger.info("Warmup complete")
            return True
            
        except Exception as e:
            logger.error(f"Warmup failed: {e}", exc_info=True)
            return False
    
    async def warmup_symbols(self, symbols: list[str]):
        """Run warmup for specific symbols (for newly added pairs)."""
        logger.info(f"Running targeted warmup for {len(symbols)} symbols...")
        
        try:
            # Update tiers for new symbols
            await self.update_tiers()
            
            # Fill gaps for these specific symbols
            gaps = await self._detect_gaps_for_symbols(symbols)
            if gaps:
                logger.info(f"Found {len(gaps)} gaps to fill for new symbols")
                await self._fill_gaps_batch(gaps)
            
            # Load history for new symbols
            await self._load_history_for_symbols(symbols)
            
            logger.info(f"Completed warmup for {len(symbols)} new symbols")
        except Exception as e:
            logger.warning(f"Warmup for new symbols failed: {e}")
    
    async def _detect_gaps_for_symbols(self, symbols: list[str]) -> list[dict]:
        """Detect gaps for specific symbols."""
        gaps = []
        with get_cursor() as cur:
            for symbol in symbols:
                cur.execute("""
                    SELECT 
                        tp.id as pair_id,
                        tp.symbol,
                        c.last_ts,
                        EXTRACT(EPOCH FROM (NOW() - COALESCE(c.last_ts, NOW() - INTERVAL '500 minutes'))) / 60 as gap_minutes
                    FROM fas_smart.trading_pairs tp
                    LEFT JOIN LATERAL (
                        SELECT MAX(ts) as last_ts 
                        FROM fas_smart.candles_1m 
                        WHERE pair_id = tp.id
                    ) c ON true
                    WHERE tp.symbol = %s AND tp.is_active = true
                """, (symbol,))
                row = cur.fetchone()
                if row and row[3] > 1:
                    gaps.append({
                        'pair_id': row[0],
                        'symbol': row[1],
                        'last_ts': row[2],
                        'gap_minutes': int(row[3])
                    })
        return gaps
    
    async def _load_history_for_symbols(self, symbols: list[str]):
        """Load history from DB to memory for specific symbols."""
        logger.info(f"Loading history for {len(symbols)} symbols...")
        
        with get_cursor() as cur:
            for symbol in symbols:
                pair_data = self.data_store.get_pair(symbol)
                if not pair_data:
                    continue
                
                cur.execute("""
                    SELECT ts, o, h, l, c, v, bv
                    FROM fas_smart.candles_1m
                    WHERE pair_id = %s
                    ORDER BY ts DESC
                    LIMIT %s
                """, (pair_data.pair_id, self.WARMUP_CANDLES)) # Changed from MIN_CANDLES_REQUIRED to WARMUP_CANDLES
                
                rows = cur.fetchall()
                for row in reversed(rows):
                    pair_data.add_candle(
                        timestamp=row[0],
                        o=float(row[1]),
                        h=float(row[2]),
                        l=float(row[3]),
                        c=float(row[4]),
                        volume=float(row[5]),
                        buy_volume=float(row[6]) if row[6] else float(row[5]) * 0.5
                    )
        
        logger.info(f"Loaded history for {len(symbols)} symbols")
    
    async def fill_gaps(self):
        """
        Detect last gap in candles_1m and fill via Binance REST API.
        """
        logger.info("Checking for gaps in candles_1m...")
        
        # Get last candle timestamp per pair (check within restore window)
        with get_cursor(dict_cursor=True) as cur:
            cur.execute(f"""
                SELECT tp.id as pair_id, tp.symbol, 
                       MAX(c.ts) as last_ts
                FROM fas_smart.trading_pairs tp
                LEFT JOIN fas_smart.candles_1m c ON tp.id = c.pair_id 
                    AND c.ts > NOW() - INTERVAL '{self.MAX_RESTORE_MINUTES} minutes'
                GROUP BY tp.id, tp.symbol
                ORDER BY tp.symbol
            """)
            pairs_status = cur.fetchall()
        
        if not pairs_status:
            logger.warning("No trading pairs found")
            return
            
        now = datetime.utcnow()
        gaps_to_fill = []
        
        for row in pairs_status:
            if row['last_ts'] is None:
                # No recent data at all - full restore up to MAX_RESTORE_MINUTES
                gap_minutes = self.MAX_RESTORE_MINUTES
            else:
                gap_minutes = (now - row['last_ts'].replace(tzinfo=None)).total_seconds() / 60
            
            if gap_minutes > self.GAP_THRESHOLD_MINUTES:
                # Skip non-ASCII symbols (Chinese characters break proxy URL encoding)
                if not row['symbol'].isascii():
                    logger.debug(f"Skipping non-ASCII symbol: {row['symbol']}")
                    continue
                    
                # Cap at MAX_RESTORE_MINUTES
                gap_minutes = min(gap_minutes, self.MAX_RESTORE_MINUTES)
                gaps_to_fill.append({
                    'pair_id': row['pair_id'],
                    'symbol': row['symbol'],
                    'gap_minutes': int(gap_minutes) + 5,  # Extra buffer
                    'last_ts': row['last_ts']
                })
        
        if not gaps_to_fill:
            logger.info("No gaps detected")
            return
            
        logger.info(f"Found {len(gaps_to_fill)} pairs with gaps, filling...")
        
        # Fill gaps via Binance API with rotation
        if config.PROXY.ENABLED:
            logger.info(f"Proxy rotation enabled, using {self.PROXY_PARALLEL_REQUESTS} parallel requests")
            # With proxy rotation - parallel requests with unique session IDs
            async with aiohttp.ClientSession() as session:
                for i in range(0, len(gaps_to_fill), self.PROXY_PARALLEL_REQUESTS):
                    chunk = gaps_to_fill[i:i + self.PROXY_PARALLEL_REQUESTS]
                    
                    # Wrap each task with timeout
                    async def fetch_with_timeout(gap):
                        try:
                            return await asyncio.wait_for(
                                self._fetch_and_insert_klines_proxy(session, gap),
                                timeout=60.0  # 60s max per symbol
                            )
                        except asyncio.TimeoutError:
                            logger.error(f"⏰ TIMEOUT: {gap['symbol']} took >60s, skipping")
                            return None
                        except Exception as e:
                            logger.error(f"❌ ERROR: {gap['symbol']}: {e}")
                            return None
                    
                    tasks = [fetch_with_timeout(gap) for gap in chunk]
                    await asyncio.gather(*tasks, return_exceptions=True)
                    logger.info(f"Processed {min(i + self.PROXY_PARALLEL_REQUESTS, len(gaps_to_fill))}/{len(gaps_to_fill)} pairs")
        elif config.IPV6.ENABLED:
            logger.info("IPv6 rotation enabled, using random addresses per request")
            for gap in gaps_to_fill:
                await self._fetch_and_insert_klines_ipv6(gap)
        else:
            # No rotation - use conservative rate limiting
            async with aiohttp.ClientSession() as session:
                chunk_size = 20  # Smaller chunks without rotation
                for i in range(0, len(gaps_to_fill), chunk_size):
                    chunk = gaps_to_fill[i:i + chunk_size]
                    tasks = [
                        self._fetch_and_insert_klines(session, gap)
                        for gap in chunk
                    ]
                    await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # Rate limit pause (safe: 20 req/2sec = 600 req/min)
                    if i + chunk_size < len(gaps_to_fill):
                        await asyncio.sleep(2)
        
        logger.info(f"Gap filling complete")
    
    async def fill_htf_gaps(self):
        """
        Fill higher timeframe candles (1h, 4h, 1d) directly into memory.
        Fetches 24 candles per timeframe for each pair.
        """
        logger.info("Loading HTF candles (1h, 4h, 1d)...")
        
        symbols = self.data_store.get_all_symbols()
        # Filter non-ASCII symbols
        symbols = [s for s in symbols if s.isascii()]
        
        htf_intervals = {
            '1h': 50,   # 50 candles for MACD(35) + padding
            '4h': 50,   # 50 candles for MACD(35) + padding
            '1d': 50,   # 50 candles for MACD(35) + padding
        }
        
        if not config.PROXY.ENABLED and not config.IPV6.ENABLED:
            logger.warning("HTF warmup requires proxy or IPv6 - skipping")
            return
        
        total_requests = len(symbols) * len(htf_intervals)
        completed = 0
        
        async with aiohttp.ClientSession() as session:
            for interval, limit in htf_intervals.items():
                logger.info(f"Fetching {interval} candles for {len(symbols)} pairs...")
                
                for i in range(0, len(symbols), self.PROXY_PARALLEL_REQUESTS):
                    chunk = symbols[i:i + self.PROXY_PARALLEL_REQUESTS]
                    tasks = [
                        self._fetch_htf_candles(session, symbol, interval, limit)
                        for symbol in chunk
                    ]
                    await asyncio.gather(*tasks, return_exceptions=True)
                    completed += len(chunk)
                    
                logger.info(f"Loaded {interval} candles for all pairs")
        
        logger.info(f"HTF candle loading complete")
    
    async def _fetch_htf_candles(
        self,
        session: aiohttp.ClientSession,
        symbol: str,
        interval: str,
        limit: int
    ):
        """Fetch HTF candles from Binance and store in memory."""
        url = f"{self.BINANCE_KLINES_URL}?symbol={symbol}&interval={interval}&limit={limit}"
        
        for attempt in range(self.PROXY_RETRY_ATTEMPTS):
            try:
                proxy_url = config.PROXY.get_rotating_url()
                async with session.get(url, proxy=proxy_url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status == 451:
                        await asyncio.sleep(0.5)
                        continue
                    if resp.status == 429:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    if resp.status == 200:
                        config.PROXY.report_decodo_success()
                        klines = await resp.json()
                        pair_data = self.data_store.get_pair(symbol)
                        if pair_data and klines:
                            for k in klines:
                                pair_data.add_htf_candle(
                                    timeframe=interval,
                                    timestamp=k[6],  # Close time
                                    o=float(k[1]),
                                    h=float(k[2]),
                                    l=float(k[3]),
                                    c=float(k[4]),
                                    volume=float(k[5]),
                                    buy_volume=float(k[9]),  # Taker buy volume
                                )
                            logger.debug(f"Loaded {len(klines)} {interval} candles for {symbol}")
                        return
            except Exception as e:
                config.PROXY.report_decodo_failure()
                if attempt == self.PROXY_RETRY_ATTEMPTS - 1:
                    logger.warning(f"Failed to fetch {interval} candles for {symbol}: {e}")
                await asyncio.sleep(1)

    
    async def _fetch_and_insert_klines(
        self, 
        session: aiohttp.ClientSession, 
        gap: dict
    ):
        """Fetch klines from Binance and insert to DB."""
        try:
            limit = min(gap['gap_minutes'], self.MAX_KLINES_PER_REQUEST)
            
            params = {
                'symbol': gap['symbol'],
                'interval': '1m',
                'limit': limit
            }
            
            async with session.get(self.BINANCE_KLINES_URL, params=params) as resp:
                if resp.status != 200:
                    logger.warning(f"Failed to fetch {gap['symbol']}: {resp.status}")
                    return
                    
                klines = await resp.json()
            
            if not klines:
                return
                
            # Insert to DB
            with get_cursor() as cur:
                for k in klines:
                    # Skip if already exists or before last_ts
                    ts = datetime.utcfromtimestamp(k[0] / 1000)
                    if gap['last_ts'] and ts <= gap['last_ts'].replace(tzinfo=None):
                        continue
                        
                    # Calculate buy_volume estimate (simplified)
                    volume = float(k[5])  # Quote volume
                    taker_buy = float(k[9]) if len(k) > 9 else volume * 0.5
                    
                    cur.execute("""
                        INSERT INTO fas_smart.candles_1m 
                        (pair_id, ts, o, h, l, c, v, bv)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (pair_id, ts) DO NOTHING
                    """, (
                        gap['pair_id'],
                        ts,
                        float(k[1]),  # open
                        float(k[2]),  # high
                        float(k[3]),  # low
                        float(k[4]),  # close
                        volume,
                        taker_buy
                    ))
                    
            logger.debug(f"Filled {len(klines)} candles for {gap['symbol']}")
            
        except Exception as e:
            logger.error(f"Error filling gap for {gap['symbol']}: {e}")
    
    async def _fetch_and_insert_klines_proxy(
        self, 
        session: aiohttp.ClientSession, 
        gap: dict
    ):
        """Fetch klines via rotating proxy - multiple batches for large gaps."""
        total_minutes = min(gap['gap_minutes'], self.MAX_RESTORE_MINUTES)
        now_ms = int(datetime.utcnow().timestamp() * 1000)
        
        # Calculate target start time
        target_start_ms = now_ms - (total_minutes * 60 * 1000)
        current_end_ms = now_ms
        total_inserted = 0
        batch_num = 0
        
        while current_end_ms > target_start_ms:
            batch_num += 1
            success = False
            
            for attempt in range(self.PROXY_RETRY_ATTEMPTS):
                try:
                    params = {
                        'symbol': gap['symbol'],
                        'interval': '1m',
                        'endTime': current_end_ms,
                        'limit': 1000  # Max per request
                    }
                    
                    # Generate unique session ID for IP rotation
                    session_id = f"{gap['symbol']}{batch_num}{random.randint(10000, 99999)}"
                    proxy_url = config.PROXY.get_rotating_url(session_id)
                    
                    async with session.get(
                        self.BINANCE_KLINES_URL, 
                        params=params,
                        proxy=proxy_url,
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as resp:
                        if resp.status == 429:
                            # Rate limited - exponential backoff
                            wait_time = 2 ** attempt
                            logger.debug(f"Rate limited on {gap['symbol']}, waiting {wait_time}s...")
                            await asyncio.sleep(wait_time)
                            continue
                        if resp.status == 451:
                            # Geo-blocked proxy - retry with different one
                            logger.debug(f"Proxy blocked (451) for {gap['symbol']}, trying another...")
                            await asyncio.sleep(0.5)
                            continue
                        if resp.status == 502 or resp.status == 503:
                            await asyncio.sleep(1 * (attempt + 1))
                            continue
                        if resp.status != 200:
                            logger.warning(f"Failed to fetch {gap['symbol']} batch {batch_num}: {resp.status}")
                            return
                            
                        klines = await resp.json()
                        config.PROXY.report_decodo_success()
                    
                    if not klines:
                        success = True
                        break  # No more data
                    
                    # Insert to DB
                    inserted = 0
                    with get_cursor() as cur:
                        for k in klines:
                            ts = datetime.utcfromtimestamp(k[0] / 1000)
                            if gap['last_ts'] and ts <= gap['last_ts'].replace(tzinfo=None):
                                continue
                                
                            volume = float(k[5])
                            taker_buy = float(k[9]) if len(k) > 9 else volume * 0.5
                            
                            cur.execute("""
                                INSERT INTO fas_smart.candles_1m 
                                (pair_id, ts, o, h, l, c, v, bv)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                                ON CONFLICT (pair_id, ts) DO NOTHING
                            """, (
                                gap['pair_id'],
                                ts,
                                float(k[1]),
                                float(k[2]),
                                float(k[3]),
                                float(k[4]),
                                volume,
                                taker_buy
                            ))
                            inserted += 1
                    
                    total_inserted += inserted
                    
                    # Move endTime to before the earliest candle in this batch
                    earliest_open_time = klines[0][0]  # First element is openTime
                    current_end_ms = earliest_open_time - 1
                    
                    success = True
                    break
                    
                except Exception as e:
                    config.PROXY.report_decodo_failure()
                    if attempt < self.PROXY_RETRY_ATTEMPTS - 1:
                        await asyncio.sleep(1 * (attempt + 1))
                    else:
                        logger.error(f"Error fetching {gap['symbol']} batch {batch_num}: {e}")
                        return
            
            if not success:
                break
        
        if total_inserted > 0:
            logger.info(f"✅ {gap['symbol']}: filled {total_inserted} candles in {batch_num} batches")
        else:
            logger.warning(f"⚠️ {gap['symbol']}: no candles inserted (gap: {gap['gap_minutes']} min)")
    
    
    
    async def _fetch_and_insert_klines_ipv6(self, gap: dict):
        """Fetch klines with IPv6 rotation - new address per request."""
        try:
            limit = min(gap['gap_minutes'], self.MAX_KLINES_PER_REQUEST)
            
            params = {
                'symbol': gap['symbol'],
                'interval': '1m',
                'limit': limit
            }
            
            # Generate random IPv6 address for this request
            ipv6_addr = generate_ipv6_address()
            
            # Create connector with specific source address
            connector = aiohttp.TCPConnector(
                local_addr=(ipv6_addr, 0) if ipv6_addr else None,
                family=0  # Allow both IPv4 and IPv6
            )
            
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.get(self.BINANCE_KLINES_URL, params=params) as resp:
                    if resp.status != 200:
                        logger.warning(f"Failed to fetch {gap['symbol']}: {resp.status}")
                        return
                        
                    klines = await resp.json()
            
            if not klines:
                return
                
            # Insert to DB
            with get_cursor() as cur:
                for k in klines:
                    ts = datetime.utcfromtimestamp(k[0] / 1000)
                    if gap['last_ts'] and ts <= gap['last_ts'].replace(tzinfo=None):
                        continue
                        
                    volume = float(k[5])
                    taker_buy = float(k[9]) if len(k) > 9 else volume * 0.5
                    
                    cur.execute("""
                        INSERT INTO fas_smart.candles_1m 
                        (pair_id, ts, o, h, l, c, v, bv)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (pair_id, ts) DO NOTHING
                    """, (
                        gap['pair_id'],
                        ts,
                        float(k[1]),
                        float(k[2]),
                        float(k[3]),
                        float(k[4]),
                        volume,
                        taker_buy
                    ))
                    
            logger.debug(f"Filled {len(klines)} candles for {gap['symbol']} via {ipv6_addr or 'default'}")
            
        except Exception as e:
            logger.error(f"Error filling gap for {gap['symbol']} with IPv6: {e}")
    
    async def load_history(self):
        """
        Load historical candles from DB to PairData memory.
        Restores CVD, smoothed_imbalance, and other state.
        """
        logger.info("Loading history from DB to memory...")
        
        # Load last N candles per pair
        # Note: Schema uses short column names: o, h, l, c, v, bv, fr
        with get_cursor(dict_cursor=True) as cur:
            cur.execute(f"""
                WITH ranked AS (
                    SELECT 
                        c.pair_id, c.ts, c.o, c.h, c.l, c.c,
                        c.v, c.bv, c.fr,
                        ROW_NUMBER() OVER (
                            PARTITION BY c.pair_id 
                            ORDER BY c.ts DESC
                        ) as rn
                    FROM fas_smart.candles_1m c
                    WHERE c.ts > NOW() - INTERVAL '3 hours'
                )
                SELECT * FROM ranked 
                WHERE rn <= {self.WARMUP_CANDLES}
                ORDER BY pair_id, ts ASC
            """)
            candles = cur.fetchall()
        
        if not candles:
            logger.warning("No historical candles found in DB")
            return
            
        # Group by pair_id and load into PairData
        current_pair_id = None
        pair_data = None
        loaded_pairs = 0
        loaded_candles = 0
        
        for candle in candles:
            if candle['pair_id'] != current_pair_id:
                current_pair_id = candle['pair_id']
                pair_data = self.data_store.get_pair_by_id(current_pair_id)
                if pair_data:
                    loaded_pairs += 1
                    
            if not pair_data:
                continue
                
            # Add candle to memory (using short column names from schema)
            bv = candle['bv']
            v = float(candle['v'])
            pair_data.add_candle(
                timestamp=int(candle['ts'].timestamp() * 1000),
                o=float(candle['o']),
                h=float(candle['h']),
                l=float(candle['l']),
                c=float(candle['c']),
                volume=v,
                buy_volume=float(bv) if bv else v * 0.5
            )
            
            # Restore funding rate if available
            if candle['fr']:
                pair_data.latest_funding_rate = float(candle['fr'])
                
            loaded_candles += 1
        
        logger.info(f"Loaded {loaded_candles} candles for {loaded_pairs} pairs")
        
        # Restore CVD cumulative from signals table (if available)
        await self._restore_cvd()
    
    async def _restore_cvd(self):
        """Restore CVD cumulative from last saved signal."""
        try:
            with get_cursor(dict_cursor=True) as cur:
                cur.execute("""
                    SELECT DISTINCT ON (pair_id) 
                        pair_id, 
                        indicators_json->>'cvd_cumulative' as cvd
                    FROM fas_smart.signals
                    WHERE ts > NOW() - INTERVAL '1 hour'
                    ORDER BY pair_id, ts DESC
                """)
                last_cvds = cur.fetchall()
            
            for row in last_cvds:
                if row['cvd']:
                    pair_data = self.data_store.get_pair_by_id(row['pair_id'])
                    if pair_data:
                        try:
                            pair_data.cvd_cumulative = float(row['cvd'])
                        except (ValueError, TypeError):
                            pass
                            
            logger.info(f"Restored CVD for {len(last_cvds)} pairs")
            
        except Exception as e:
            logger.warning(f"Could not restore CVD: {e}")
    
    async def update_tiers(self):
        """
        Fetch 24h tickers from Binance and update liquidity tier per pair.
        
        Uses quoteVolume (24h volume in USD) to determine tier:
        - TIER_1: >= $100M
        - TIER_2: >= $10M  
        - TIER_3: < $10M
        """
        from config import ThresholdConfig
        
        logger.info("Fetching 24h tickers for tier classification...")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.BINANCE_TICKERS_URL) as resp:
                    if resp.status != 200:
                        logger.warning(f"Failed to fetch tickers: {resp.status}")
                        return
                    
                    tickers = await resp.json()
            
            # Map symbol -> quoteVolume
            volume_map = {}
            for t in tickers:
                symbol = t.get('symbol', '')
                if symbol.endswith('USDT'):
                    try:
                        volume_map[symbol] = float(t.get('quoteVolume', 0))
                    except (ValueError, TypeError):
                        pass
            
            # Update tier for each pair in data store
            updated = 0
            for symbol, pair_data in self.data_store.pairs.items():
                vol_24h = volume_map.get(symbol, 0)
                pair_data.tier = ThresholdConfig.get_tier(vol_24h)
                updated += 1
            
            # Count tiers
            tier_counts = {'TIER_1': 0, 'TIER_2': 0, 'TIER_3': 0}
            for pair_data in self.data_store.pairs.values():
                tier_counts[pair_data.tier] = tier_counts.get(pair_data.tier, 0) + 1
            
            logger.info(f"Updated tiers for {updated} pairs: {tier_counts}")
            
        except Exception as e:
            logger.warning(f"Failed to update tiers: {e}")

