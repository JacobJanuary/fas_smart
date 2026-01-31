"""
Startup warmup module for FAS Smart.

Handles:
1. Gap detection and filling from Binance REST API
2. Loading historical data from DB to memory
"""

import asyncio
import logging
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
    MAX_KLINES_PER_REQUEST = 1000
    GAP_THRESHOLD_MINUTES = 1  # Detect gaps >= 1 minute
    MAX_RESTORE_MINUTES = 10080  # 7 days = 7 * 24 * 60
    PROXY_PARALLEL_REQUESTS = 100  # Parallel requests with proxy rotation
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
            
            # Step 1: Detect and fill gaps
            await self.fill_gaps()
            
            # Step 2: Load history from DB to memory
            await self.load_history()
            
            self.warmup_complete = True
            logger.info("Warmup complete")
            return True
            
        except Exception as e:
            logger.error(f"Warmup failed: {e}", exc_info=True)
            return False
    
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
                    tasks = [
                        self._fetch_and_insert_klines_proxy(session, gap)
                        for gap in chunk
                    ]
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
        """Fetch klines via rotating proxy - new IP per request."""
        try:
            limit = min(gap['gap_minutes'], self.MAX_KLINES_PER_REQUEST)
            
            params = {
                'symbol': gap['symbol'],
                'interval': '1m',
                'limit': limit
            }
            
            # Generate unique session ID for IP rotation
            session_id = f"{gap['symbol']}{random.randint(10000, 99999)}"
            proxy_url = config.PROXY.get_rotating_url(session_id)
            
            async with session.get(
                self.BINANCE_KLINES_URL, 
                params=params,
                proxy=proxy_url,
                timeout=15
            ) as resp:
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
                    
            logger.debug(f"Filled {len(klines)} candles for {gap['symbol']} via proxy")
            
        except Exception as e:
            logger.error(f"Error filling gap for {gap['symbol']} with proxy: {e}")
    
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
            pair_data.add_candle(
                timestamp=int(candle['ts'].timestamp() * 1000),
                o=float(candle['o']),
                h=float(candle['h']),
                l=float(candle['l']),
                c=float(candle['c']),
                volume=float(candle['v']),
                buy_volume=float(candle['bv'] or candle['v'] * 0.5)
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
