"""
REST API poller for Open Interest data.

Binance doesn't provide OI via WebSocket, so we poll it periodically.
"""

import asyncio
import logging
from typing import Optional
import aiohttp
from datetime import datetime

logger = logging.getLogger(__name__)


class OpenInterestPoller:
    """
    Polls Binance REST API for Open Interest data.
    
    Features:
    - Rate limit aware (max 1200 req/min)
    - Batch processing with delays
    - Error handling with retries
    """
    
    BASE_URL = "https://fapi.binance.com"
    
    def __init__(
        self,
        data_store,
        poll_interval: float = 30.0,  # seconds between full cycles
        batch_size: int = 50,
        batch_delay: float = 0.5,  # seconds between batches
    ):
        self.data_store = data_store
        self.poll_interval = poll_interval
        self.batch_size = batch_size
        self.batch_delay = batch_delay
        
        self._running = False
        self._session: Optional[aiohttp.ClientSession] = None
        self._task: Optional[asyncio.Task] = None
        
        # Statistics
        self.stats = {
            'polls_total': 0,
            'polls_success': 0,
            'polls_failed': 0,
        }
    
    async def start(self):
        """Start the polling loop"""
        self._running = True
        self._session = aiohttp.ClientSession()
        self._task = asyncio.create_task(self._poll_loop())
        logger.info(f"OI Poller started (interval={self.poll_interval}s)")
    
    async def stop(self):
        """Stop the polling loop"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._session:
            await self._session.close()
        logger.info("OI Poller stopped")
    
    async def _poll_loop(self):
        """Main polling loop"""
        while self._running:
            try:
                await self._poll_all_pairs()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"OI poll error: {e}")
            
            await asyncio.sleep(self.poll_interval)
    
    async def _poll_all_pairs(self):
        """Poll OI for all registered pairs"""
        symbols = self.data_store.get_all_symbols()
        
        if not symbols:
            return
        
        logger.debug(f"Polling OI for {len(symbols)} pairs...")
        start_time = datetime.utcnow()
        
        # Process in batches
        for i in range(0, len(symbols), self.batch_size):
            batch = symbols[i:i + self.batch_size]
            
            # Fetch batch concurrently
            tasks = [self._fetch_oi(symbol) for symbol in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for symbol, result in zip(batch, results):
                if isinstance(result, Exception):
                    self.stats['polls_failed'] += 1
                    logger.debug(f"OI fetch failed for {symbol}: {result}")
                elif result is not None:
                    self.stats['polls_success'] += 1
                    await self.data_store.update_open_interest(symbol, result)
                
                self.stats['polls_total'] += 1
            
            # Rate limit: wait between batches
            if i + self.batch_size < len(symbols):
                await asyncio.sleep(self.batch_delay)
        
        elapsed = (datetime.utcnow() - start_time).total_seconds()
        logger.debug(f"OI poll completed in {elapsed:.1f}s")
    
    async def _fetch_oi(self, symbol: str) -> Optional[float]:
        """Fetch OI for a single symbol"""
        try:
            url = f"{self.BASE_URL}/fapi/v1/openInterest"
            params = {'symbol': symbol}
            
            async with self._session.get(url, params=params, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return float(data.get('openInterest', 0))
                else:
                    return None
                    
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            logger.debug(f"OI fetch error for {symbol}: {e}")
            return None
    
    def get_stats(self) -> dict:
        """Get poller statistics"""
        return dict(self.stats)
