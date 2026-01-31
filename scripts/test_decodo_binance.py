#!/usr/bin/env python3
"""
Test Decodo Proxy with Binance API - Gap Recovery Logic
Uses the same fetch pattern as warmup.py
"""
import asyncio
import aiohttp
import random
import time
from typing import Optional

# DECODO Configuration
DECODO_USER = "sppcmd7blj"
DECODO_PASS = "Of_3y7UoigR7syr1kR"
DECODO_HOST = "dc.decodo.com"
DECODO_PORT_RANGE = (10001, 60000)

# Binance API
BINANCE_FAPI_BASE = "https://fapi.binance.com"
KLINES_ENDPOINT = "/fapi/v1/klines"

# Test parameters
TEST_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
PARALLEL_REQUESTS = 10
RETRY_ATTEMPTS = 3


def get_proxy_url(session_id: Optional[str] = None) -> str:
    """Generate rotating proxy URL with unique session ID."""
    port = random.randint(*DECODO_PORT_RANGE)
    # Decodo uses simple user:pass format, session rotation via random ports
    return f"http://{DECODO_USER}:{DECODO_PASS}@{DECODO_HOST}:{port}"


async def fetch_klines_with_proxy(
    session: aiohttp.ClientSession,
    symbol: str,
    interval: str = "1m",
    limit: int = 100
) -> dict:
    """Fetch klines via Decodo proxy - same logic as warmup.py"""
    url = f"{BINANCE_FAPI_BASE}{KLINES_ENDPOINT}"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    
    result = {
        "symbol": symbol,
        "success": False,
        "status": None,
        "candles": 0,
        "error": None,
        "proxy_port": None,
        "latency_ms": 0
    }
    
    for attempt in range(RETRY_ATTEMPTS):
        session_id = f"{symbol}_{attempt}_{int(time.time()*1000)}"
        proxy_url = get_proxy_url(session_id)
        result["proxy_port"] = proxy_url.split(":")[-1].split("@")[0]
        
        start = time.time()
        try:
            async with session.get(
                url,
                params=params,
                proxy=proxy_url,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                result["status"] = resp.status
                result["latency_ms"] = int((time.time() - start) * 1000)
                
                if resp.status == 200:
                    data = await resp.json()
                    result["success"] = True
                    result["candles"] = len(data)
                    return result
                elif resp.status == 451:
                    # Geo-blocked - retry with different proxy
                    print(f"  ⚠️ {symbol}: Geo-blocked (451), retrying...")
                    await asyncio.sleep(0.5)
                    continue
                elif resp.status == 429:
                    # Rate limited - wait and retry
                    print(f"  ⚠️ {symbol}: Rate limited (429), waiting...")
                    await asyncio.sleep(2)
                    continue
                else:
                    result["error"] = f"HTTP {resp.status}"
                    return result
                    
        except asyncio.TimeoutError:
            result["error"] = "Timeout"
            result["latency_ms"] = 30000
        except Exception as e:
            result["error"] = str(e)[:50]
    
    return result


async def run_parallel_test():
    """Run parallel requests to simulate warmup behavior."""
    print("\n" + "="*60)
    print("🔧 DECODO Proxy Test - Binance Klines API")
    print("="*60)
    print(f"\nHost: {DECODO_HOST}")
    print(f"Port Range: {DECODO_PORT_RANGE[0]} - {DECODO_PORT_RANGE[1]}")
    print(f"Parallel Requests: {PARALLEL_REQUESTS}")
    print(f"Test Symbols: {', '.join(TEST_SYMBOLS)}")
    print("\n" + "-"*60)
    
    # Create tasks for all symbols
    tasks = []
    connector = aiohttp.TCPConnector(limit=PARALLEL_REQUESTS, force_close=True)
    
    async with aiohttp.ClientSession(connector=connector) as session:
        # Test each symbol multiple times
        for _ in range(3):  # 3 rounds
            for symbol in TEST_SYMBOLS:
                tasks.append(fetch_klines_with_proxy(session, symbol))
        
        print(f"\n📡 Sending {len(tasks)} requests...")
        start = time.time()
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start
    
    # Analyze results
    success = sum(1 for r in results if r["success"])
    failed = len(results) - success
    avg_latency = sum(r["latency_ms"] for r in results) / len(results)
    
    print("\n" + "="*60)
    print("📊 RESULTS")
    print("="*60)
    print(f"✅ Success: {success}/{len(results)}")
    print(f"❌ Failed: {failed}/{len(results)}")
    print(f"⏱️ Avg Latency: {avg_latency:.0f}ms")
    print(f"⏱️ Total Time: {total_time:.2f}s")
    print(f"📈 Throughput: {len(results)/total_time:.1f} req/s")
    
    # Show errors
    errors = [r for r in results if not r["success"]]
    if errors:
        print("\n❌ Errors:")
        for r in errors[:5]:
            print(f"   {r['symbol']}: {r['error']} (status={r['status']})")
    
    # Verdict
    print("\n" + "="*60)
    if success >= len(results) * 0.8:
        print("✅ PROXY TEST PASSED - Ready for warmup!")
    else:
        print("❌ PROXY TEST FAILED - Check configuration")
    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(run_parallel_test())
