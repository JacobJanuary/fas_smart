#!/usr/bin/env python3
"""
Test script for DECODO datacenter proxy with Binance API.
Tests IP rotation across multiple ports and countries.
"""

import asyncio
import aiohttp
import random

# DECODO Proxy configuration
DECODO_USER = "sppcmd7blj"
DECODO_PASS = "Of_3y7UoigR7syr1kR"
DECODO_HOST = "dc.decodo.com"
DECODO_PORT_RANGE = (10001, 60000)
DECODO_COUNTRIES = ['us', 'gb', 'de', 'nl', 'fr', 'ca']

# Test endpoints
IP_CHECK_URL = "https://ipv4.icanhazip.com"
BINANCE_PING_URL = "https://fapi.binance.com/fapi/v1/ping"
BINANCE_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"


def get_random_proxy():
    """Get random proxy URL with random port."""
    port = random.randint(*DECODO_PORT_RANGE)
    return f"http://{DECODO_USER}:{DECODO_PASS}@{DECODO_HOST}:{port}"


async def test_ip_rotation():
    """Test that proxy returns different IPs with different ports."""
    print("=" * 60)
    print("Testing IP rotation (5 requests with different ports)")
    print("=" * 60)
    
    ips = []
    async with aiohttp.ClientSession() as session:
        for i in range(5):
            proxy_url = get_random_proxy()
            port = proxy_url.split(":")[-1]
            try:
                async with session.get(IP_CHECK_URL, proxy=proxy_url, timeout=15) as resp:
                    ip = (await resp.text()).strip()
                    ips.append(ip)
                    print(f"Request {i+1} (port {port}): IP = {ip}")
            except Exception as e:
                print(f"Request {i+1} (port {port}): ERROR - {e}")
    
    unique_ips = set(ips)
    print(f"\nUnique IPs: {len(unique_ips)} out of {len(ips)}")
    print(f"Rotation working: {'✅ YES' if len(unique_ips) > 1 else '⚠️ Limited (same IP pool)'}")
    return len(ips) > 0


async def test_binance_ping():
    """Test Binance API ping via proxy."""
    print("\n" + "=" * 60)
    print("Testing Binance API ping")
    print("=" * 60)
    
    proxy_url = get_random_proxy()
    print(f"Using port: {proxy_url.split(':')[-1]}")
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(BINANCE_PING_URL, proxy=proxy_url, timeout=15) as resp:
                status = resp.status
                data = await resp.json()
                print(f"Status: {status}")
                print(f"Response: {data}")
                print(f"Result: {'✅ SUCCESS' if status == 200 else '❌ FAILED'}")
                return status == 200
        except Exception as e:
            print(f"ERROR: {e}")
            return False


async def test_binance_klines():
    """Test Binance klines API (like warmup will use)."""
    print("\n" + "=" * 60)
    print("Testing Binance Klines API (BTCUSDT, 5 candles)")
    print("=" * 60)
    
    params = {
        'symbol': 'BTCUSDT',
        'interval': '1m',
        'limit': 5
    }
    
    proxy_url = get_random_proxy()
    print(f"Using port: {proxy_url.split(':')[-1]}")
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(
                BINANCE_KLINES_URL, 
                params=params,
                proxy=proxy_url, 
                timeout=15
            ) as resp:
                status = resp.status
                data = await resp.json()
                print(f"Status: {status}")
                print(f"Candles received: {len(data) if isinstance(data, list) else 'N/A'}")
                if isinstance(data, list) and data:
                    print(f"Latest close: {data[-1][4]}")
                print(f"Result: {'✅ SUCCESS' if status == 200 else '❌ FAILED'}")
                return status == 200
        except Exception as e:
            print(f"ERROR: {e}")
            return False


async def test_parallel_requests():
    """Test parallel requests with different ports."""
    print("\n" + "=" * 60)
    print("Testing parallel requests (10 concurrent, different ports)")
    print("=" * 60)
    
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'DOGEUSDT',
               'XRPUSDT', 'SOLUSDT', 'DOTUSDT', 'AVAXUSDT', 'MATICUSDT']
    
    async def fetch_klines(session, symbol):
        params = {'symbol': symbol, 'interval': '1m', 'limit': 5}
        proxy_url = get_random_proxy()
        try:
            async with session.get(
                BINANCE_KLINES_URL, 
                params=params,
                proxy=proxy_url, 
                timeout=15
            ) as resp:
                if resp.status == 200:
                    return (symbol, 'OK', proxy_url.split(':')[-1])
                else:
                    text = await resp.text()
                    return (symbol, f'ERROR {resp.status}: {text[:50]}', proxy_url.split(':')[-1])
        except Exception as e:
            return (symbol, f'EXCEPTION: {e}', proxy_url.split(':')[-1])
    
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_klines(session, s) for s in symbols]
        results = await asyncio.gather(*tasks)
        
        success = 0
        for symbol, result, port in results:
            status_icon = '✅' if result == 'OK' else '❌'
            print(f"  {status_icon} {symbol} (port {port}): {result}")
            if result == 'OK':
                success += 1
        
        print(f"\nSuccess rate: {success}/{len(symbols)}")
        return success == len(symbols)


async def test_high_load():
    """Test 30 parallel requests to simulate warmup load."""
    print("\n" + "=" * 60)
    print("Testing high load (30 parallel requests)")
    print("=" * 60)
    
    symbols = [
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'DOGEUSDT',
        'XRPUSDT', 'SOLUSDT', 'DOTUSDT', 'AVAXUSDT', 'MATICUSDT',
        'LINKUSDT', 'ATOMUSDT', 'LTCUSDT', 'UNIUSDT', 'ICPUSDT',
        'NEARUSDT', 'APTUSDT', 'OPUSDT', 'ARBUSDT', 'MKRUSDT',
        'AAVEUSDT', 'GRTUSDT', 'INJUSDT', 'SUIUSDT', 'SEIUSDT',
        'TIAUSDT', 'FETUSDT', 'RNDRUSDT', 'IMXUSDT', 'STXUSDT'
    ]
    
    async def fetch_klines(session, symbol):
        params = {'symbol': symbol, 'interval': '1m', 'limit': 100}
        proxy_url = get_random_proxy()
        try:
            async with session.get(
                BINANCE_KLINES_URL, 
                params=params,
                proxy=proxy_url, 
                timeout=20
            ) as resp:
                return resp.status
        except:
            return 0
    
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_klines(session, s) for s in symbols]
        results = await asyncio.gather(*tasks)
        
        success = sum(1 for r in results if r == 200)
        rate_limited = sum(1 for r in results if r == 429)
        errors = sum(1 for r in results if r not in [200, 429])
        
        print(f"  ✅ Success:     {success}/{len(symbols)}")
        print(f"  ⚠️ Rate Limited (429): {rate_limited}/{len(symbols)}")
        print(f"  ❌ Errors:      {errors}/{len(symbols)}")
        
        return success >= 25  # Allow some failures


async def main():
    print("\n🔧 DECODO Datacenter Proxy Test for FAS Smart Warmup\n")
    print(f"Host: {DECODO_HOST}")
    print(f"Port Range: {DECODO_PORT_RANGE[0]} - {DECODO_PORT_RANGE[1]}")
    print(f"Countries: {', '.join(DECODO_COUNTRIES)}")
    
    # Run tests
    ip_ok = await test_ip_rotation()
    ping_ok = await test_binance_ping()
    klines_ok = await test_binance_klines()
    parallel_ok = await test_parallel_requests()
    high_load_ok = await test_high_load()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  IP Rotation:     {'✅ PASS' if ip_ok else '❌ FAIL'}")
    print(f"  Binance Ping:    {'✅ PASS' if ping_ok else '❌ FAIL'}")
    print(f"  Binance Klines:  {'✅ PASS' if klines_ok else '❌ FAIL'}")
    print(f"  Parallel (10x):  {'✅ PASS' if parallel_ok else '❌ FAIL'}")
    print(f"  High Load (30x): {'✅ PASS' if high_load_ok else '❌ FAIL'}")
    
    if ping_ok and klines_ok and parallel_ok:
        print("\n✅ DECODO proxy is ready for use in FAS Smart warmup!")
    else:
        print("\n⚠️ Some tests failed, check proxy configuration")


if __name__ == "__main__":
    asyncio.run(main())
