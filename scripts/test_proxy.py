#!/usr/bin/env python3
"""
Test script for rotating proxy with Binance API.
Tests that each request comes from a different IP.
"""

import asyncio
import aiohttp

# Proxy configuration
PROXY_URL = "http://pceYcodHcK-res-any:PC_45BaupQvY0CmNxWaq@proxy-eu.proxy-cheap.com:5959"

# Test endpoints
IP_CHECK_URL = "https://ipv4.icanhazip.com"
BINANCE_PING_URL = "https://fapi.binance.com/fapi/v1/ping"
BINANCE_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"


async def test_ip_rotation():
    """Test that proxy returns different IPs."""
    print("=" * 50)
    print("Testing IP rotation (5 requests to icanhazip.com)")
    print("=" * 50)
    
    ips = []
    async with aiohttp.ClientSession() as session:
        for i in range(5):
            try:
                async with session.get(IP_CHECK_URL, proxy=PROXY_URL, timeout=10) as resp:
                    ip = (await resp.text()).strip()
                    ips.append(ip)
                    print(f"Request {i+1}: IP = {ip}")
            except Exception as e:
                print(f"Request {i+1}: ERROR - {e}")
    
    unique_ips = set(ips)
    print(f"\nUnique IPs: {len(unique_ips)} out of {len(ips)}")
    print(f"Rotation working: {'✅ YES' if len(unique_ips) > 1 else '❌ NO'}")
    return len(unique_ips) > 1


async def test_binance_ping():
    """Test Binance API ping via proxy."""
    print("\n" + "=" * 50)
    print("Testing Binance API ping")
    print("=" * 50)
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(BINANCE_PING_URL, proxy=PROXY_URL, timeout=10) as resp:
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
    print("\n" + "=" * 50)
    print("Testing Binance Klines API (BTCUSDT, 5 candles)")
    print("=" * 50)
    
    params = {
        'symbol': 'BTCUSDT',
        'interval': '1m',
        'limit': 5
    }
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(
                BINANCE_KLINES_URL, 
                params=params,
                proxy=PROXY_URL, 
                timeout=10
            ) as resp:
                status = resp.status
                data = await resp.json()
                print(f"Status: {status}")
                print(f"Candles received: {len(data)}")
                if data:
                    print(f"Latest close: {data[-1][4]}")
                print(f"Result: {'✅ SUCCESS' if status == 200 else '❌ FAILED'}")
                return status == 200
        except Exception as e:
            print(f"ERROR: {e}")
            return False


async def test_parallel_requests():
    """Test parallel requests to see rate limiting behavior."""
    print("\n" + "=" * 50)
    print("Testing parallel requests (10 concurrent)")
    print("=" * 50)
    
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'DOGEUSDT',
               'XRPUSDT', 'SOLUSDT', 'DOTUSDT', 'AVAXUSDT', 'MATICUSDT']
    
    async def fetch_klines(session, symbol):
        params = {'symbol': symbol, 'interval': '1m', 'limit': 5}
        try:
            async with session.get(
                BINANCE_KLINES_URL, 
                params=params,
                proxy=PROXY_URL, 
                timeout=15
            ) as resp:
                if resp.status == 200:
                    return (symbol, 'OK')
                else:
                    text = await resp.text()
                    return (symbol, f'ERROR {resp.status}: {text[:50]}')
        except Exception as e:
            return (symbol, f'EXCEPTION: {e}')
    
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_klines(session, s) for s in symbols]
        results = await asyncio.gather(*tasks)
        
        success = 0
        for symbol, result in results:
            status_icon = '✅' if result == 'OK' else '❌'
            print(f"  {status_icon} {symbol}: {result}")
            if result == 'OK':
                success += 1
        
        print(f"\nSuccess rate: {success}/{len(symbols)}")
        return success == len(symbols)


async def main():
    print("\n🔧 Rotating Proxy Test for FAS Smart Warmup\n")
    
    # Run tests
    ip_ok = await test_ip_rotation()
    ping_ok = await test_binance_ping()
    klines_ok = await test_binance_klines()
    parallel_ok = await test_parallel_requests()
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"  IP Rotation:     {'✅ PASS' if ip_ok else '❌ FAIL'}")
    print(f"  Binance Ping:    {'✅ PASS' if ping_ok else '❌ FAIL'}")
    print(f"  Binance Klines:  {'✅ PASS' if klines_ok else '❌ FAIL'}")
    print(f"  Parallel (10x):  {'✅ PASS' if parallel_ok else '❌ FAIL'}")
    
    if ip_ok and ping_ok and klines_ok:
        print("\n✅ Proxy is ready for use in FAS Smart warmup!")
    else:
        print("\n⚠️ Some tests failed, check proxy configuration")


if __name__ == "__main__":
    asyncio.run(main())
