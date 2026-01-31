#!/usr/bin/env python3
"""Quick test for aiohttp proxy usage with Binance."""

import asyncio
import aiohttp
import random

# Read first proxy from file
with open('proxy.txt', 'r') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#'):
            proxy = line
            break

print(f"Testing proxy: {proxy}")
proxy_url = f"http://{proxy}"

async def test():
    async with aiohttp.ClientSession() as session:
        # Test ping
        print("\n1. Testing ping...")
        async with session.get(
            "https://fapi.binance.com/fapi/v1/ping",
            proxy=proxy_url,
            timeout=aiohttp.ClientTimeout(total=15)
        ) as resp:
            print(f"   Status: {resp.status}")
            print(f"   Response: {await resp.text()}")
        
        # Test klines
        print("\n2. Testing klines...")
        async with session.get(
            "https://fapi.binance.com/fapi/v1/klines",
            params={'symbol': 'BTCUSDT', 'interval': '1m', 'limit': 3},
            proxy=proxy_url,
            timeout=aiohttp.ClientTimeout(total=15)
        ) as resp:
            print(f"   Status: {resp.status}")
            if resp.status == 200:
                data = await resp.json()
                print(f"   Candles: {len(data)}")
            else:
                print(f"   Error: {await resp.text()}")

if __name__ == "__main__":
    asyncio.run(test())
