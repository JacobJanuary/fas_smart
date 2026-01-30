#!/usr/bin/env python3
"""
Обновление списка Binance Futures пар с гистерезисом.

Запускать каждые 4 часа через cron:
0 */4 * * * cd ~/fas_smart && ./venv/bin/python scripts/update_pairs.py

Логика:
- Добавляем: volume_24h >= $100K
- Удаляем (is_active=false): volume_7d_avg < $30K 
  AND last_signal_at > 7 days ago
  AND added_at > 14 days ago
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import requests
from datetime import datetime
from config import config
from db import get_cursor


def get_binance_futures_pairs() -> dict[str, dict]:
    """Получает все USDT-M Futures пары с Binance"""
    
    # Exchange info
    resp = requests.get("https://fapi.binance.com/fapi/v1/exchangeInfo")
    resp.raise_for_status()
    exchange_info = resp.json()
    
    symbols = {
        s['symbol']: {
            'base_asset': s['baseAsset'],
            'quote_asset': s['quoteAsset'],
        }
        for s in exchange_info['symbols']
        if s['contractType'] == 'PERPETUAL'
        and s['quoteAsset'] == 'USDT'
        and s['status'] == 'TRADING'
    }
    
    # 24h tickers
    resp = requests.get("https://fapi.binance.com/fapi/v1/ticker/24hr")
    resp.raise_for_status()
    
    for ticker in resp.json():
        symbol = ticker['symbol']
        if symbol in symbols:
            symbols[symbol]['volume_24h'] = float(ticker.get('quoteVolume', 0))
    
    return symbols


def calculate_tier(volume: float) -> str:
    """Определяет tier по объёму"""
    if volume >= config.THRESHOLDS.TIER_1_VOLUME:
        return 'TIER_1'
    elif volume >= config.THRESHOLDS.TIER_2_VOLUME:
        return 'TIER_2'
    return 'TIER_3'


def update_pairs():
    """Основная логика обновления пар"""
    
    print(f"[{datetime.now()}] Запуск обновления пар...")
    print(f"DB: {config.DB.HOST}:{config.DB.PORT}/{config.DB.NAME}")
    
    # Получаем данные с Binance
    binance_pairs = get_binance_futures_pairs()
    print(f"Получено {len(binance_pairs)} пар с Binance")
    
    with get_cursor() as cur:
        # 1. Получаем текущие пары
        cur.execute("""
            SELECT id, symbol, avg_volume_24h, volume_7d_avg, 
                   added_at, last_signal_at, is_active
            FROM fas_smart.trading_pairs
        """)
        existing = {row[1]: {
            'id': row[0],
            'avg_volume_24h': row[2],
            'volume_7d_avg': row[3],
            'added_at': row[4],
            'last_signal_at': row[5],
            'is_active': row[6],
        } for row in cur.fetchall()}
        
        added = 0
        updated = 0
        reactivated = 0
        
        # 2. Обрабатываем пары с Binance
        for symbol, data in binance_pairs.items():
            volume = data.get('volume_24h', 0)
            tier = calculate_tier(volume)
            
            if symbol in existing:
                ex = existing[symbol]
                
                # Скользящее среднее 7 дней
                old_7d = ex['volume_7d_avg'] or volume
                new_7d = (old_7d * 6 + volume) / 7
                
                cur.execute("""
                    UPDATE fas_smart.trading_pairs
                    SET avg_volume_24h = %s,
                        volume_7d_avg = %s,
                        tier = %s::fas_smart.liquidity_tier,
                        updated_at = NOW()
                    WHERE symbol = %s
                """, (volume, new_7d, tier, symbol))
                updated += 1
                
                # Реактивация
                if not ex['is_active'] and volume >= config.THRESHOLDS.ENTRY_VOLUME:
                    cur.execute("""
                        UPDATE fas_smart.trading_pairs
                        SET is_active = true,
                            deactivated_at = NULL,
                            deactivation_reason = NULL
                        WHERE symbol = %s
                    """, (symbol,))
                    reactivated += 1
                    print(f"  Реактивирована: {symbol} (${volume:,.0f})")
            
            else:
                # Новая пара
                if volume >= config.THRESHOLDS.ENTRY_VOLUME:
                    cur.execute("""
                        INSERT INTO fas_smart.trading_pairs 
                        (symbol, base_asset, quote_asset, tier, avg_volume_24h, 
                         volume_7d_avg, is_active, added_at)
                        VALUES (%s, %s, %s, %s::fas_smart.liquidity_tier, %s, %s, true, NOW())
                        ON CONFLICT (symbol) DO NOTHING
                    """, (symbol, data['base_asset'], data['quote_asset'], 
                          tier, volume, volume))
                    added += 1
                    print(f"  Добавлена: {symbol} (${volume:,.0f}, {tier})")
        
        # 3. Деактивация
        cur.execute("""
            UPDATE fas_smart.trading_pairs
            SET is_active = false,
                deactivated_at = NOW(),
                deactivation_reason = 'low_volume'
            WHERE is_active = true
              AND volume_7d_avg < %s
              AND added_at < NOW() - INTERVAL '%s days'
              AND (last_signal_at IS NULL OR last_signal_at < NOW() - INTERVAL '%s days')
            RETURNING symbol, volume_7d_avg
        """, (config.THRESHOLDS.EXIT_VOLUME, 
              config.THRESHOLDS.MIN_DAYS_BEFORE_REMOVE, 
              config.THRESHOLDS.SIGNAL_GRACE_DAYS))
        
        deactivated = 0
        for row in cur.fetchall():
            deactivated += 1
            print(f"  Деактивирована: {row[0]} (avg ${row[1]:,.0f})")
        
        # Статистика
        cur.execute("""
            SELECT 
                COUNT(*) FILTER (WHERE is_active) as active,
                COUNT(*) FILTER (WHERE NOT is_active) as inactive,
                COUNT(*) FILTER (WHERE tier = 'TIER_1' AND is_active) as tier1,
                COUNT(*) FILTER (WHERE tier = 'TIER_2' AND is_active) as tier2,
                COUNT(*) FILTER (WHERE tier = 'TIER_3' AND is_active) as tier3
            FROM fas_smart.trading_pairs
        """)
        stats = cur.fetchone()
        
        print(f"\nИтого:")
        print(f"  Добавлено: {added}")
        print(f"  Обновлено: {updated}")
        print(f"  Реактивировано: {reactivated}")
        print(f"  Деактивировано: {deactivated}")
        print(f"\nТекущее состояние:")
        print(f"  Активных: {stats[0]}")
        print(f"  Неактивных: {stats[1]}")
        print(f"  TIER_1: {stats[2]}, TIER_2: {stats[3]}, TIER_3: {stats[4]}")


if __name__ == "__main__":
    update_pairs()
