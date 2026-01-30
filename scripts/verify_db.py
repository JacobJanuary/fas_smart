#!/usr/bin/env python3
"""
Database verification script.
Checks that schema exists and data is being saved correctly.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config import config
from db import get_cursor, test_connection


def verify_database():
    print("=" * 60)
    print("FAS Smart Database Verification")
    print(f"Host: {config.DB.HOST}:{config.DB.PORT}")
    print(f"Database: {config.DB.NAME}")
    print("=" * 60)
    
    # Test connection
    print("\n1. Testing connection...")
    if test_connection():
        print("   ✓ Connection OK")
    else:
        print("   ✗ Connection FAILED")
        return
    
    with get_cursor() as cur:
        # Check schema exists
        print("\n2. Checking schema...")
        cur.execute("""
            SELECT schema_name FROM information_schema.schemata 
            WHERE schema_name = 'fas_smart'
        """)
        if cur.fetchone():
            print("   ✓ Schema 'fas_smart' exists")
        else:
            print("   ✗ Schema 'fas_smart' NOT FOUND")
            return
        
        # Check tables
        print("\n3. Checking tables...")
        tables = ['trading_pairs', 'candles_1m', 'signals', 'signal_patterns', 'config']
        for table in tables:
            cur.execute(f"""
                SELECT COUNT(*) FROM information_schema.tables 
                WHERE table_schema = 'fas_smart' AND table_name = %s
            """, (table,))
            exists = cur.fetchone()[0] > 0
            status = "✓" if exists else "✗"
            print(f"   {status} fas_smart.{table}")
        
        # Count trading pairs
        print("\n4. Data counts...")
        cur.execute("SELECT COUNT(*) FROM fas_smart.trading_pairs WHERE is_active = true")
        active_pairs = cur.fetchone()[0]
        print(f"   Active trading pairs: {active_pairs}")
        
        cur.execute("SELECT COUNT(*) FROM fas_smart.trading_pairs")
        total_pairs = cur.fetchone()[0]
        print(f"   Total trading pairs: {total_pairs}")
        
        # Check signals
        cur.execute("SELECT COUNT(*) FROM fas_smart.signals")
        signals_count = cur.fetchone()[0]
        print(f"   Signals: {signals_count}")
        
        # Latest signals
        if signals_count > 0:
            print("\n5. Latest signals...")
            cur.execute("""
                SELECT tp.symbol, s.total_score, s.direction, s.confidence, 
                       s.entry_price, s.timestamp
                FROM fas_smart.signals s
                JOIN fas_smart.trading_pairs tp ON tp.id = s.trading_pair_id
                ORDER BY s.timestamp DESC
                LIMIT 5
            """)
            for row in cur.fetchall():
                print(f"   {row[0]}: score={row[1]:.1f}, dir={row[2]}, "
                      f"conf={row[3]:.0f}%, price={row[4]:.4f}, time={row[5]}")
        
        # Check candles
        cur.execute("SELECT COUNT(*) FROM fas_smart.candles_1m")
        candles_count = cur.fetchone()[0]
        print(f"\n6. Candles in DB: {candles_count}")
        
        if candles_count > 0:
            cur.execute("""
                SELECT tp.symbol, COUNT(*) as cnt, MAX(c.timestamp) as last_ts
                FROM fas_smart.candles_1m c
                JOIN fas_smart.trading_pairs tp ON tp.id = c.trading_pair_id
                GROUP BY tp.symbol
                ORDER BY cnt DESC
                LIMIT 5
            """)
            for row in cur.fetchall():
                print(f"   {row[0]}: {row[1]} candles, last: {row[2]}")
    
    print("\n" + "=" * 60)
    print("Verification complete!")


if __name__ == "__main__":
    verify_database()
