# FAS V2 ATR Analysis Report

## 1. Расчёт (calculate_atr_batch_v2)

### True Range формула
```sql
true_range = GREATEST(
    high_price - low_price,
    ABS(high_price - LAG(close_price)),
    ABS(low_price - LAG(close_price))
)
```

### ATR формула
```sql
ATR = AVG(true_range)  -- over p_period candles
```

**SMA-based** (НЕ EMA)

### Данные
- `market_data_aggregated`: high_price, low_price, close_price
- Period: configurable (default 14)

---

## 2. Использование в Indicator Score

**❌ НЕ ИСПОЛЬЗУЕТСЯ в indicator_score!**

ATR рассчитывается, но не участвует в scoring. Используется только в паттернах.

---

## 3. Таймфреймы

Все 4: 15m, 1h, 4h, 1d

---

## 4. Сравнение с fas_smart

### fas_smart (indicators.py)
```python
def calculate_atr(highs, lows, closes, period=14):
    tr = np.maximum(
        highs - lows,
        np.maximum(
            np.abs(highs - np.roll(closes, 1)),
            np.abs(lows - np.roll(closes, 1))
        )
    )
    return np.mean(tr[-period:])  # SMA
```

### ⚠️ MINOR РАСХОЖДЕНИЕ

| Аспект | FAS V2 | fas_smart |
|--------|--------|-----------|
| True Range | GREATEST(3 values) | np.maximum ✅ |
| ATR | SMA (AVG) | SMA (np.mean) ✅ |
| Usage in score | ❌ | ❌ ✅ |
| Period | configurable | 14 ✅ |

**✅ Алгоритм соответствует.**
