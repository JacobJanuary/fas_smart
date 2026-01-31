# FAS V2 ACCUMULATION Pattern Analysis

## 1. Detection Logic

### Conditions (Optimized Phase 1-2)
```sql
WHERE buy_volume > sell_volume * 2.0      -- buy/sell ratio > 2x
  AND price_change BETWEEN -0.8 AND 0.8  -- sideways price
  AND oi_delta > 0.3                      -- OI increasing
  AND normalized_imbalance > 0.35         -- CVD positive
```

### Score
```sql
30.0  -- Fixed bullish
```

### Confidence
```sql
LEAST(70 + (buy_volume/sell_volume - 1) * 20, 85)
```

---

## 2. Data Required
- `buy_volume`, `sell_volume`
- `price_change_pct`
- `oi_delta_pct`
- `normalized_imbalance`

---

## 3. Timeframes
Все: 15m, 1h, 4h, 1d

---

## 4. Сравнение с fas_smart

### fas_smart thresholds
```python
sideways_price_range: float = 1.0   # FAS V2: 0.8
accum_cvd_threshold: float = 0.1    # FAS V2: 0.35
accum_volume_threshold: float = 1.5 # FAS V2: 2.0
```

### ⚠️ РАСХОЖДЕНИЯ

| Параметр | FAS V2 | fas_smart |
|----------|--------|-----------|
| Volume ratio | 2.0 | 1.5 ⚠️ |
| Price range | ±0.8% | ±1.0% ⚠️ |
| CVD threshold | 0.35 | 0.1 ⚠️ |
| OI delta min | 0.3 | ??? |

---

## 5. План исправления

```python
sideways_price_range: float = 0.8
accum_cvd_threshold: float = 0.35
accum_volume_threshold: float = 2.0
```
