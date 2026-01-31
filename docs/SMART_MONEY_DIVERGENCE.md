# FAS V2 SMART_MONEY_DIVERGENCE Pattern Analysis

## 1. Detection Logic

### Conditions
```sql
WHERE ABS(oi_delta_pct) > 3.0                    -- Strong OI move
  AND SIGN(oi_delta_pct) != SIGN(price_change)  -- Divergence
  AND ABS(price_change) >= 1.0                   -- Significant price
```

### Score (Dynamic)
```sql
CASE
    WHEN oi_delta > 3 AND price_change < -1 THEN 40.0   -- OI up, price down = bullish
    WHEN oi_delta < -3 AND price_change > 1 THEN -40.0  -- OI down, price up = bearish
    ELSE 20.0
END
```

### Confidence
```sql
LEAST(65 + ABS(oi_delta_pct) * 3, 85)
```

---

## 2. Data Required
- `oi_delta_pct`
- `price_change_pct`

---

## 3. Timeframes
Все: 15m, 1h, 4h, 1d

---

## 4. Сравнение с fas_smart

```python
# fas_smart thresholds
oi_divergence_threshold: float = 3.0  ✅
```

Нужно проверить реализацию _detect_smart_money_divergence.
