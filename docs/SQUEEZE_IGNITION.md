# FAS V2 SQUEEZE_IGNITION Pattern Analysis

## 1. Detection Logic

### Conditions
```sql
WHERE price_change > 2.0           -- Strong price move
  AND volume_zscore > 3.0          -- High volume spike
  AND funding_rate < -0.0003       -- Negative funding (shorts)
  AND oi_delta > 2.0               -- OI increasing
```

### Score
```sql
50.0  -- Fixed bullish (short squeeze confirmation)
```

### Confidence
```sql
LEAST(70 + ABS(funding_rate * 10000), 90)
```

---

## 2. Data Required
- `price_change_pct`
- `volume_zscore`
- `funding_rate_avg`
- `oi_delta_pct`

---

## 3. Timeframes
Все: 15m, 1h, 4h, 1d

---

## 4. Сравнение с fas_smart

### fas_smart
```python
def _detect_squeeze_ignition(self, indicators, pair_data):
    # price > 2%, volume_zscore > 3, funding < -0.0003, oi_delta > 2%
```

Нужно проверить текущую реализацию.
