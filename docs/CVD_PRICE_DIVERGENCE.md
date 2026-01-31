# FAS V2 CVD_PRICE_DIVERGENCE Pattern Analysis

## 1. Detection Logic (signal_detect_patterns)

### Conditions
```sql
WHERE ABS(price_change_pct - normalized_imbalance) > v_cvd_divergence_threshold  -- 3.0
  AND SIGN(price_change_pct) != SIGN(normalized_imbalance)
  AND ABS(price_change_pct) >= v_price_change_min_cvd  -- 2.0
```

### Thresholds
- `v_cvd_divergence_threshold`: **3.0**
- `v_price_change_min_cvd`: **2.0%**

### Score
```sql
CASE
    WHEN price_change > 0 AND normalized_imbalance < 0 THEN -20.0  -- Bearish
    WHEN price_change < 0 AND normalized_imbalance > 0 THEN 20.0   -- Bullish
    ELSE 0.0
END
```

### Confidence
```sql
LEAST(60 + ABS(price_change - normalized_imbalance) * 10, 85)
```

---

## 2. Data Required
- `price_change_pct`
- `normalized_imbalance` (CVD normalized)

---

## 3. Timeframes
Все: 15m, 1h, 4h, 1d

---

## 4. Сравнение с fas_smart

### fas_smart (patterns.py)
```python
cvd_divergence_threshold: float = 5.0
price_change_min_cvd: float = 1.0
```

### ⚠️ РАСХОЖДЕНИЯ

| Аспект | FAS V2 | fas_smart |
|--------|--------|-----------|
| CVD divergence threshold | 3.0 | 5.0 ⚠️ |
| Price change min | 2.0% | 1.0% ⚠️ |
| Score | ±20.0 | ±20.0 ✅ |

---

## 5. План исправления

```python
cvd_divergence_threshold: float = 3.0   # было 5.0
price_change_min_cvd: float = 2.0       # было 1.0
```
