# FAS V2 DISTRIBUTION Pattern Analysis

## 1. Detection Logic

### Conditions (Optimized Phase 1-2)
```sql
WHERE sell_volume > buy_volume * 2.0      -- sell/buy ratio > 2x
  AND price_change BETWEEN -0.8 AND 0.8  -- sideways price
  AND oi_delta < -0.3                     -- OI decreasing
  AND normalized_imbalance < -0.35        -- CVD negative
```

### Score
```sql
-30.0  -- Fixed bearish
```

### Confidence
```sql
LEAST(70 + (sell_volume/buy_volume - 1) * 20, 85)
```

---

## 2. Data Required
Зеркально ACCUMULATION

---

## 3. Timeframes
Все: 15m, 1h, 4h, 1d

---

## 4. Соответствие fas_smart

**✅ Thresholds уже исправлены** вместе с ACCUMULATION:
- `accum_volume_threshold: 2.0` (используется для обоих)
- `sideways_price_range: 0.8`
- `accum_cvd_threshold: 0.35`

DISTRIBUTION и ACCUMULATION используют одни и те же thresholds с противоположными знаками.
