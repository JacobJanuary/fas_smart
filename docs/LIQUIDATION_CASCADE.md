# FAS V2 LIQUIDATION_CASCADE Pattern Analysis

## 1. Detection Logic

### Condition
```sql
(liquidations_long + liquidations_short) > volume * v_liquidation_threshold
```

### Thresholds (Tier-Based)
| Tier | Threshold |
|------|-----------|
| TIER_1 | 0.02 (2%) |
| TIER_2 | 0.025 (2.5%) |
| TIER_3 | 0.03 (3%) |

### Score (Multi-Level)
```sql
CASE
    WHEN total_liq > volume * 0.1 THEN 80.0   -- >10%
    WHEN total_liq > volume * 0.05 THEN 60.0  -- >5%
    ELSE 40.0
END
```

### Confidence
```sql
LEAST(70 + (liq_ratio * 100), 95)
```

---

## 2. Data Required
- `liquidations_long`, `liquidations_short`
- `volume`

---

## 3. Timeframes
Все: 15m, 1h, 4h, 1d

---

## 4. Сравнение с fas_smart

### fas_smart
```python
liq_ratio_threshold: float = 0.025  # 2.5%
```

### ⚠️ РАСХОЖДЕНИЯ

| Аспект | FAS V2 | fas_smart |
|--------|--------|-----------|
| Threshold | Tier-based (0.02-0.03) | Fixed 0.025 ⚠️ |
| Score | 40/60/80 multi-level | Single score? |

---

## 5. План

fas_smart уже имеет tier-based liq_ratio_threshold. Проверить multi-level scoring.
