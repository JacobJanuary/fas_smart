# FAS V2 OI_EXPLOSION Pattern Analysis

## 1. Detection Logic (signal_detect_patterns)

### Condition
```sql
oi_delta_pct > 7.0  -- Phase 7: raised from 5.0
```

### Thresholds (Tier-Based)
| Tier | Threshold |
|------|-----------|
| TIER_1 (>= $100M) | 3.0% |
| TIER_2 (>= $10M) | 4.0% |
| TIER_3 (< $10M) | 5.0% |

**Note:** FAS V2 использует фиксированный 7.0% (Phase 7 optimization), но tier-based adaptive thresholds доступны.

### Score
```sql
30.0  -- Fixed positive score
```

### Confidence
```sql
LEAST(60 + oi_delta_pct * 2, 95)
```

---

## 2. Data Required
- `oi_delta_pct` = (current_oi - prev_oi) / prev_oi * 100
- `tier` for threshold selection

---

## 3. Timeframes
Все: 15m, 1h, 4h, 1d

---

## 4. Сравнение с fas_smart

### fas_smart (patterns.py)
```python
def _detect_oi_explosion(self, indicators):
    if indicators.oi_delta_pct > self.thresholds.oi_explosion_threshold:
        # thresholds.oi_explosion_threshold per tier:
        # TIER_1: 3.0, TIER_2: 4.0, TIER_3: 5.0
        return Pattern(score=30.0, ...)
```

### ✅ СООТВЕТСТВИЕ (с улучшением)

| Аспект | FAS V2 | fas_smart |
|--------|--------|-----------|
| Condition | oi_delta > 7.0 (fixed) | oi_delta > tier_threshold ✅ |
| TIER_1 threshold | 3.0 (adaptive) | 3.0 ✅ |
| TIER_2 threshold | 4.0 (adaptive) | 4.0 ✅ |
| TIER_3 threshold | 5.0 (adaptive) | 5.0 ✅ |
| Score | 30.0 | 30.0 ✅ |
| Confidence | 60 + oi*2, cap 95 | ✅ |

**fas_smart использует tier-adaptive thresholds (лучше чем fixed 7.0)**
