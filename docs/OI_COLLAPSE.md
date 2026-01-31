# FAS V2 OI_COLLAPSE Pattern Analysis

## 1. Detection Logic (signal_detect_patterns)

### Condition
```sql
oi_delta_pct < v_oi_collapse_threshold  -- v_oi_collapse_threshold = -10.0
```

Т.е. OI упал более чем на 10%.

### Score
```sql
-50.0  -- Fixed negative score (bearish signal)
```

### Confidence
```sql
LEAST(70 + ABS(oi_delta_pct) * 2, 90)
```

---

## 2. Data Required
- `oi_delta_pct` = (current_oi - prev_oi) / prev_oi * 100

---

## 3. Timeframes
Все: 15m, 1h, 4h, 1d

---

## 4. Сравнение с fas_smart

### fas_smart (patterns.py)
```python
# PatternThresholds
oi_collapse_threshold: float = -7.0

def _detect_oi_collapse(self, indicators):
    if indicators.oi_delta_pct < self.thresholds.oi_collapse_threshold:
        return Pattern(score=-50.0, ...)
```

### ⚠️ РАСХОЖДЕНИЕ

| Аспект | FAS V2 | fas_smart |
|--------|--------|-----------|
| Threshold | -10.0% | -7.0% ⚠️ |
| Score | -50.0 | -50.0 ✅ |
| Confidence | 70 + |oi|*2, cap 90 | ✅ |

---

## 5. План исправления

Изменить `oi_collapse_threshold` в `PatternThresholds`:
```python
oi_collapse_threshold: float = -10.0  # FAS V2 value
```
