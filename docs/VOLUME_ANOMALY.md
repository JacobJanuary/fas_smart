# FAS V2 VOLUME_ANOMALY Pattern Analysis

## 1. Detection Logic (signal_detect_patterns)

### Condition
```sql
ABS(volume_zscore) > (opt_volume_threshold + volume_atr_mult * current_atr)
```

### Thresholds (Tier-Based)
| Tier | Threshold |
|------|-----------|
| TIER_1 (>= $100M) | 2.5 |
| TIER_2 (>= $10M) | 3.0 |
| TIER_3 (< $10M) | 3.5 |

### Score
```sql
CASE WHEN volume_zscore > 0 THEN 30.0 ELSE -30.0 END
```

### Confidence
```sql
LEAST(50 + ABS(volume_zscore) * 10, 95)
```

---

## 2. Data Required
- `volume_zscore` from indicators
- `tier` for threshold selection

---

## 3. Timeframes
Все: 15m, 1h, 4h, 1d

---

## 4. Сравнение с fas_smart

### fas_smart (patterns.py) — ПОСЛЕ ИСПРАВЛЕНИЯ
```python
# PatternThresholds.for_tier() возвращает tier-adjusted thresholds
thresholds = PatternThresholds.for_tier(pair_data.tier)
# TIER_1: volume_zscore_threshold = 2.5
# TIER_2: volume_zscore_threshold = 3.0
# TIER_3: volume_zscore_threshold = 3.5
```

### ✅ ПОЛНОЕ СООТВЕТСТВИЕ

| Аспект | FAS V2 | fas_smart |
|--------|--------|-----------|
| Tier-based thresholds | ✅ | ✅ |
| TIER_1 threshold | 2.5 | 2.5 ✅ |
| TIER_2 threshold | 3.0 | 3.0 ✅ |
| TIER_3 threshold | 3.5 | 3.5 ✅ |
| Score | ±30 | ±30 ✅ |
| Confidence | 50 + \|z\|*10, cap 95 | ✅ |

---

## 5. Изменения внесены

- `config.py`: добавлен `ThresholdConfig.get_tier(volume_24h)`
- `patterns.py`: добавлен `PatternThresholds.for_tier(tier)`
- `storage.py`: добавлено поле `tier` в PairData
- `service.py`: tier-aware PatternDetector
- `warmup.py`: `update_tiers()` из Binance API
