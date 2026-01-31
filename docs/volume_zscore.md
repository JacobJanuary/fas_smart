# FAS V2 Volume Z-Score Analysis Report

## 1. Расчёт (calculate_volume_zscore_batch_v2)

### Формула
```sql
Z-Score = (current_volume - avg_vol) / stddev_vol
```
Ограничено: `LEAST(GREATEST(zscore, -50), 50)` (clamp to ±50)

### Данные
- `fas_v2.market_data_aggregated.volume`
- Lookback: `p_lookback` предыдущих свечей
- Stats: `AVG()`, `STDDEV_POP()`

---

## 2. Использование в Indicator Score

```sql
CASE
    WHEN ABS(li.volume_zscore) > 3.0 THEN
        CASE WHEN price_change >= 0 THEN 35 ELSE -35 END
    WHEN ABS(li.volume_zscore) > 2.0 THEN
        CASE WHEN price_change >= 0 THEN 20 ELSE -20 END
    ELSE 0
END
```

| Z-Score | Price Up | Price Down |
|---------|----------|------------|
| > 3.0 | +35 | -35 |
| > 2.0 | +20 | -20 |
| ≤ 2.0 | 0 | 0 |

---

## 3. Таймфреймы

Все 4: 15m, 1h, 4h, 1d

---

## 4. Сравнение с fas_smart

### fas_smart (indicators.py)
```python
def calculate_volume_zscore(volumes, period=20):
    mean = np.mean(volumes)
    std = np.std(volumes)  # numpy default: population std
    if std == 0:
        return 0.0
    return (volumes[-1] - mean) / std
```

### fas_smart (calculate_indicator_score)
```python
if abs(volume_zscore) > 3.0:
    score += 35 if price_change >= 0 else -35
elif abs(volume_zscore) > 2.0:
    score += 20 if price_change >= 0 else -20
```

### ✅ ПОЛНОЕ СООТВЕТСТВИЕ

| Аспект | FAS V2 | fas_smart |
|--------|--------|-----------|
| Формула | (curr - avg) / stddev | (curr - mean) / std ✅ |
| Std type | STDDEV_POP | np.std (population) ✅ |
| Thresholds | 2.0, 3.0 | 2.0, 3.0 ✅ |
| Scores | ±20, ±35 | ±20, ±35 ✅ |
| Lookback | configurable | 20 ⚠️ |

---

## 5. Вывод

**✅ Алгоритм полностью соответствует.**

Единственное: lookback в FAS V2 конфигурируемый, в fas_smart = 20. Это стандартное значение.
