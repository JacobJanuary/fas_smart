# FAS V2 Normalized Imbalance Analysis Report

## 1. Расчёт (calculate_indicators_bulk_v3)

### Формула
```sql
normalized_imbalance = (2 * buy_volume - volume) / NULLIF(volume, 0)
```

Эквивалент:
```
= (2 * buy - total) / total
= (buy - sell) / total
```

### Диапазон
- **+1.0** = 100% buy
- **-1.0** = 100% sell
- **0.0** = 50/50

### Данные
- `fas_v2.market_data_aggregated.buy_volume`
- `fas_v2.market_data_aggregated.volume`

---

## 2. Использование в Indicator Score

**Не используется напрямую!**

Используется `smoothed_imbalance`:
```sql
CASE
    WHEN smoothed_imbalance > 0.3 THEN 15
    WHEN smoothed_imbalance > 0.1 THEN 7
    WHEN smoothed_imbalance < -0.3 THEN -15
    WHEN smoothed_imbalance < -0.1 THEN -7
    ELSE 0
END
```

---

## 3. Таймфреймы

Все 4: 15m, 1h, 4h, 1d

---

## 4. Сравнение с fas_smart

### fas_smart (indicators.py)
```python
def calculate_normalized_imbalance(volume: float, buy_volume: float):
    if volume == 0:
        return 0.0
    return (2 * buy_volume - volume) / volume
```

### ✅ ПОЛНОЕ СООТВЕТСТВИЕ

| Аспект | FAS V2 | fas_smart |
|--------|--------|-----------|
| Формула | (2*buy - vol) / vol | (2*buy - vol) / vol ✅ |
| Range | [-1, +1] | [-1, +1] ✅ |
| Div by zero | NULLIF | if volume == 0 ✅ |
