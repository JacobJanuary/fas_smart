# FAS V2 CVD Cumulative Analysis Report

## 1. Расчёт (calculate_indicators_bulk_v3)

### CVD Delta формула
```sql
cvd_delta = buy_volume - sell_volume
```

### CVD Cumulative формула
```sql
cvd_cumulative = SUM(cvd_delta) OVER (
    PARTITION BY trading_pair_id
    ORDER BY timestamp
)
```

**Running sum** всех cvd_delta с начала истории.

### Данные
- `market_data_aggregated.buy_volume`
- `market_data_aggregated.sell_volume` (= volume - buy_volume)

---

## 2. Использование в Indicator Score

```sql
CASE
    WHEN cvd_cumulative > prev_cvd THEN 20   -- buyers dominating
    WHEN cvd_cumulative < prev_cvd THEN -20  -- sellers dominating
    ELSE 0
END
```

| Условие | Score |
|---------|-------|
| CVD растёт | +20 |
| CVD падает | -20 |

---

## 3. Таймфреймы

Все 4: 15m, 1h, 4h, 1d

---

## 4. Сравнение с fas_smart

### fas_smart (storage.py - PairData)
```python
# In add_candle():
cvd_delta = buy_volume - (volume - buy_volume)  # = 2*buy - volume
self.cvd_cumulative += cvd_delta
```

### fas_smart (calculate_indicator_score)
```python
# CVD direction
if indicators.cvd_cumulative > indicators.prev_cvd_cumulative:
    score += 20
elif indicators.cvd_cumulative < indicators.prev_cvd_cumulative:
    score -= 20
```

### ⚠️ MINOR РАСХОЖДЕНИЕ формулы дельты

| Аспект | FAS V2 | fas_smart |
|--------|--------|-----------|
| CVD Delta | buy - sell | 2*buy - volume |
| Математически | buy - (vol-buy) = 2*buy-vol | 2*buy - vol ✅ |
| Score logic | ±20 based on direction | ±20 based on direction ✅ |

**✅ Формулы эквивалентны:** `buy - sell = buy - (vol - buy) = 2*buy - vol`

---

## 5. Вывод

**✅ ПОЛНОЕ СООТВЕТСТВИЕ**

CVD delta формулы математически эквивалентны. Scoring logic идентичен.
