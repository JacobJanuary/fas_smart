# FAS V2 Smoothed Imbalance Analysis Report

## 1. Расчёт (calculate_smoothed_imbalance)

### Raw Imbalance
```sql
raw_imb = (buy_volume - sell_volume) / NULLIF(volume, 0)
```

### Smoothed Imbalance
```sql
ema_3 = AVG(raw_imb) OVER (ROWS BETWEEN 2 PRECEDING AND CURRENT)  -- 3 периода
ema_6 = AVG(raw_imb) OVER (ROWS BETWEEN 5 PRECEDING AND CURRENT)  -- 6 периодов

smoothed_imbalance = (ema_3 + ema_6) / 2
```

**Комментарий:** Называется "EMA", но на самом деле SMA (AVG) по 3 и 6 периодам.

### Требуемые данные
- 6 свечей минимум
- `market_data_aggregated.buy_volume, sell_volume, volume`

---

## 2. Использование в Indicator Score

```sql
CASE
    WHEN smoothed_imbalance > 0.3 THEN 15
    WHEN smoothed_imbalance > 0.1 THEN 7
    WHEN smoothed_imbalance < -0.3 THEN -15
    WHEN smoothed_imbalance < -0.1 THEN -7
    ELSE 0
END
```

| Threshold | Score |
|-----------|-------|
| > 0.3 | +15 |
| > 0.1 | +7 |
| < -0.1 | -7 |
| < -0.3 | -15 |

---

## 3. Сравнение с fas_smart

### fas_smart
```python
# storage.py - хранит историю imbalance
imbalance_history = deque(maxlen=6)

# indicators.py
def calculate_smoothed_imbalance(imbalances):
    sma3 = np.mean(imbalances[-3:])
    sma6 = np.mean(imbalances[-6:])
    return (sma3 + sma6) / 2
```

### ✅ ПОЛНОЕ СООТВЕТСТВИЕ

| Аспект | FAS V2 | fas_smart |
|--------|--------|-----------|
| Raw | (buy-sell)/vol | (2*buy-vol)/vol (эквивалент) ⚠️ |
| SMA periods | 3, 6 | 3, 6 ✅ |
| Formula | (sma3+sma6)/2 | (sma3+sma6)/2 ✅ |
| Thresholds | 0.1, 0.3 | 0.1, 0.3 ✅ |
| Scores | ±7, ±15 | ±7, ±15 ✅ |

---

## 4. Вывод

**✅ Алгоритм соответствует.**

Разница в raw imbalance (buy-sell vs 2*buy-vol) не влияет — формулы эквивалентны:
`(buy-sell)/vol = (buy - (vol-buy))/vol = (2*buy-vol)/vol`
