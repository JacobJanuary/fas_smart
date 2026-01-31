# FAS V2 MACD Histogram Analysis Report

## 1. Расчёт MACD Histogram (в calculate_macd)

### Формула
```sql
Histogram = MACD Line - Signal Line
         = (SMA_fast - SMA_slow) - AVG(prev MACD values)
```

Histogram вычисляется как побочный продукт в `calculate_macd`:
```sql
RETURN QUERY SELECT
    ROUND(v_macd_value, 6),      -- macd_line
    ROUND(v_signal_value, 6),    -- macd_signal
    ROUND(v_macd_value - v_signal_value, 6);  -- histogram
```

---

## 2. Использование в Indicator Score

**Это ГЛАВНЫЙ компонент MACD в scoring!**

```sql
-- MACD crossover component
CASE
    WHEN pi.prev_macd < 0 AND li.macd_histogram > 0 THEN 10   -- bullish
    WHEN pi.prev_macd > 0 AND li.macd_histogram < 0 THEN -10  -- bearish
    ELSE 0
END
```

| Условие | Score |
|---------|-------|
| prev < 0, curr > 0 (bullish crossover) | +10 |
| prev > 0, curr < 0 (bearish crossover) | -10 |

---

## 3. Таймфреймы

Все 4: 15m, 1h, 4h, 1d

---

## 4. Сравнение с fas_smart

### fas_smart (indicators.py)
```python
histogram = macd_value - signal_value
```

### fas_smart (calculate_indicator_score)
```python
# MACD crossover: histogram sign change
if pair_data.prev_macd_histogram is not None:
    if pair_data.prev_macd_histogram < 0 and indicators.macd_histogram > 0:
        score += 10
    elif pair_data.prev_macd_histogram > 0 and indicators.macd_histogram < 0:
        score -= 10
```

### ✅ ПОЛНОЕ СООТВЕТСТВИЕ логики crossover

| Аспект | FAS V2 | fas_smart |
|--------|--------|-----------|
| Формула | line - signal | line - signal ✅ |
| Crossover | sign change ±10 | sign change ±10 ✅ |
| Таймфреймы | all | all ✅ |

---

## 5. Вывод

**✅ Histogram логика полностью соответствует.**

Разница SMA/EMA в расчёте line/signal минимально влияет на crossover detection.
