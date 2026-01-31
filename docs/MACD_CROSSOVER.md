# MACD_CROSSOVER Pattern Analysis

## 1. Статус в FAS V2

**❌ НЕТ отдельного паттерна MACD_CROSSOVER!**

MACD используется в FAS V2 только в **indicator_score**:

### indicator_score (MACD histogram crossover)
```sql
-- Crossover detection via histogram sign change
CASE
    WHEN macd_histogram > 0 AND prev_macd_histogram < 0 THEN 25  -- Bullish crossover
    WHEN macd_histogram < 0 AND prev_macd_histogram > 0 THEN -25 -- Bearish crossover
    ELSE 0
END
```

---

## 2. fas_smart: indicator_score

### Текущая реализация
```python
# MACD Crossover (in calculate_indicator_score)
if current.macd_histogram > 0 and prev.get('macd_histogram', 0) <= 0:
    score += 25  # Bullish crossover
elif current.macd_histogram < 0 and prev.get('macd_histogram', 0) >= 0:
    score -= 25  # Bearish crossover
```

---

## 3. Сравнение

| Аспект | FAS V2 | fas_smart |
|--------|--------|-----------|
| Detection | histogram sign change | histogram sign change ✅ |
| Score | ±25 | ±25 ✅ |
| Location | indicator_score | indicator_score ✅ |

---

## 4. Вывод

**✅ ПОЛНОЕ СООТВЕТСТВИЕ** — MACD crossover уже реализован как часть indicator_score, не как отдельный паттерн.
