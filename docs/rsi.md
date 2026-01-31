# FAS V2 RSI Analysis Report

## 1. Расчёт RSI (calculate_rsi_batch_v2)

### Формула
```sql
RSI = 100 - (100 / (1 + avg_gain / avg_loss))
```

### Метод
- **SMA-based** (Simple Moving Average), НЕ EMA
- Window: `ROWS BETWEEN p_period PRECEDING AND CURRENT ROW`
- Данные: `fas_v2.market_data_aggregated.close_price`

### Требуемые данные
- `p_period * 2` свечей lookback для window calculation
- Default period: 14

---

## 2. Использование в Indicator Score

```sql
CASE
    WHEN li.rsi > 60 THEN 5
    WHEN li.rsi < 40 THEN -5
    ELSE 0
END
```

- **RSI > 60** → +5 (bullish momentum)
- **RSI < 40** → -5 (bearish momentum)
- Между 40-60 → 0

---

## 3. Таймфреймы

| Timeframe | Records | Avg RSI |
|-----------|---------|---------|
| 15m | 2M | 49.3 |
| 1h | 517K | 48.7 |
| 4h | 129K | 47.7 |
| 1d | 21K | 46.6 |

---

## 4. Сравнение с fas_smart

### fas_smart (indicators.py)
```python
def calculate_rsi(closes, period=14, ema_gain=None, ema_loss=None):
    # Uses EMA-based smoothing (Wilder's method)
    ...
```

### ⚠️ НЕСООТВЕТСТВИЕ: SMA vs EMA

| Аспект | FAS V2 | fas_smart |
|--------|--------|-----------|
| Метод | SMA | EMA (Wilder) |
| Formula | AVG() over window | alpha * current + (1-alpha) * prev |

### Indicator Score
```python
# fas_smart: calculate_indicator_score
if indicators.rsi > 60:
    score += 5
elif indicators.rsi < 40:
    score -= 5
```
**✅ СООТВЕТСТВУЕТ** порогам FAS V2

---

## 5. План исправлений

### Опция A: Изменить fas_smart на SMA (FAS V2 parity)
- Заменить EMA на SMA в calculate_rsi()
- Проще, но менее стандартно

### Опция B: Оставить EMA (industry standard)
- EMA = стандартный Wilder's RSI
- Небольшое расхождение в значениях, но пороги те же

### Рекомендация
**Опция B** - оставить EMA. Разница минимальна (~1-2 пункта RSI), пороги (40/60) одинаковые.
