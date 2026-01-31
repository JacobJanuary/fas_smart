# FAS V2 MACD Analysis Report

## 1. Расчёт MACD (calculate_macd)

### Формула
```sql
MACD Line = SMA(fast) - SMA(slow)
Signal Line = AVG(previous MACD values)
Histogram = MACD Line - Signal Line
```

### ⚠️ ВАЖНО: Использует SMA, НЕ EMA!

### Параметры (из config)
- Fast: 12
- Slow: 26  
- Signal: 9

### Требуемые данные
- Минимум `slow` (26) свечей
- Данные: `fas_v2.market_data_aggregated.close_price`
- Signal: история из `fas_v2.indicators.macd_line`

---

## 2. Использование в Indicator Score

```sql
-- MACD crossover component
CASE
    WHEN prev_macd < 0 AND li.macd_histogram > 0 THEN 10  -- bullish crossover
    WHEN prev_macd > 0 AND li.macd_histogram < 0 THEN -10 -- bearish crossover
    ELSE 0
END
```

- **Bullish crossover** (hist: neg→pos) → +10
- **Bearish crossover** (hist: pos→neg) → -10

---

## 3. Таймфреймы

Все 4: 15m, 1h, 4h, 1d

---

## 4. Сравнение с fas_smart

### fas_smart (indicators.py)
```python
def calculate_macd(closes, fast=12, slow=26, signal=9):
    ema_fast = calculate_ema(closes, fast)  # EMA!
    ema_slow = calculate_ema(closes, slow)  # EMA!
    macd_line = ema_fast - ema_slow
    signal = calculate_ema(valid_macd, signal)  # EMA!
```

### ⚠️ НЕСООТВЕТСТВИЕ: SMA vs EMA

| Аспект | FAS V2 | fas_smart |
|--------|--------|-----------|
| MACD Line | SMA(fast) - SMA(slow) | EMA(fast) - EMA(slow) |
| Signal | AVG(prev MACD) | EMA(MACD) |
| Periods | 12/26/9 ✅ | 12/26/9 ✅ |
| Crossover logic | hist sign change ✅ | hist sign change ✅ |

---

## 5. План исправлений

### Опция A: Изменить fas_smart на SMA
- Заменить EMA на SMA в calculate_macd()
- 100% parity с FAS V2
- НО: SMA MACD — нестандартный

### Опция B: Оставить EMA (industry standard)
- EMA = стандартный MACD (TradingView, etc)
- Небольшое расхождение в значениях
- Crossover логика та же (+10/-10)

### Рекомендация
**Опция B** — EMA более чувствителен и стандартен. Crossover thresholds идентичны.
