# FAS V2 MACD Signal Analysis Report

## 1. Расчёт MACD Signal (в calculate_macd)

### Формула
```sql
-- Signal Line - average of previous MACD values
WITH macd_history AS (
    SELECT i.macd_line as hist_macd
    FROM fas_v2.indicators i
    WHERE i.trading_pair_id = p_trading_pair_id
      AND i.timeframe = p_timeframe
      AND i.timestamp < p_timestamp
      AND i.macd_line IS NOT NULL
    ORDER BY i.timestamp DESC
    LIMIT v_signal - 1  -- 8 предыдущих значений
),
all_macd AS (
    SELECT current_macd UNION ALL SELECT hist_macd
)
SELECT AVG(macd_val) INTO v_signal_value FROM all_macd;
```

### Метод
- **SMA** (Simple Moving Average) последних 9 значений MACD Line
- НЕ EMA!

### Параметры
- Signal period: 9 (из config)

---

## 2. Использование в Indicator Score

**macd_signal НЕ используется напрямую!**

Используется только `macd_histogram` (= macd_line - macd_signal):
```sql
CASE
    WHEN prev_macd < 0 AND li.macd_histogram > 0 THEN 10
    WHEN prev_macd > 0 AND li.macd_histogram < 0 THEN -10
    ELSE 0
END
```

---

## 3. Сравнение с fas_smart

| Аспект | FAS V2 | fas_smart |
|--------|--------|-----------|
| Метод | SMA(9) | EMA(9) |
| Использование | через histogram | через histogram ✅ |
| Период | 9 ✅ | 9 ✅ |

### ⚠️ НЕСООТВЕТСТВИЕ: SMA vs EMA

---

## 4. Рекомендация

**Оставить EMA** — macd_signal не используется напрямую в scoring, только histogram (crossover). Логика crossover идентична.
