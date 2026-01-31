# FAS V2 MOMENTUM_EXHAUSTION Pattern Analysis

## 1. Detection Logic

### Conditions
```sql
WHERE (rsi > 70 AND price_change < 0.5 AND volume_zscore < -1) OR
      (rsi < 30 AND price_change > -0.5 AND volume_zscore < -1)
```

RSI extreme + small price move + low volume = exhaustion.

### Score (Dynamic)
```sql
CASE
    WHEN rsi > 70 THEN -40.0  -- Overbought → bearish reversal
    ELSE 40.0                  -- Oversold → bullish reversal
END
```

### Confidence
```sql
LEAST(70 + (distance from 50), 90)
```

---

## 2. Data Required
- `rsi`
- `price_change_pct`
- `volume_zscore`

---

## 3. Timeframes
Все: 15m, 1h, 4h, 1d

---

## 4. Сравнение с fas_smart

### fas_smart
MOMENTUM_EXHAUSTION отсутствует как отдельный паттерн.
RSI_EXTREME частично покрывает эту функциональность.

### ⚠️ ОТСУТСТВУЕТ В fas_smart

Нужно добавить паттерн с условиями:
- RSI > 70 or < 30
- Price stalled (< 0.5%)
- Low volume (zscore < -1)

---

## 5. План

Добавить `_detect_momentum_exhaustion` с FAS V2 логикой.
