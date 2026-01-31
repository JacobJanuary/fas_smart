# FAS V2 Price Change Pct Analysis Report

## 1. Расчёт (calculate_indicators_bulk_v3)

### Формула
```sql
price_change_pct = ROUND(
    (close_price - prev_close) / NULLIF(prev_close, 0) * 100, 
    4
)
```

### Данные
- `fas_v2.indicators.close_price`
- `LAG(close_price)` для предыдущей свечи
- Требует: 2 свечи минимум

---

## 2. Использование в Indicator Score

Косвенное — влияет на direction Volume Z-Score:
```sql
CASE WHEN price_change_pct >= 0 THEN +score ELSE -score END
```

**Напрямую НЕ добавляет к score**, только определяет знак компонентов.

---

## 3. Таймфреймы

Все 4: 15m, 1h, 4h, 1d

---

## 4. Сравнение с fas_smart

### fas_smart (indicators.py)
```python
def calculate_price_change(closes: np.ndarray) -> Optional[float]:
    if len(closes) < 2:
        return None
    prev = closes[-2]
    if prev == 0:
        return 0.0
    return (closes[-1] - prev) / prev * 100
```

### ✅ ПОЛНОЕ СООТВЕТСТВИЕ

| Аспект | FAS V2 | fas_smart |
|--------|--------|-----------|
| Формула | (curr - prev) / prev * 100 | (curr - prev) / prev * 100 ✅ |
| Division by zero | NULLIF | if prev == 0 ✅ |
| Precision | 4 decimals | float |
| Min data | 2 candles | 2 candles ✅ |
