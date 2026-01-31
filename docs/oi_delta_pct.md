# FAS V2 oi_delta_pct Analysis Report

## 1. Расчёт (calculate_oi_delta)

### Формула
```sql
oi_delta_pct = ((current_oi - previous_oi) / previous_oi) * 100
```

### Данные
- `market_data_aggregated.open_interest` (текущий)
- Предыдущий `open_interest` (по timestamp)

---

## 2. Использование в Indicator Score

**Не используется напрямую в indicator_score!**

Используется в паттернах:
- `OI_EXPLOSION`: oi_delta > 7%
- `OI_COLLAPSE`: oi_delta < -7%
- `OI_DIVERGENCE`: oi vs price divergence

---

## 3. Таймфреймы

Все 4: 15m, 1h, 4h, 1d

---

## 4. Сравнение с fas_smart

### fas_smart (ws/handlers.py)
```python
# В handle_open_interest
if prev_oi and prev_oi > 0:
    oi_delta_pct = ((current_oi - prev_oi) / prev_oi) * 100
```

### fas_smart (patterns.py)
```python
OI_EXPLOSION_THRESHOLD = 7.0   # %
OI_COLLAPSE_THRESHOLD = -7.0   # %
```

### ✅ ПОЛНОЕ СООТВЕТСТВИЕ

| Аспект | FAS V2 | fas_smart |
|--------|--------|-----------|
| Formula | (curr-prev)/prev * 100 | (curr-prev)/prev * 100 ✅ |
| Div by zero | check prev > 0 | check prev > 0 ✅ |
| Thresholds | 7% | 7% ✅ |
