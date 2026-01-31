# FAS V2 prev_cvd_cumulative Analysis Report

## 1. Расчёт

**prev_cvd_cumulative НЕ является отдельным индикатором!**

Это предыдущее значение `cvd_cumulative`, получаемое динамически:

```sql
-- In calculate_indicator_scores_batch_v2
prev_indicators AS (
    SELECT DISTINCT ON (i.trading_pair_id, i.timeframe)
        i.trading_pair_id,
        i.timeframe,
        i.cvd_cumulative as prev_cvd,  -- <-- динамически
        i.macd_histogram as prev_macd
    FROM fas_v2.indicators i
    ...
    AND i.timestamp < li.timestamp  -- предыдущий timestamp
    ORDER BY i.timestamp DESC
)
```

---

## 2. Использование в Indicator Score

```sql
CASE
    WHEN li.cvd_cumulative > pi.prev_cvd THEN 20
    WHEN li.cvd_cumulative < pi.prev_cvd THEN -20
    ELSE 0
END
```

Сравнивает **текущий CVD** с **предыдущим** для определения направления.

---

## 3. Сравнение с fas_smart

### fas_smart (IndicatorResult)
```python
@dataclass
class IndicatorResult:
    cvd_cumulative: float
    prev_cvd_cumulative: float  # stored in-memory
```

### fas_smart (storage.py)
```python
# prev хранится как предыдущее значение в памяти
self.prev_cvd_cumulative = self.cvd_cumulative
self.cvd_cumulative += cvd_delta
```

### ✅ ПОЛНОЕ СООТВЕТСТВИЕ

| Аспект | FAS V2 | fas_smart |
|--------|--------|-----------|
| Источник | динамический запрос | in-memory ✅ |
| Логика | сравнение curr vs prev | сравнение curr vs prev ✅ |
