# FUNDING_EXTREME Pattern Analysis

## 1. Статус в FAS V2

**❌ НЕТ паттерна FUNDING_EXTREME!**

В FAS V2 существует `FUNDING_DIVERGENCE`, который включает SHORT_SQUEEZE логику.

---

## 2. FAS V2: FUNDING_DIVERGENCE

### Условия
```sql
WHERE (funding_rate_avg < -0.0003 AND price_change > 1.0) OR
      (funding_rate_avg > 0.001 AND price_change < -1.0)
```

### Score
```sql
CASE
    WHEN funding_rate < -0.001 THEN 70.0   -- Strong short squeeze
    WHEN funding_rate < -0.0005 THEN 50.0
    WHEN funding_rate > 0.001 THEN -50.0   -- Long squeeze
    ELSE 30.0
END
```

### Thresholds
- Strong negative: **-0.001** (0.1%)
- Moderate negative: **-0.0005** (0.05%)
- Positive extreme: **0.001** (0.1%)

---

## 3. fas_smart FUNDING_EXTREME

### Текущие thresholds
```python
funding_extreme_threshold: float = 0.001  # 0.1%
```

### ⚠️ НЕСООТВЕТСТВИЕ КОНЦЕПЦИИ

fas_smart имеет `FUNDING_EXTREME` — отдельный паттерн.
FAS V2 использует `FUNDING_DIVERGENCE` с multi-level thresholds.

---

## 4. План

**Переименовать/объединить:** fas_smart FUNDING_EXTREME → FUNDING_DIVERGENCE для соответствия FAS V2, или оставить как расширение.
