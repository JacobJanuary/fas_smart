# OI_DIVERGENCE Pattern Analysis

## 1. Статус в FAS V2

**❌ НЕ СУЩЕСТВУЕТ в FAS V2!**

FAS V2 signal_pattern_type enum содержит только 11 паттернов:
- OI_EXPLOSION
- LIQUIDATION_CASCADE
- SQUEEZE_IGNITION
- CVD_PRICE_DIVERGENCE
- FUNDING_DIVERGENCE
- VOLUME_ANOMALY
- ACCUMULATION
- DISTRIBUTION
- OI_COLLAPSE
- SMART_MONEY_DIVERGENCE
- MOMENTUM_EXHAUSTION

---

## 2. Реализация в fas_smart

OI_DIVERGENCE — это **наше собственное расширение**, добавленное в fas_smart.

### Логика (_detect_oi_divergence)
```python
# OI up + price down = bearish divergence
if oi_delta > threshold and price_change < -0.5:
    return Pattern(score=-25.0, direction='BEARISH')

# OI down + price up = bullish divergence
if oi_delta < -threshold and price_change > 0.5:
    return Pattern(score=25.0, direction='BULLISH')
```

### Параметры
- `oi_divergence_threshold`: 3.0%
- Score: ±25

---

## 3. Вывод

**⚠️ fas_smart EXTENSION** — не требует соответствия FAS V2, т.к. этого паттерна там нет.

Рекомендация: оставить как есть, это полезное расширение для обнаружения дивергенций OI/Price.
