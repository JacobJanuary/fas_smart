# FUNDING_FLIP Pattern Analysis

## 1. Статус в FAS V2

**❌ НЕТ паттерна FUNDING_FLIP!**

В FAS V2 enum signal_pattern_type содержит 11 паттернов, FUNDING_FLIP отсутствует:
- OI_EXPLOSION
- LIQUIDATION_CASCADE
- SQUEEZE_IGNITION
- CVD_PRICE_DIVERGENCE
- FUNDING_DIVERGENCE ← (это ближайший аналог)
- VOLUME_ANOMALY
- ACCUMULATION
- DISTRIBUTION
- OI_COLLAPSE
- SMART_MONEY_DIVERGENCE
- MOMENTUM_EXHAUSTION

---

## 2. fas_smart: FUNDING_FLIP

### Текущая реализация
```python
def _detect_funding_flip(self, pair_data, prev_funding):
    # Positive to Negative flip (bearish)
    if prev_funding > 0.0001 and current_funding < -0.0001:
        return Pattern(score=-30.0, ...)
    
    # Negative to Positive flip (bullish)
    if prev_funding < -0.0001 and current_funding > 0.0001:
        return Pattern(score=30.0, ...)
```

---

## 3. Вывод

**⚠️ fas_smart EXTENSION** — наше расширение, не из FAS V2.

Логика технически обоснована (смена знака funding = изменение рыночного сентимента), но это не part of FAS V2 reference.

Рекомендация: **оставить** — полезный паттерн для обнаружения изменения сентимента.
