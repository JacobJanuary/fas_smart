# MACD_DIVERGENCE Pattern Analysis

## 1. Статус в FAS V2

**❌ НЕТ паттерна MACD_DIVERGENCE!**

FAS V2 signal_pattern_type enum содержит 11 паттернов, MACD_DIVERGENCE отсутствует.

---

## 2. fas_smart: MACD_DIVERGENCE

### Текущая реализация
```python
def _detect_macd_divergence(self, indicators, prev_indicators):
    # Price up but histogram weakening = bearish
    if price_change > 1.0 and histogram < prev_histogram and histogram > 0:
        return Pattern(score=-15.0, type='BEARISH')
    
    # Price down but histogram strengthening = bullish
    if price_change < -1.0 and histogram > prev_histogram and histogram < 0:
        return Pattern(score=15.0, type='BULLISH')
```

### Параметры
- Score: **±15**

---

## 3. Вывод

**⚠️ fas_smart EXTENSION** — полезный паттерн для обнаружения дивергенций MACD/Price.

Технически обоснован: классический индикатор ослабления тренда.

Рекомендация: **оставить** как есть.
