# RSI_DIVERGENCE Pattern Analysis

## 1. Статус в FAS V2

**❌ НЕТ паттерна RSI_DIVERGENCE!**

FAS V2 enum содержит 11 паттернов, RSI_DIVERGENCE отсутствует.

---

## 2. fas_smart: RSI_DIVERGENCE

### Текущая реализация
```python
def _detect_rsi_divergence(self, indicators, prev_indicators):
    # Price up but RSI down = bearish divergence
    if price_change > 1.0 and current_rsi < prev_rsi - 5:
        return Pattern(score=-20.0, type='BEARISH')
    
    # Price down but RSI up = bullish divergence  
    if price_change < -1.0 and current_rsi > prev_rsi + 5:
        return Pattern(score=20.0, type='BULLISH')
```

### Параметры
- Price change threshold: **1.0%**
- RSI change threshold: **5 points**
- Score: **±20**

---

## 3. Вывод

**⚠️ fas_smart EXTENSION** — полезный паттерн для обнаружения дивергенций RSI/Price.

Технически обоснован: классический индикатор разворота.

Рекомендация: **оставить** как есть.
