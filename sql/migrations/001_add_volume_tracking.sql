-- ============================================
-- Migration: Add volume tracking fields
-- ============================================

-- Новые поля для контроля гистерезиса
ALTER TABLE fas_smart.trading_pairs 
    ADD COLUMN IF NOT EXISTS volume_7d_avg NUMERIC(24,2),
    ADD COLUMN IF NOT EXISTS added_at TIMESTAMPTZ DEFAULT NOW(),
    ADD COLUMN IF NOT EXISTS last_signal_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS deactivated_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS deactivation_reason VARCHAR(50);

-- Индекс для поиска активных пар
CREATE INDEX IF NOT EXISTS idx_pairs_active_tier 
    ON fas_smart.trading_pairs(tier, is_active) 
    WHERE is_active = true;

COMMENT ON COLUMN fas_smart.trading_pairs.volume_7d_avg IS 'Средний объём за 7 дней (скользящий)';
COMMENT ON COLUMN fas_smart.trading_pairs.added_at IS 'Когда пара добавлена в мониторинг';
COMMENT ON COLUMN fas_smart.trading_pairs.last_signal_at IS 'Время последнего сигнала по паре';
COMMENT ON COLUMN fas_smart.trading_pairs.deactivated_at IS 'Когда пара деактивирована';
COMMENT ON COLUMN fas_smart.trading_pairs.deactivation_reason IS 'Причина деактивации';
