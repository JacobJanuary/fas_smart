-- Add missing pattern types for signal detection
-- Run: psql -d tradingbot_db -f sql/migrations/003_add_pattern_types.sql

ALTER TYPE fas_smart.pattern_type ADD VALUE IF NOT EXISTS 'FUNDING_EXTREME';
ALTER TYPE fas_smart.pattern_type ADD VALUE IF NOT EXISTS 'FUNDING_FLIP';
ALTER TYPE fas_smart.pattern_type ADD VALUE IF NOT EXISTS 'RSI_EXTREME';
ALTER TYPE fas_smart.pattern_type ADD VALUE IF NOT EXISTS 'RSI_DIVERGENCE';
ALTER TYPE fas_smart.pattern_type ADD VALUE IF NOT EXISTS 'OI_DIVERGENCE';
ALTER TYPE fas_smart.pattern_type ADD VALUE IF NOT EXISTS 'MACD_CROSSOVER';
ALTER TYPE fas_smart.pattern_type ADD VALUE IF NOT EXISTS 'MACD_DIVERGENCE';
