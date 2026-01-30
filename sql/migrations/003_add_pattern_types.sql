-- Add missing pattern types for signal detection
-- Run on server: psql -d tradingbot_db -f sql/migrations/003_add_pattern_types.sql

-- Original missing types
ALTER TYPE fas_smart.pattern_type ADD VALUE IF NOT EXISTS 'FUNDING_EXTREME';
ALTER TYPE fas_smart.pattern_type ADD VALUE IF NOT EXISTS 'FUNDING_FLIP';
ALTER TYPE fas_smart.pattern_type ADD VALUE IF NOT EXISTS 'RSI_EXTREME';
ALTER TYPE fas_smart.pattern_type ADD VALUE IF NOT EXISTS 'RSI_DIVERGENCE';
ALTER TYPE fas_smart.pattern_type ADD VALUE IF NOT EXISTS 'OI_DIVERGENCE';
ALTER TYPE fas_smart.pattern_type ADD VALUE IF NOT EXISTS 'MACD_CROSSOVER';
ALTER TYPE fas_smart.pattern_type ADD VALUE IF NOT EXISTS 'MACD_DIVERGENCE';

-- New FAS V2 patterns
ALTER TYPE fas_smart.pattern_type ADD VALUE IF NOT EXISTS 'OI_COLLAPSE';
ALTER TYPE fas_smart.pattern_type ADD VALUE IF NOT EXISTS 'ACCUMULATION';
ALTER TYPE fas_smart.pattern_type ADD VALUE IF NOT EXISTS 'DISTRIBUTION';
ALTER TYPE fas_smart.pattern_type ADD VALUE IF NOT EXISTS 'MOMENTUM_EXHAUSTION';
ALTER TYPE fas_smart.pattern_type ADD VALUE IF NOT EXISTS 'SQUEEZE_IGNITION';
