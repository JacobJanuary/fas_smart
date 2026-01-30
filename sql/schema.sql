-- ============================================
-- FAS_SMART Database Schema
-- Rolling Window Real-Time Signal System
-- ============================================

-- Создание базы данных (выполнить отдельно под superuser)
-- CREATE DATABASE fas_smart WITH ENCODING 'UTF8';

-- ============================================
-- SCHEMA
-- ============================================
CREATE SCHEMA IF NOT EXISTS fas_smart;
SET search_path TO fas_smart, public;

-- ============================================
-- ENUM TYPES
-- ============================================

CREATE TYPE fas_smart.timeframe_enum AS ENUM (
    '1m', '5m', '15m', '1h', '4h', '1d'
);

CREATE TYPE fas_smart.liquidity_tier AS ENUM (
    'TIER_1', 'TIER_2', 'TIER_3'
);

CREATE TYPE fas_smart.pattern_type AS ENUM (
    'VOLUME_ANOMALY',
    'OI_EXPLOSION',
    'OI_COLLAPSE',
    'LIQUIDATION_CASCADE',
    'SQUEEZE_IGNITION',
    'CVD_PRICE_DIVERGENCE',
    'FUNDING_DIVERGENCE',
    'SMART_MONEY_DIVERGENCE',
    'MOMENTUM_EXHAUSTION',
    'ACCUMULATION',
    'DISTRIBUTION'
);

CREATE TYPE fas_smart.signal_direction AS ENUM (
    'LONG', 'SHORT', 'NEUTRAL'
);

CREATE TYPE fas_smart.market_regime AS ENUM (
    'BULL', 'BEAR', 'NEUTRAL'
);

-- ============================================
-- TABLE 1: trading_pairs
-- ============================================

CREATE TABLE fas_smart.trading_pairs (
    id              SERIAL PRIMARY KEY,
    symbol          VARCHAR(20) NOT NULL UNIQUE,
    base_asset      VARCHAR(10),
    quote_asset     VARCHAR(10) DEFAULT 'USDT',
    tier            fas_smart.liquidity_tier DEFAULT 'TIER_3',
    is_active       BOOLEAN DEFAULT true,
    avg_volume_24h  NUMERIC(24,2),
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_pairs_active ON fas_smart.trading_pairs(is_active) WHERE is_active = true;
CREATE INDEX idx_pairs_tier ON fas_smart.trading_pairs(tier);

COMMENT ON TABLE fas_smart.trading_pairs IS 'Торговые пары для мониторинга';

-- ============================================
-- TABLE 2: candles_1m (partitioned)
-- ============================================

CREATE TABLE fas_smart.candles_1m (
    pair_id         INT NOT NULL,
    ts              TIMESTAMPTZ NOT NULL,
    o               NUMERIC(18,8) NOT NULL,       -- open
    h               NUMERIC(18,8) NOT NULL,       -- high
    l               NUMERIC(18,8) NOT NULL,       -- low
    c               NUMERIC(18,8) NOT NULL,       -- close
    v               NUMERIC(24,8) NOT NULL,       -- volume
    bv              NUMERIC(24,8) NOT NULL,       -- buy_volume (taker)
    oi              NUMERIC(24,8),                -- open_interest
    fr              NUMERIC(12,8),                -- funding_rate
    
    PRIMARY KEY (pair_id, ts)
) PARTITION BY RANGE (ts);

-- Создаём партиции на 7 дней вперёд (автоматизировать через cron)
CREATE TABLE fas_smart.candles_1m_default PARTITION OF fas_smart.candles_1m DEFAULT;

COMMENT ON TABLE fas_smart.candles_1m IS '1-минутные свечи для rolling window агрегации';

-- ============================================
-- TABLE 3: liquidations
-- ============================================

CREATE TABLE fas_smart.liquidations (
    id              BIGSERIAL,
    pair_id         INT NOT NULL,
    ts              TIMESTAMPTZ NOT NULL,
    side            CHAR(1) NOT NULL,             -- 'L' = long liquidated, 'S' = short
    qty             NUMERIC(24,8) NOT NULL,
    price           NUMERIC(18,8) NOT NULL,
    
    PRIMARY KEY (id, ts)
) PARTITION BY RANGE (ts);

CREATE TABLE fas_smart.liquidations_default PARTITION OF fas_smart.liquidations DEFAULT;

CREATE INDEX idx_liq_pair_ts ON fas_smart.liquidations(pair_id, ts DESC);

COMMENT ON TABLE fas_smart.liquidations IS 'Ликвидации с биржи в реальном времени';

-- ============================================
-- TABLE 4: signals (основная таблица результатов)
-- ============================================

CREATE TABLE fas_smart.signals (
    id              BIGSERIAL PRIMARY KEY,
    pair_id         INT NOT NULL,
    ts              TIMESTAMPTZ NOT NULL,         -- время генерации
    
    -- Scores
    pattern_score   NUMERIC(8,2) DEFAULT 0,
    indicator_score NUMERIC(8,2) DEFAULT 0,
    total_score     NUMERIC(8,2) GENERATED ALWAYS AS (pattern_score + indicator_score) STORED,
    
    -- Direction
    direction       fas_smart.signal_direction,
    confidence      NUMERIC(5,2),                 -- 0-100%
    
    -- Market context
    market_regime   fas_smart.market_regime,
    
    -- Prices at signal
    entry_price     NUMERIC(18,8),
    
    -- Metadata
    patterns_json   JSONB,                        -- детали паттернов
    indicators_json JSONB,                        -- значения индикаторов
    
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    
    CONSTRAINT uq_signal_pair_ts UNIQUE (pair_id, ts)
);

CREATE INDEX idx_signals_ts ON fas_smart.signals(ts DESC);
CREATE INDEX idx_signals_score ON fas_smart.signals(total_score DESC) WHERE total_score != 0;
CREATE INDEX idx_signals_pair ON fas_smart.signals(pair_id, ts DESC);

COMMENT ON TABLE fas_smart.signals IS 'Сгенерированные сигналы с total_score';

-- ============================================
-- TABLE 5: signal_patterns (детализация паттернов)
-- ============================================

CREATE TABLE fas_smart.signal_patterns (
    id              BIGSERIAL PRIMARY KEY,
    signal_id       BIGINT NOT NULL REFERENCES fas_smart.signals(id) ON DELETE CASCADE,
    pattern         fas_smart.pattern_type NOT NULL,
    timeframe       fas_smart.timeframe_enum NOT NULL,
    score_impact    NUMERIC(8,2) NOT NULL,
    confidence      NUMERIC(5,2),
    details         JSONB,
    
    CONSTRAINT uq_pattern_signal UNIQUE (signal_id, pattern, timeframe)
);

CREATE INDEX idx_patterns_signal ON fas_smart.signal_patterns(signal_id);

COMMENT ON TABLE fas_smart.signal_patterns IS 'Обнаруженные паттерны для каждого сигнала';

-- ============================================
-- TABLE 6: config (параметры системы)
-- ============================================

CREATE TABLE fas_smart.config (
    key             VARCHAR(50) PRIMARY KEY,
    value           TEXT NOT NULL,
    value_type      VARCHAR(10) DEFAULT 'string', -- string, int, float, bool, json
    description     TEXT,
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Дефолтные настройки
INSERT INTO fas_smart.config (key, value, value_type, description) VALUES
    ('volume_zscore_threshold', '3.0', 'float', 'Порог для VOLUME_ANOMALY'),
    ('oi_explosion_threshold', '7.0', 'float', 'Порог для OI_EXPLOSION'),
    ('oi_collapse_threshold', '-10.0', 'float', 'Порог для OI_COLLAPSE'),
    ('liquidation_ratio_threshold', '0.03', 'float', 'Минимальный ratio ликвидаций'),
    ('min_signal_score', '30', 'int', 'Минимальный score для отправки сигнала'),
    ('rolling_window_minutes', '15', 'int', 'Размер rolling window'),
    ('history_days', '7', 'int', 'Сколько дней хранить 1m свечи');

COMMENT ON TABLE fas_smart.config IS 'Конфигурация системы';

-- ============================================
-- TABLE 7: pattern_thresholds (оптимизированные пороги)
-- ============================================

CREATE TABLE fas_smart.pattern_thresholds (
    id              SERIAL PRIMARY KEY,
    tier            fas_smart.liquidity_tier NOT NULL,
    timeframe       fas_smart.timeframe_enum NOT NULL,
    pattern         fas_smart.pattern_type NOT NULL,
    threshold_name  VARCHAR(30) NOT NULL,         -- 'minimum', 'atr_multiplier', etc
    threshold_value NUMERIC(10,4) NOT NULL,
    is_active       BOOLEAN DEFAULT true,
    updated_at      TIMESTAMPTZ DEFAULT NOW(),
    
    CONSTRAINT uq_threshold UNIQUE (tier, timeframe, pattern, threshold_name)
);

COMMENT ON TABLE fas_smart.pattern_thresholds IS 'Оптимизированные пороги по tier/timeframe';

-- ============================================
-- TABLE 8: execution_log (логирование)
-- ============================================

CREATE TABLE fas_smart.execution_log (
    id              BIGSERIAL PRIMARY KEY,
    ts              TIMESTAMPTZ DEFAULT NOW(),
    operation       VARCHAR(50) NOT NULL,
    duration_ms     INT,
    records_count   INT,
    status          VARCHAR(20) DEFAULT 'success',
    error_message   TEXT,
    details         JSONB
);

CREATE INDEX idx_log_ts ON fas_smart.execution_log(ts DESC);

COMMENT ON TABLE fas_smart.execution_log IS 'Лог выполнения операций';

-- ============================================
-- HELPER FUNCTIONS
-- ============================================

-- Функция получения config значения
CREATE OR REPLACE FUNCTION fas_smart.get_config(p_key VARCHAR)
RETURNS TEXT AS $$
    SELECT value FROM fas_smart.config WHERE key = p_key;
$$ LANGUAGE sql STABLE;

CREATE OR REPLACE FUNCTION fas_smart.get_config_int(p_key VARCHAR)
RETURNS INT AS $$
    SELECT value::INT FROM fas_smart.config WHERE key = p_key;
$$ LANGUAGE sql STABLE;

CREATE OR REPLACE FUNCTION fas_smart.get_config_float(p_key VARCHAR)
RETURNS NUMERIC AS $$
    SELECT value::NUMERIC FROM fas_smart.config WHERE key = p_key;
$$ LANGUAGE sql STABLE;

-- ============================================
-- PARTITION MANAGEMENT
-- ============================================

-- Функция создания партиций на N дней вперёд
CREATE OR REPLACE FUNCTION fas_smart.create_partitions(p_days INT DEFAULT 7)
RETURNS INT AS $$
DECLARE
    v_date DATE;
    v_partition_name TEXT;
    v_count INT := 0;
BEGIN
    FOR i IN 0..p_days LOOP
        v_date := CURRENT_DATE + i;
        
        -- candles_1m partitions
        v_partition_name := 'candles_1m_' || TO_CHAR(v_date, 'YYYYMMDD');
        IF NOT EXISTS (
            SELECT 1 FROM pg_tables 
            WHERE schemaname = 'fas_smart' AND tablename = v_partition_name
        ) THEN
            EXECUTE format(
                'CREATE TABLE fas_smart.%I PARTITION OF fas_smart.candles_1m 
                 FOR VALUES FROM (%L) TO (%L)',
                v_partition_name, v_date, v_date + 1
            );
            v_count := v_count + 1;
        END IF;
        
        -- liquidations partitions
        v_partition_name := 'liquidations_' || TO_CHAR(v_date, 'YYYYMMDD');
        IF NOT EXISTS (
            SELECT 1 FROM pg_tables 
            WHERE schemaname = 'fas_smart' AND tablename = v_partition_name
        ) THEN
            EXECUTE format(
                'CREATE TABLE fas_smart.%I PARTITION OF fas_smart.liquidations 
                 FOR VALUES FROM (%L) TO (%L)',
                v_partition_name, v_date, v_date + 1
            );
            v_count := v_count + 1;
        END IF;
    END LOOP;
    
    RETURN v_count;
END;
$$ LANGUAGE plpgsql;

-- Функция удаления старых партиций
CREATE OR REPLACE FUNCTION fas_smart.drop_old_partitions(p_keep_days INT DEFAULT 7)
RETURNS INT AS $$
DECLARE
    v_cutoff DATE;
    v_partition RECORD;
    v_count INT := 0;
BEGIN
    v_cutoff := CURRENT_DATE - p_keep_days;
    
    FOR v_partition IN
        SELECT tablename FROM pg_tables
        WHERE schemaname = 'fas_smart'
          AND (tablename LIKE 'candles_1m_%' OR tablename LIKE 'liquidations_%')
          AND tablename !~ '_default$'
          AND SUBSTRING(tablename FROM '\d{8}$')::DATE < v_cutoff
    LOOP
        EXECUTE format('DROP TABLE fas_smart.%I', v_partition.tablename);
        v_count := v_count + 1;
    END LOOP;
    
    RETURN v_count;
END;
$$ LANGUAGE plpgsql;

-- ============================================
-- INITIAL SETUP
-- ============================================

-- Создаём партиции на ближайшие 7 дней
SELECT fas_smart.create_partitions(7);

-- ============================================
-- GRANTS (настроить под своего пользователя)
-- ============================================

-- GRANT USAGE ON SCHEMA fas_smart TO your_app_user;
-- GRANT ALL ON ALL TABLES IN SCHEMA fas_smart TO your_app_user;
-- GRANT ALL ON ALL SEQUENCES IN SCHEMA fas_smart TO your_app_user;
-- GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA fas_smart TO your_app_user;
