-- Fix: Увеличить размер base_asset
ALTER TABLE fas_smart.trading_pairs 
    ALTER COLUMN base_asset TYPE VARCHAR(20);
