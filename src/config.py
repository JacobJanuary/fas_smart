"""
Configuration loader for FAS Smart.
Loads settings from .env file and environment variables.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Find .env file (search up from current dir)
_env_path = Path(__file__).parent.parent / '.env'
if _env_path.exists():
    load_dotenv(_env_path)
else:
    load_dotenv()  # Try default locations


class DBConfig:
    """Database connection configuration"""
    HOST: str = os.getenv('DB_HOST', 'localhost')
    PORT: int = int(os.getenv('DB_PORT', 5432))
    NAME: str = os.getenv('DB_NAME', 'fas_smart')
    USER: str = os.getenv('DB_USER', 'tradingbot')
    PASSWORD: str = os.getenv('DB_PASSWORD', '')
    
    @classmethod
    def get_dsn(cls) -> str:
        """Returns PostgreSQL DSN string"""
        return f"postgresql://{cls.USER}:{cls.PASSWORD}@{cls.HOST}:{cls.PORT}/{cls.NAME}"
    
    @classmethod
    def get_dict(cls) -> dict:
        """Returns connection dict for psycopg2"""
        return {
            'host': cls.HOST,
            'port': cls.PORT,
            'database': cls.NAME,
            'user': cls.USER,
            'password': cls.PASSWORD,
        }


class ThresholdConfig:
    """Trading pair thresholds"""
    ENTRY_VOLUME: float = float(os.getenv('ENTRY_VOLUME_THRESHOLD', 100_000))
    EXIT_VOLUME: float = float(os.getenv('EXIT_VOLUME_THRESHOLD', 30_000))
    MIN_DAYS_BEFORE_REMOVE: int = 14
    SIGNAL_GRACE_DAYS: int = 7
    
    # Tier boundaries
    TIER_1_VOLUME: float = 100_000_000  # >$100M
    TIER_2_VOLUME: float = 10_000_000   # >$10M


class IPv6Config:
    """IPv6 rotation configuration for API rate limit bypass"""
    # IPv6 prefix (e.g., "2001:db8::/64" -> generates random addresses in this range)
    PREFIX: str = os.getenv('IPV6_PREFIX', '')
    PREFIX_LENGTH: int = int(os.getenv('IPV6_PREFIX_LENGTH', '64'))
    ENABLED: bool = bool(os.getenv('IPV6_ENABLED', ''))


class Config:
    """Main configuration"""
    DB = DBConfig
    THRESHOLDS = ThresholdConfig
    IPV6 = IPv6Config
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    
    # Rolling window settings
    ROLLING_WINDOW_MINUTES: int = 15
    HISTORY_DAYS: int = 7


# Singleton
config = Config()
