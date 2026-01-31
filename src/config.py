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
    
    @staticmethod
    def get_tier(volume_24h: float) -> str:
        """Determine liquidity tier based on 24h volume (FAS V2 parity)."""
        if volume_24h >= ThresholdConfig.TIER_1_VOLUME:
            return 'TIER_1'
        elif volume_24h >= ThresholdConfig.TIER_2_VOLUME:
            return 'TIER_2'
        else:
            return 'TIER_3'


class IPv6Config:
    """IPv6 rotation configuration for API rate limit bypass"""
    # IPv6 prefix (e.g., "2001:db8::/64" -> generates random addresses in this range)
    PREFIX: str = os.getenv('IPV6_PREFIX', '')
    PREFIX_LENGTH: int = int(os.getenv('IPV6_PREFIX_LENGTH', '64'))
    ENABLED: bool = bool(os.getenv('IPV6_ENABLED', ''))


class ProxyConfig:
    """Proxy list configuration - loads from proxy.txt file"""
    ENABLED: bool = bool(os.getenv('PROXY_ENABLED', ''))
    PROXY_FILE: str = os.path.join(os.path.dirname(__file__), '..', 'proxy.txt')
    
    _proxies: list = None
    
    @classmethod
    def _load_proxies(cls) -> list:
        """Load proxies from file (cached)."""
        if cls._proxies is not None:
            return cls._proxies
        
        cls._proxies = []
        try:
            proxy_path = os.path.abspath(cls.PROXY_FILE)
            with open(proxy_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    # Skip comments and empty lines
                    if line and not line.startswith('#'):
                        cls._proxies.append(line)
            print(f"Loaded {len(cls._proxies)} proxies from {proxy_path}")
        except FileNotFoundError:
            print(f"Warning: proxy.txt not found at {cls.PROXY_FILE}")
        return cls._proxies
    
    @classmethod
    def get_rotating_url(cls, session_id: str = None) -> str:
        """Get random proxy URL from list."""
        import random
        if not cls.ENABLED:
            return None
        
        proxies = cls._load_proxies()
        if not proxies:
            return None
        
        # Random proxy from list
        proxy = random.choice(proxies)
        
        # Format: ip:port -> http://ip:port
        return f"http://{proxy}"


class Config:
    """Main configuration"""
    DB = DBConfig
    THRESHOLDS = ThresholdConfig
    IPV6 = IPv6Config
    PROXY = ProxyConfig
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    
    # Rolling window settings
    ROLLING_WINDOW_MINUTES: int = 15
    HISTORY_DAYS: int = 7


# Singleton
config = Config()
