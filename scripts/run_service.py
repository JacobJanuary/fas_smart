#!/usr/bin/env python3
"""
FAS Smart Service Entry Point.

Usage:
    python scripts/run_service.py [--test] [--pairs N] [--duration S]

Options:
    --test       Run in test mode (limited pairs, verbose logging)
    --pairs N    Limit to N pairs (default: all)
    --duration S Run for S seconds then exit (default: infinite)
"""

import sys
import asyncio
import signal
import logging
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from service import FASService
from config import config


def setup_logging(debug: bool = False):
    """Configure logging to both console and file"""
    level = logging.DEBUG if debug else logging.INFO
    
    # Create logs directory
    log_dir = Path(__file__).parent.parent / 'logs'
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / 'fas_smart.log'
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    
    # File handler (always DEBUG level for troubleshooting)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Reduce noise from libraries
    logging.getLogger('websockets').setLevel(logging.WARNING)
    logging.getLogger('aiohttp').setLevel(logging.WARNING)
    
    logging.info(f"Logging to file: {log_file}")


async def run_with_timeout(service: FASService, duration: int = 0):
    """Run service with optional timeout"""
    try:
        await service.start()
        
        if duration > 0:
            logging.info(f"Running for {duration} seconds...")
            await asyncio.sleep(duration)
        else:
            # Run forever
            while True:
                await asyncio.sleep(60)
    finally:
        await service.stop()


def main():
    parser = argparse.ArgumentParser(description='FAS Smart Real-Time Signal Service')
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    parser.add_argument('--pairs', type=int, default=0, help='Limit number of pairs')
    parser.add_argument('--duration', type=int, default=0, help='Run duration in seconds')
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(debug=args.test)
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("FAS Smart Service Starting")
    logger.info(f"Database: {config.DB.HOST}:{config.DB.PORT}/{config.DB.NAME}")
    logger.info(f"Rolling Window: {config.ROLLING_WINDOW_MINUTES} minutes")
    logger.info("=" * 60)
    
    # Create service
    service = FASService()
    
    # Handle signals
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    def handle_signal(sig):
        logger.info(f"Received signal {sig.name}, shutting down...")
        loop.create_task(service.stop())
    
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda s=sig: handle_signal(s))
    
    try:
        loop.run_until_complete(run_with_timeout(service, args.duration))
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        loop.close()
        logger.info("Service shutdown complete")


if __name__ == "__main__":
    main()
