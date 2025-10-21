"""
Logging configuration for Traffix system
"""
import logging
import logging.handlers
import os
from datetime import datetime
from pathlib import Path

from config import settings


def setup_logging():
    """Setup comprehensive logging for the Traffix system"""
    
    # Create logs directory
    log_dir = Path("./logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler for all logs
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "traffix.log",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(file_handler)
    
    # Error file handler
    error_handler = logging.handlers.RotatingFileHandler(
        log_dir / "traffix_errors.log",
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(error_handler)
    
    # Agent-specific loggers
    agent_loggers = [
        "traffix.data_collector",
        "traffix.analyzer", 
        "traffix.storyteller",
        "traffix.reporter",
        "traffix.quick_mode",
        "traffix.deep_mode",
        "traffix.ritis",
        "traffix.news",
        "traffix.weather",
        "traffix.social"
    ]
    
    for logger_name in agent_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        
        # Agent-specific file handler
        agent_file_handler = logging.handlers.RotatingFileHandler(
            log_dir / f"{logger_name.split('.')[-1]}.log",
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3
        )
        agent_file_handler.setLevel(logging.DEBUG)
        agent_file_handler.setFormatter(detailed_formatter)
        logger.addHandler(agent_file_handler)
    
    # Performance logging
    perf_logger = logging.getLogger("traffix.performance")
    perf_logger.setLevel(logging.INFO)
    
    perf_handler = logging.handlers.RotatingFileHandler(
        log_dir / "performance.log",
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3
    )
    perf_handler.setLevel(logging.INFO)
    perf_handler.setFormatter(detailed_formatter)
    perf_logger.addHandler(perf_handler)
    
    # API logging
    api_logger = logging.getLogger("traffix.api")
    api_logger.setLevel(logging.INFO)
    
    api_handler = logging.handlers.RotatingFileHandler(
        log_dir / "api.log",
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3
    )
    api_handler.setLevel(logging.INFO)
    api_handler.setFormatter(detailed_formatter)
    api_logger.addHandler(api_handler)
    
    # Log startup
    logger = logging.getLogger("traffix.main")
    logger.info("Logging system initialized")
    logger.info(f"Log level: {settings.log_level}")
    logger.info(f"Log directory: {log_dir.absolute()}")


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with proper configuration"""
    return logging.getLogger(f"traffix.{name}")


class PerformanceLogger:
    """Logger for performance metrics"""
    
    def __init__(self):
        self.logger = logging.getLogger("traffix.performance")
    
    def log_analysis_performance(self, location: str, mode: str, duration: float, 
                               data_sources: int, confidence: float):
        """Log analysis performance metrics"""
        self.logger.info(
            f"ANALYSIS_PERF|{location}|{mode}|{duration:.2f}s|sources:{data_sources}|conf:{confidence:.2f}"
        )
    
    def log_agent_performance(self, agent_type: str, task_id: str, duration: float, 
                            success: bool):
        """Log agent performance metrics"""
        status = "SUCCESS" if success else "FAILED"
        self.logger.info(
            f"AGENT_PERF|{agent_type}|{task_id}|{duration:.2f}s|{status}"
        )
    
    def log_data_collection_performance(self, source: str, location: str, 
                                      duration: float, records: int):
        """Log data collection performance metrics"""
        self.logger.info(
            f"DATA_COLLECTION_PERF|{source}|{location}|{duration:.2f}s|records:{records}"
        )


# Initialize logging when module is imported
setup_logging()
