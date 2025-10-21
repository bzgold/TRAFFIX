"""
Data Processors Package
Handles various data sources including RITIS, news, and other traffic data
"""

from .ritis_processor import RITISProcessor, get_ritis_processor

__all__ = [
    "RITISProcessor",
    "get_ritis_processor"
]
