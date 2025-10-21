"""Error handling utilities for TRAFFIX"""
import logging
import functools
from typing import Any, Callable, Optional, TypeVar, cast

# Set up logger
logger = logging.getLogger("traffix.error_handler")


class DataCollectionError(Exception):
    """Exception raised for errors in data collection"""
    pass


class APIError(Exception):
    """Exception raised for API-related errors"""
    pass


class ProcessingError(Exception):
    """Exception raised for data processing errors"""
    pass


F = TypeVar('F', bound=Callable[..., Any])


def handle_errors(
    default_return: Any = None,
    log_error: bool = True,
    raise_on_error: bool = False
) -> Callable[[F], F]:
    """
    Decorator to handle errors in functions
    
    Args:
        default_return: Value to return if an error occurs
        log_error: Whether to log the error
        raise_on_error: Whether to re-raise the error after logging
    
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if log_error:
                    logger.error(
                        f"Error in {func.__name__}: {str(e)}",
                        exc_info=True
                    )
                
                if raise_on_error:
                    raise
                
                return default_return
        
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_error:
                    logger.error(
                        f"Error in {func.__name__}: {str(e)}",
                        exc_info=True
                    )
                
                if raise_on_error:
                    raise
                
                return default_return
        
        # Determine if function is async
        if asyncio.iscoroutinefunction(func):
            return cast(F, async_wrapper)
        else:
            return cast(F, sync_wrapper)
    
    return decorator


import asyncio

