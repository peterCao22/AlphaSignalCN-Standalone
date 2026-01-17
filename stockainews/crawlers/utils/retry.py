"""
重试工具模块

提供指数退避重试装饰器和工具函数。
"""
import asyncio
import time
from typing import Callable, TypeVar, Any, Optional
from functools import wraps
from stockainews.core.logger import logger

T = TypeVar('T')


def async_retry(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    retry_on: Optional[tuple] = None,
    retry_log: bool = True
):
    """
    异步函数重试装饰器（指数退避）
    
    Args:
        max_retries: 最大重试次数（不包括首次尝试）
        initial_delay: 初始延迟时间（秒）
        max_delay: 最大延迟时间（秒）
        exponential_base: 指数退避的底数
        retry_on: 需要重试的异常类型元组（如果为None，则重试所有异常）
        retry_log: 是否记录重试日志
    
    Returns:
        装饰后的函数
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    # 检查是否需要重试
                    if retry_on and not isinstance(e, retry_on):
                        # 不在重试列表中的异常，直接抛出
                        raise
                    
                    # 如果已经达到最大重试次数，抛出异常
                    if attempt >= max_retries:
                        if retry_log:
                            logger.error(
                                f"函数 {func.__name__} 在 {max_retries + 1} 次尝试后仍然失败: {e}",
                                exc_info=True
                            )
                        raise
                    
                    # 计算延迟时间（指数退避）
                    delay = min(
                        initial_delay * (exponential_base ** attempt),
                        max_delay
                    )
                    
                    if retry_log:
                        logger.warning(
                            f"函数 {func.__name__} 第 {attempt + 1} 次尝试失败: {e}，"
                            f"{delay:.2f}秒后重试（剩余 {max_retries - attempt} 次）"
                        )
                    
                    # 等待后重试
                    await asyncio.sleep(delay)
            
            # 理论上不会到达这里
            if last_exception:
                raise last_exception
            
        return wrapper
    return decorator


def sync_retry(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    retry_on: Optional[tuple] = None,
    retry_log: bool = True
):
    """
    同步函数重试装饰器（指数退避）
    
    Args:
        max_retries: 最大重试次数（不包括首次尝试）
        initial_delay: 初始延迟时间（秒）
        max_delay: 最大延迟时间（秒）
        exponential_base: 指数退避的底数
        retry_on: 需要重试的异常类型元组（如果为None，则重试所有异常）
        retry_log: 是否记录重试日志
    
    Returns:
        装饰后的函数
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    # 检查是否需要重试
                    if retry_on and not isinstance(e, retry_on):
                        # 不在重试列表中的异常，直接抛出
                        raise
                    
                    # 如果已经达到最大重试次数，抛出异常
                    if attempt >= max_retries:
                        if retry_log:
                            logger.error(
                                f"函数 {func.__name__} 在 {max_retries + 1} 次尝试后仍然失败: {e}",
                                exc_info=True
                            )
                        raise
                    
                    # 计算延迟时间（指数退避）
                    delay = min(
                        initial_delay * (exponential_base ** attempt),
                        max_delay
                    )
                    
                    if retry_log:
                        logger.warning(
                            f"函数 {func.__name__} 第 {attempt + 1} 次尝试失败: {e}，"
                            f"{delay:.2f}秒后重试（剩余 {max_retries - attempt} 次）"
                        )
                    
                    # 等待后重试
                    time.sleep(delay)
            
            # 理论上不会到达这里
            if last_exception:
                raise last_exception
            
        return wrapper
    return decorator

