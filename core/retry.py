"""
重试机制模块
功能：
- 指数退避重试
- 可配置重试次数、延迟
- 支持指定异常类型
- 异步支持
"""
import time
import asyncio
import logging
from functools import wraps
from typing import Type, Tuple, Union, Callable, Any, Optional

from config import settings

logger = logging.getLogger("retry")
logging.basicConfig(level=logging.INFO)


# 默认重试的异常类型
DEFAULT_RETRY_EXCEPTIONS = (
    Exception,
)


class RetryConfig:
    """重试配置"""
    def __init__(
        self,
        max_attempts: int = None,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
        jitter: bool = True,
        exceptions: Tuple[Type[Exception], ...] = None,
    ):
        self.max_attempts = max_attempts or settings.llm.max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter
        self.exceptions = exceptions or DEFAULT_RETRY_EXCEPTIONS


def calculate_delay(
    attempt: int,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
) -> float:
    """
    计算指数退避延迟
    :param attempt: 当前尝试次数（从0开始）
    :param initial_delay: 初始延迟（秒）
    :param max_delay: 最大延迟（秒）
    :param backoff_factor: 退避因子
    :param jitter: 是否添加抖动
    :return: 延迟秒数
    """
    delay = initial_delay * (backoff_factor ** attempt)
    delay = min(delay, max_delay)

    if jitter:
        import random
        # 添加±20%的抖动
        jitter_range = delay * 0.2
        delay += random.uniform(-jitter_range, jitter_range)
        delay = max(0, delay)

    return delay


def retry(
    config: Optional[RetryConfig] = None,
    max_attempts: int = None,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    exceptions: Tuple[Type[Exception], ...] = None,
    on_retry: Optional[Callable[[int, Exception, float], None]] = None,
):
    """
    重试装饰器（同步函数）
    :param config: 重试配置对象
    :param max_attempts: 最大尝试次数
    :param initial_delay: 初始延迟（秒）
    :param max_delay: 最大延迟（秒）
    :param backoff_factor: 退避因子
    :param jitter: 是否添加抖动
    :param exceptions: 需要重试的异常类型
    :param on_retry: 重试回调函数 (attempt, exception, delay)
    """
    if config is None:
        config = RetryConfig(
            max_attempts=max_attempts,
            initial_delay=initial_delay,
            max_delay=max_delay,
            backoff_factor=backoff_factor,
            jitter=jitter,
            exceptions=exceptions,
        )

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(config.max_attempts):
                try:
                    return func(*args, **kwargs)
                except config.exceptions as e:
                    last_exception = e

                    if attempt == config.max_attempts - 1:
                        # 最后一次尝试，不再重试
                        logger.warning(
                            f"[Retry] {func.__qualname__} 失败，已达最大重试次数 {config.max_attempts}"
                        )
                        raise

                    # 计算延迟
                    delay = calculate_delay(
                        attempt=attempt,
                        initial_delay=config.initial_delay,
                        max_delay=config.max_delay,
                        backoff_factor=config.backoff_factor,
                        jitter=config.jitter,
                    )

                    # 回调
                    if on_retry:
                        on_retry(attempt, e, delay)

                    logger.warning(
                        f"[Retry] {func.__qualname__} 失败 (attempt {attempt + 1}/{config.max_attempts}): "
                        f"{type(e).__name__} - {e}, {delay:.2f}s 后重试"
                    )

                    time.sleep(delay)

            raise last_exception

        return wrapper
    return decorator


def async_retry(
    config: Optional[RetryConfig] = None,
    max_attempts: int = None,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    exceptions: Tuple[Type[Exception], ...] = None,
    on_retry: Optional[Callable[[int, Exception, float], None]] = None,
):
    """
    重试装饰器（异步函数）
    参数同上
    """
    if config is None:
        config = RetryConfig(
            max_attempts=max_attempts,
            initial_delay=initial_delay,
            max_delay=max_delay,
            backoff_factor=backoff_factor,
            jitter=jitter,
            exceptions=exceptions,
        )

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(config.max_attempts):
                try:
                    return await func(*args, **kwargs)
                except config.exceptions as e:
                    last_exception = e

                    if attempt == config.max_attempts - 1:
                        logger.warning(
                            f"[AsyncRetry] {func.__qualname__} 失败，已达最大重试次数 {config.max_attempts}"
                        )
                        raise

                    delay = calculate_delay(
                        attempt=attempt,
                        initial_delay=config.initial_delay,
                        max_delay=config.max_delay,
                        backoff_factor=config.backoff_factor,
                        jitter=config.jitter,
                    )

                    if on_retry:
                        on_retry(attempt, e, delay)

                    logger.warning(
                        f"[AsyncRetry] {func.__qualname__} 失败 (attempt {attempt + 1}/{config.max_attempts}): "
                        f"{type(e).__name__} - {e}, {delay:.2f}s 后重试"
                    )

                    await asyncio.sleep(delay)

            raise last_exception

        return wrapper
    return decorator


# ========== 便捷装饰器 ==========
def llm_retry(func):
    """LLM调用专用重试装饰器"""
    config = RetryConfig(
        max_attempts=settings.llm.max_retries,
        initial_delay=1.0,
        max_delay=30.0,
        backoff_factor=2.0,
        jitter=True,
    )
    return retry(config=config)(func)


def async_llm_retry(func):
    """异步LLM调用专用重试装饰器"""
    config = RetryConfig(
        max_attempts=settings.llm.max_retries,
        initial_delay=1.0,
        max_delay=30.0,
        backoff_factor=2.0,
        jitter=True,
    )
    return async_retry(config=config)(func)


def skill_retry(func):
    """Skill调用专用重试装饰器"""
    config = RetryConfig(
        max_attempts=2,
        initial_delay=0.5,
        max_delay=5.0,
        backoff_factor=1.5,
        jitter=True,
    )
    return retry(config=config)(func)
