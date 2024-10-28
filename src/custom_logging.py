from loguru import logger
import psutil
from functools import wraps

from utils.stdout_wrapper import SAFE_STDOUT


# Remove all default handlers
logger.remove()

# Add a new handler
logger.add(
    SAFE_STDOUT,
    format="<g>{time:MM-DD HH:mm:ss}</g> |<lvl>{level:^8}</lvl>| {file}:{line} | {message}",
    backtrace=True,
    diagnose=True,
)

# メモリ使用量を取得してログに出力するデコレータ
def log_memory_usage(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process()
        before_memory_info = process.memory_info()
        before_memory_usage_mb = before_memory_info.rss / (1024 * 1024)

        result = func(*args, **kwargs)

        after_memory_info = process.memory_info()
        after_memory_usage_mb = after_memory_info.rss / (1024 * 1024)

        memory_diff_mb = after_memory_usage_mb - before_memory_usage_mb

        logger.info(f"Function: {func.__name__}")
        logger.info(f"Memory Usage Before: {before_memory_usage_mb:.2f} MB")
        logger.info(f"Memory Usage After: {after_memory_usage_mb:.2f} MB")
        logger.info(f"Memory Usage Difference: {memory_diff_mb:.2f} MB")

        return result

    return wrapper