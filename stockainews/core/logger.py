"""日志配置模块"""
from loguru import logger
import sys
from pathlib import Path
from .config import config


# 标志位，确保只初始化一次
_initialized = False


def _init_logger():
    """初始化全局日志配置（仅执行一次）"""
    global _initialized
    if _initialized:
        return
    
    level = config.log_level
    
    # 移除默认handler
    logger.remove()
    
    # 添加控制台输出
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
               "<level>{message}</level>",
        colorize=True,
    )
    
    # 创建日志目录
    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 添加文件输出
    logger.add(
        log_dir / "stockainews_{time:YYYY-MM-DD}.log",
        rotation=config.log_rotation,
        retention=config.log_retention,
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        encoding="utf-8",
    )
    
    # 添加错误日志单独输出
    logger.add(
        log_dir / "error_{time:YYYY-MM-DD}.log",
        rotation=config.log_rotation,
        retention=config.log_retention,
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}\n{exception}",
        encoding="utf-8",
    )
    
    _initialized = True
    logger.info(f"Logger initialized with level: {level}")


def setup_logger(name: str = None):
    """
    获取logger实例
    
    Args:
        name: 模块名称（通常传入__name__）
        
    Returns:
        logger实例
    """
    # 确保日志系统已初始化
    _init_logger()
    
    # Loguru的logger是全局单例，直接返回即可
    # name参数会自动在日志格式中的{name}字段显示
    return logger


# 初始化日志系统
_init_logger()

