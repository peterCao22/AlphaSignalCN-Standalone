"""默认配置定义"""
import os
from pathlib import Path

# 获取项目根目录
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
CACHE_DIR = PROJECT_DIR / "cache"
LOGS_DIR = PROJECT_DIR / "logs"

DEFAULT_CONFIG = {
    # 项目路径配置
    "project_dir": str(PROJECT_DIR),
    "data_dir": str(DATA_DIR),
    "cache_dir": str(CACHE_DIR),
    "logs_dir": str(LOGS_DIR),
    
    # LLM设置
    # 支持的提供商: "doubao", "dashscope", "openai"
    "llm_provider": os.getenv("LLM_PROVIDER", "doubao"),
    "deep_think_llm": os.getenv("DEEP_THINK_LLM", "doubao-pro-32k"),
    "quick_think_llm": os.getenv("QUICK_THINK_LLM", "doubao-lite-4k"),
    "llm_temperature": float(os.getenv("LLM_TEMPERATURE", "0.7")),
    "llm_max_tokens": int(os.getenv("LLM_MAX_TOKENS", "4096")),
    
    # Agent设置
    "max_analysis_rounds": int(os.getenv("MAX_ANALYSIS_ROUNDS", "1")),
    "enable_parallel_analysis": os.getenv("ENABLE_PARALLEL_ANALYSIS", "true").lower() == "true",
    "agent_timeout": int(os.getenv("AGENT_TIMEOUT", "300")),  # 秒
    
    # 爬虫设置
    "crawler_headless": os.getenv("CRAWLER_HEADLESS", "true").lower() == "true",
    "crawler_timeout": int(os.getenv("CRAWLER_TIMEOUT", "30000")),  # 毫秒
    "crawler_max_retries": int(os.getenv("CRAWLER_MAX_RETRIES", "3")),
    "max_concurrent_crawlers": int(os.getenv("MAX_CONCURRENT_CRAWLERS", "3")),
    
    # 数据库设置
    "database_url": os.getenv("DATABASE_URL", "postgresql://localhost/stockainews"),
    "database_pool_size": int(os.getenv("DATABASE_POOL_SIZE", "10")),
    "database_max_overflow": int(os.getenv("DATABASE_MAX_OVERFLOW", "20")),
    
    # Redis设置
    "redis_url": os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    "redis_enabled": os.getenv("REDIS_ENABLED", "true").lower() == "true",
    "cache_ttl": int(os.getenv("CACHE_TTL", "3600")),  # 1小时
    
    # API设置
    "api_host": os.getenv("API_HOST", "0.0.0.0"),
    "api_port": int(os.getenv("API_PORT", "8000")),
    "api_reload": os.getenv("API_RELOAD", "false").lower() == "true",
    
    # 日志设置
    "log_level": os.getenv("LOG_LEVEL", "INFO"),
    "log_rotation": os.getenv("LOG_ROTATION", "00:00"),
    "log_retention": os.getenv("LOG_RETENTION", "30 days"),
    
    # 业务设置
    "announcement_days": int(os.getenv("ANNOUNCEMENT_DAYS", "30")),
    "research_days": int(os.getenv("RESEARCH_DAYS", "90")),
    "max_announcements": int(os.getenv("MAX_ANNOUNCEMENTS", "50")),
    "max_research_reports": int(os.getenv("MAX_RESEARCH_REPORTS", "30")),
}


def get_config(key: str, default=None):
    """
    获取配置值
    
    Args:
        key: 配置键
        default: 默认值
        
    Returns:
        配置值
    """
    return DEFAULT_CONFIG.get(key, default)


def update_config(updates: dict):
    """
    更新配置
    
    Args:
        updates: 要更新的配置字典
    """
    DEFAULT_CONFIG.update(updates)

