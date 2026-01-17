"""配置管理模块"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
from pathlib import Path


class Config(BaseSettings):
    """配置管理类"""
    
    model_config = SettingsConfigDict(
        # 查找项目根目录的.env文件（向上找到stockainews包的父目录）
        env_file=str(Path(__file__).parent.parent.parent / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # 项目基础配置
    project_name: str = "StockAiNews"
    version: str = "0.2.0"
    debug: bool = False
    
    # 数据库配置
    database_url: str = "postgresql://localhost/stockainews"
    database_pool_size: int = 10
    database_max_overflow: int = 20
    
    # Redis配置
    redis_url: str = "redis://localhost:6379/0"
    redis_enabled: bool = True
    cache_ttl: int = 3600  # 1小时
    
    # LLM API Keys
    doubao_api_key: Optional[str] = None
    doubao_base_url: str = "https://ark.cn-beijing.volces.com/api/v3"
    
    dashscope_api_key: Optional[str] = None
    
    openai_api_key: Optional[str] = None
    openai_base_url: str = "https://api.openai.com/v1"
    
    # LLM配置
    llm_provider: str = "doubao"  # doubao, dashscope, openai
    deep_think_llm: str = "doubao-pro-32k"
    quick_think_llm: str = "doubao-lite-4k"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 4096
    
    # 爬虫配置
    crawler_headless: bool = True
    crawler_timeout: int = 30000  # 毫秒
    crawler_max_retries: int = 3
    crawler_user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    
    # Agent配置
    max_analysis_rounds: int = 1
    enable_parallel_analysis: bool = True
    agent_timeout: int = 300  # 秒
    
    # 日志配置
    log_level: str = "INFO"
    log_dir: str = "logs"
    log_rotation: str = "00:00"  # 每天午夜轮转
    log_retention: str = "30 days"
    
    # API配置
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = False
    
    # 智兔API配置
    zhitu_api_key: Optional[str] = None
    zhitu_rate_limit: float = 0.2  # 秒，默认0.2秒（1分钟300次请求）
    zhitu_max_requests_per_minute: int = 300  # 每分钟最大请求数
    
    # 魔码云服API配置
    moma_api_key: Optional[str] = None
    moma_rate_limit: float = 0.2  # 秒，默认0.2秒（1分钟300次请求）
    moma_max_requests_per_minute: int = 300  # 每分钟最大请求数


# 全局配置实例
config = Config()

