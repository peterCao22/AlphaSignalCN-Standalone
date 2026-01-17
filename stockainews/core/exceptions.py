"""自定义异常类"""


class StockAiNewsError(Exception):
    """基础异常类"""
    
    def __init__(self, message: str, code: str = None, details: dict = None):
        self.message = message
        self.code = code or "UNKNOWN_ERROR"
        self.details = details or {}
        super().__init__(self.message)
    
    def __str__(self):
        return f"[{self.code}] {self.message}"


class CrawlerError(StockAiNewsError):
    """爬虫相关异常"""
    
    def __init__(self, message: str, url: str = None, **kwargs):
        super().__init__(message, code="CRAWLER_ERROR", **kwargs)
        self.url = url


class LLMError(StockAiNewsError):
    """LLM调用相关异常"""
    
    def __init__(self, message: str, provider: str = None, model: str = None, **kwargs):
        super().__init__(message, code="LLM_ERROR", **kwargs)
        self.provider = provider
        self.model = model


class DataValidationError(StockAiNewsError):
    """数据验证异常"""
    
    def __init__(self, message: str, field: str = None, value=None, **kwargs):
        super().__init__(message, code="VALIDATION_ERROR", **kwargs)
        self.field = field
        self.value = value


class ConfigError(StockAiNewsError):
    """配置错误异常"""
    
    def __init__(self, message: str, config_key: str = None, **kwargs):
        super().__init__(message, code="CONFIG_ERROR", **kwargs)
        self.config_key = config_key


class DatabaseError(StockAiNewsError):
    """数据库操作异常"""
    
    def __init__(self, message: str, operation: str = None, **kwargs):
        super().__init__(message, code="DATABASE_ERROR", **kwargs)
        self.operation = operation


class AgentError(StockAiNewsError):
    """智能体执行异常"""
    
    def __init__(self, message: str, agent_name: str = None, **kwargs):
        super().__init__(message, code="AGENT_ERROR", **kwargs)
        self.agent_name = agent_name

