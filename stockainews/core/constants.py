"""常量定义"""
from enum import Enum


class AnnouncementType(str, Enum):
    """公告类型"""
    PERFORMANCE = "performance"  # 业绩预告
    MAJOR_EVENT = "major_event"  # 重大事项
    SUSPENSION = "suspension"    # 停牌复牌
    DIVIDEND = "dividend"        # 分红派息
    FINANCING = "financing"      # 融资相关
    GOVERNANCE = "governance"    # 公司治理
    OTHER = "other"              # 其他


class ImpactLevel(str, Enum):
    """影响级别"""
    HIGH = "high"       # 高影响
    MEDIUM = "medium"   # 中等影响
    LOW = "low"         # 低影响


class Rating(str, Enum):
    """研报评级"""
    STRONG_BUY = "strong_buy"  # 强力买入
    BUY = "buy"                # 买入
    HOLD = "hold"              # 持有
    REDUCE = "reduce"          # 减持
    SELL = "sell"              # 卖出


class AnalysisStatus(str, Enum):
    """分析状态"""
    PENDING = "pending"        # 待处理
    PROCESSING = "processing"  # 处理中
    COMPLETED = "completed"    # 已完成
    FAILED = "failed"          # 失败


# 数据源常量
DATA_SOURCES = {
    "EASTMONEY_ANNOUNCEMENT": "东方财富公告",
    "EASTMONEY_RESEARCH": "东方财富研报",
    "ZHITU": "智兔数据",
}

# 缓存键前缀
CACHE_PREFIX = {
    "ANNOUNCEMENT": "announcement",
    "RESEARCH": "research",
    "ANALYSIS": "analysis",
    "STOCK_INFO": "stock_info",
}

# 时间相关常量
DEFAULT_DAYS = {
    "ANNOUNCEMENT": 30,  # 公告默认查询30天
    "RESEARCH": 90,      # 研报默认查询90天
}

# 爬虫相关常量
CRAWLER_CONFIG = {
    "MAX_RETRIES": 3,
    "TIMEOUT": 30000,  # 毫秒
    "WAIT_TIME": 2000,  # 毫秒
}

