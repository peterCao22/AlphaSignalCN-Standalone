"""
代理IP池管理模块（简化版）

为爬虫提供代理IP支持。
注意：完整的代理池功能将在后续阶段实现。
"""
from dataclasses import dataclass
from typing import Optional, Dict
from datetime import datetime, timedelta
from enum import Enum

from stockainews.core.logger import setup_logger

logger = setup_logger(__name__)


class ProxyStatus(Enum):
    """代理IP状态"""
    ACTIVE = "active"
    EXPIRED = "expired"
    FAILED = "failed"
    TESTING = "testing"


@dataclass
class ProxyIP:
    """代理IP信息"""
    ip: str
    port: int
    username: Optional[str] = None
    password: Optional[str] = None
    created_at: datetime = None
    expires_at: datetime = None
    status: ProxyStatus = ProxyStatus.ACTIVE
    use_count: int = 0
    success_count: int = 0
    fail_count: int = 0
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.expires_at is None:
            # 默认60秒后过期
            self.expires_at = self.created_at + timedelta(seconds=60)
    
    @property
    def is_expired(self) -> bool:
        """检查IP是否已过期"""
        return datetime.now() >= self.expires_at
    
    @property
    def is_valid(self) -> bool:
        """检查IP是否有效"""
        return self.status == ProxyStatus.ACTIVE and not self.is_expired
    
    @property
    def proxy_url(self) -> str:
        """获取代理URL"""
        if self.username and self.password:
            return f"http://{self.username}:{self.password}@{self.ip}:{self.port}"
        return f"http://{self.ip}:{self.port}"
    
    @property
    def proxy_dict(self) -> Dict[str, str]:
        """获取代理字典"""
        if self.username and self.password:
            return {
                "http://": f"http://{self.username}:{self.password}@{self.ip}:{self.port}",
                "https://": f"http://{self.username}:{self.password}@{self.ip}:{self.port}"
            }
        return {
            "http://": f"http://{self.ip}:{self.port}",
            "https://": f"http://{self.ip}:{self.port}"
        }
    
    @property
    def remaining_seconds(self) -> float:
        """获取剩余有效时间（秒）"""
        if self.is_expired:
            return 0.0
        delta = self.expires_at - datetime.now()
        return delta.total_seconds()


class ProxyPoolManager:
    """
    代理IP池管理器（简化版）
    
    当前版本不使用代理池，始终返回None。
    完整的代理池功能将在后续阶段实现。
    """
    
    def __init__(self):
        # 简化版：代理始终禁用
        logger.info("ProxyPoolManager initialized (simplified version, proxy disabled)")
    
    @property
    def proxy_enabled(self) -> bool:
        """代理是否启用（简化版始终返回False）"""
        return False
    
    def enable_proxy(self):
        """启用代理（简化版不做任何操作）"""
        pass
    
    def disable_proxy(self):
        """禁用代理（简化版不做任何操作）"""
        pass
    
    def get_proxy(self, force_new: bool = False) -> Optional[ProxyIP]:
        """获取一个可用的代理IP（当前返回None）"""
        return None
    
    def return_proxy(self, proxy: ProxyIP, success: bool = True):
        """归还代理IP"""
        pass
    
    def mark_proxy_failed(self, proxy: ProxyIP):
        """标记代理IP失败"""
        pass
    
    def add_ips_to_whitelist(self, ips: list) -> bool:
        """添加IP到白名单（简化版不做任何操作）"""
        return False


# 全局代理池管理器实例
_proxy_pool_manager = None


def get_proxy_pool_manager() -> ProxyPoolManager:
    """获取全局代理池管理器实例"""
    global _proxy_pool_manager
    if _proxy_pool_manager is None:
        _proxy_pool_manager = ProxyPoolManager()
    return _proxy_pool_manager

