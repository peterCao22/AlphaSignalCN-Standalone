"""
Adapters模块

包含各种数据源适配器。
"""
from stockainews.adapters.zhitu_adapter import ZhituAdapter
from stockainews.adapters.moma_adapter import MomaAdapter

__all__ = ["ZhituAdapter", "MomaAdapter"]

