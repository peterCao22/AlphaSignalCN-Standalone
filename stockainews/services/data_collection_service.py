"""
统一数据收集服务

协调多个数据源（智兔API、东方财富、同花顺），实现数据收集编排和存储。
"""
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from stockainews.adapters.zhitu_adapter import ZhituAdapter
from stockainews.crawlers.eastmoney.research import EastMoneyResearchCrawler
from stockainews.crawlers.eastmoney.announcement import EastMoneyAnnouncementCrawler
from stockainews.crawlers.tonghuashun.hot_topics import TonghuashunHotTopicsCrawler
from stockainews.data.repositories.stock_signal_repository import StockSignalRepository
from stockainews.data.models.stock_signal import StockSignal
from stockainews.core.logger import setup_logger
from stockainews.core.exceptions import CrawlerError

logger = setup_logger(__name__)


class DataCollectionService:
    """统一数据收集服务"""
    
    def __init__(
        self,
        zhitu_adapter: Optional[ZhituAdapter] = None,
        research_crawler: Optional[EastMoneyResearchCrawler] = None,
        announcement_crawler: Optional[EastMoneyAnnouncementCrawler] = None,
        hot_topics_crawler: Optional[TonghuashunHotTopicsCrawler] = None
    ):
        """
        初始化数据收集服务
        
        Args:
            zhitu_adapter: 智兔API适配器（如果为None则自动创建）
            research_crawler: 东方财富研报爬虫（如果为None则自动创建）
            announcement_crawler: 东方财富公告爬虫（如果为None则自动创建）
            hot_topics_crawler: 同花顺热点爬虫（如果为None则自动创建）
        """
        self.zhitu_adapter = zhitu_adapter or ZhituAdapter()
        self.research_crawler = research_crawler or EastMoneyResearchCrawler()
        self.announcement_crawler = announcement_crawler or EastMoneyAnnouncementCrawler()
        self.hot_topics_crawler = hot_topics_crawler or TonghuashunHotTopicsCrawler()
        
        logger.info("数据收集服务初始化完成")
    
    async def collect_stock_data(
        self,
        stock_code: str,
        days: int = 90,
        include_research: bool = True,
        include_announcements: bool = True,
        include_hot_topics: bool = True,
        include_fundamentals: bool = True,
        include_kline: bool = False,
        include_technical_indicators: bool = False
    ) -> Dict[str, Any]:
        """
        收集单只股票的完整数据
        
        Args:
            stock_code: 股票代码（6位数字，如"000001"）
            days: 收集最近N天的数据（默认90天）
            include_research: 是否收集研报（默认True）
            include_announcements: 是否收集公告（默认True）
            include_hot_topics: 是否收集热点（默认True）
            include_fundamentals: 是否收集基本面数据（默认True）
            include_kline: 是否收集K线数据（默认False，因为数据量大）
            include_technical_indicators: 是否收集技术指标（默认False，需要K线数据）
        
        Returns:
            Dict: 包含所有收集到的数据
                - stock_code: 股票代码
                - stock_name: 股票名称（从智兔API获取）
                - zhitu_fundamentals: 智兔基本面数据
                - eastmoney_research_reports: 东方财富研报列表
                - eastmoney_announcements: 东方财富公告列表
                - hot_topics: 同花顺热点列表（与股票相关的）
                - recent_kline: 近期K线数据（如果include_kline=True）
                - technical_indicators: 技术指标数据（如果include_technical_indicators=True）
                - collect_time: 收集时间
                - errors: 收集过程中的错误列表
        """
        logger.info(f"开始收集股票数据: {stock_code}")
        
        result = {
            "stock_code": stock_code,
            "stock_name": None,
            "zhitu_fundamentals": None,
            "eastmoney_research_reports": [],
            "eastmoney_announcements": [],
            "hot_topics": [],
            "recent_kline": None,
            "technical_indicators": None,
            "collect_time": datetime.now().isoformat(),
            "errors": []
        }
        
        # 并行收集数据（使用asyncio.gather）
        tasks = []
        
        # 1. 收集智兔基本面数据
        if include_fundamentals:
            tasks.append(self._collect_zhitu_fundamentals(stock_code, result))
        
        # 2. 收集东方财富研报
        if include_research:
            tasks.append(self._collect_research_reports(stock_code, days, result))
        
        # 3. 收集东方财富公告
        if include_announcements:
            tasks.append(self._collect_announcements(stock_code, days, result))
        
        # 4. 收集同花顺热点（与股票相关的）
        if include_hot_topics:
            tasks.append(self._collect_hot_topics(stock_code, result))
        
        # 5. 收集K线数据（可选）
        if include_kline:
            tasks.append(self._collect_kline_data(stock_code, result))
        
        # 6. 收集技术指标（可选，需要K线数据）
        if include_technical_indicators:
            tasks.append(self._collect_technical_indicators(stock_code, result))
        
        # 等待所有任务完成
        await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info(f"股票数据收集完成: {stock_code}, 错误数: {len(result['errors'])}")
        
        return result
    
    async def _collect_zhitu_fundamentals(self, stock_code: str, result: Dict[str, Any]) -> None:
        """收集智兔基本面数据（带重试机制）"""
        max_retries = 2
        retry_delay = 1  # 秒
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"收集智兔基本面数据: {stock_code} (尝试 {attempt + 1}/{max_retries})")
                fundamentals = self.zhitu_adapter.get_stock_fundamentals(stock_code)
                result["zhitu_fundamentals"] = fundamentals
                
                # 从基本面数据中提取股票名称
                if fundamentals and "basic_info" in fundamentals:
                    basic_info = fundamentals.get("basic_info", {})
                    result["stock_name"] = basic_info.get("name") or basic_info.get("stock_name")
                
                logger.debug(f"智兔基本面数据收集完成: {stock_code}")
                return  # 成功则返回
            except Exception as e:
                error_msg = f"收集智兔基本面数据失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}"
                logger.warning(error_msg)
                
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    logger.info(f"重试收集智兔基本面数据: {stock_code}")
                else:
                    logger.error(f"收集智兔基本面数据最终失败: {stock_code}", exc_info=True)
                    result["errors"].append(error_msg)
                    result["zhitu_fundamentals"] = None
    
    async def _collect_research_reports(
        self,
        stock_code: str,
        days: int,
        result: Dict[str, Any]
    ) -> None:
        """收集东方财富研报"""
        try:
            logger.debug(f"收集东方财富研报: {stock_code}")
            reports = await self.research_crawler.crawl_research_reports(
                stock_code=stock_code,
                days=days
            )
            result["eastmoney_research_reports"] = reports
            logger.debug(f"东方财富研报收集完成: {stock_code}, 共{len(reports)}条")
        except Exception as e:
            error_msg = f"收集东方财富研报失败: {str(e)}"
            logger.error(error_msg, exc_info=True)
            result["errors"].append(error_msg)
    
    async def _collect_announcements(
        self,
        stock_code: str,
        days: int,
        result: Dict[str, Any]
    ) -> None:
        """收集东方财富公告"""
        try:
            logger.debug(f"收集东方财富公告: {stock_code}")
            announcements = await self.announcement_crawler.crawl_financial_announcements(
                stock_code=stock_code,
                days=days
            )
            result["eastmoney_announcements"] = announcements
            logger.debug(f"东方财富公告收集完成: {stock_code}, 共{len(announcements)}条")
        except Exception as e:
            error_msg = f"收集东方财富公告失败: {str(e)}"
            logger.error(error_msg, exc_info=True)
            result["errors"].append(error_msg)
    
    async def _collect_hot_topics(self, stock_code: str, result: Dict[str, Any]) -> None:
        """收集同花顺热点（与股票相关的，带重试机制）"""
        max_retries = 2
        retry_delay = 2  # 秒
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"收集同花顺热点: {stock_code} (尝试 {attempt + 1}/{max_retries})")
                related_topics = await self.hot_topics_crawler.get_hot_topics_with_stocks(stock_code)
                result["hot_topics"] = related_topics
                logger.debug(f"同花顺热点收集完成: {stock_code}, 共{len(related_topics)}条")
                return  # 成功则返回
            except Exception as e:
                error_msg = f"收集同花顺热点失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}"
                logger.warning(error_msg)
                
                if attempt < max_retries - 1:
                    # 等待后重试
                    await asyncio.sleep(retry_delay)
                    logger.info(f"重试收集同花顺热点: {stock_code}")
                else:
                    # 最后一次尝试失败，记录错误但不抛出异常
                    logger.error(f"收集同花顺热点最终失败: {stock_code}", exc_info=True)
                    result["errors"].append(error_msg)
                    result["hot_topics"] = []  # 设置为空列表，避免后续处理出错
    
    async def _collect_kline_data(self, stock_code: str, result: Dict[str, Any]) -> None:
        """收集K线数据（最近1年）"""
        try:
            logger.debug(f"收集K线数据: {stock_code}")
            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")
            
            kline_data = self.zhitu_adapter.get_history_kline(
                stock_code=stock_code,
                period="d",  # 日线
                adjust="f",  # 前复权
                start_date=start_date,
                end_date=end_date
            )
            result["recent_kline"] = {
                "data": kline_data,
                "count": len(kline_data) if kline_data else 0,
                "start_date": start_date,
                "end_date": end_date
            }
            logger.debug(f"K线数据收集完成: {stock_code}, 共{len(kline_data) if kline_data else 0}条")
        except Exception as e:
            error_msg = f"收集K线数据失败: {str(e)}"
            logger.error(error_msg, exc_info=True)
            result["errors"].append(error_msg)
    
    async def _collect_technical_indicators(self, stock_code: str, result: Dict[str, Any]) -> None:
        """收集技术指标数据"""
        try:
            logger.debug(f"收集技术指标数据: {stock_code}")
            
            # 计算日期范围（过去1年，用于计算技术指标）
            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")
            
            # 并行获取所有技术指标
            indicators = {}
            
            try:
                # MACD指标
                macd_data = self.zhitu_adapter.get_history_macd(
                    stock_code=stock_code,
                    period="d",
                    adjust="f",
                    start_date=start_date,
                    end_date=end_date
                )
                if macd_data:
                    indicators["macd"] = macd_data
            except Exception as e:
                logger.warning(f"获取MACD指标失败: {e}")
            
            try:
                # 均线指标
                ma_data = self.zhitu_adapter.get_history_ma(
                    stock_code=stock_code,
                    period="d",
                    adjust="f",
                    start_date=start_date,
                    end_date=end_date
                )
                if ma_data:
                    indicators["ma"] = ma_data
            except Exception as e:
                logger.warning(f"获取MA指标失败: {e}")
            
            try:
                # 布林带指标
                boll_data = self.zhitu_adapter.get_history_boll(
                    stock_code=stock_code,
                    period="d",
                    adjust="f",
                    start_date=start_date,
                    end_date=end_date
                )
                if boll_data:
                    indicators["boll"] = boll_data
            except Exception as e:
                logger.warning(f"获取BOLL指标失败: {e}")
            
            try:
                # KDJ指标
                kdj_data = self.zhitu_adapter.get_history_kdj(
                    stock_code=stock_code,
                    period="d",
                    adjust="f",
                    start_date=start_date,
                    end_date=end_date
                )
                if kdj_data:
                    indicators["kdj"] = kdj_data
            except Exception as e:
                logger.warning(f"获取KDJ指标失败: {e}")
            
            result["technical_indicators"] = indicators if indicators else None
            logger.debug(f"技术指标数据收集完成: {stock_code}, 指标数: {len(indicators)}")
        except Exception as e:
            error_msg = f"收集技术指标数据失败: {str(e)}"
            logger.error(error_msg, exc_info=True)
            result["errors"].append(error_msg)
    
    async def collect_strong_stocks_pool(
        self,
        pool_type: str = "strong",
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        收集强势股池数据
        
        Args:
            pool_type: 股池类型，"strong"（强势股池）或"limit_up"（涨跌停股池）
            limit: 限制返回数量（可选）
        
        Returns:
            List[Dict]: 强势股池数据列表
        """
        logger.info(f"开始收集{pool_type}股池数据")
        
        try:
            if pool_type == "strong":
                pool_data = self.zhitu_adapter.get_strong_stock_pool()
            elif pool_type == "limit_up":
                pool_data = self.zhitu_adapter.get_limit_up_pool()
            else:
                raise ValueError(f"不支持的股池类型: {pool_type}")
            
            if limit and pool_data:
                pool_data = pool_data[:limit]
            
            logger.info(f"{pool_type}股池数据收集完成, 共{len(pool_data) if pool_data else 0}条")
            return pool_data or []
            
        except Exception as e:
            logger.error(f"收集{pool_type}股池数据失败: {e}", exc_info=True)
            raise
    
    async def save_stock_signals(
        self,
        repository: StockSignalRepository,
        pool_data: List[Dict[str, Any]],
        signal_type: str = "strong_stock"
    ) -> int:
        """
        将股池数据保存到数据库
        
        Args:
            repository: StockSignalRepository实例
            pool_data: 股池数据列表（来自智兔API）
            signal_type: 信号类型（默认"strong_stock"）
        
        Returns:
            int: 成功保存的信号数量
        """
        logger.info(f"开始保存股票信号到数据库, 共{len(pool_data)}条")
        
        saved_count = 0
        
        for item in pool_data:
            try:
                # 使用StockSignal.from_zhitu_data()创建模型实例
                signal = StockSignal.from_zhitu_data(item, signal_type=signal_type)
                
                # 保存到数据库
                await repository.create(signal)
                saved_count += 1
                
            except Exception as e:
                logger.error(f"保存股票信号失败: {item.get('code', 'unknown')}, 错误: {e}", exc_info=True)
        
        logger.info(f"股票信号保存完成, 成功: {saved_count}/{len(pool_data)}")
        
        return saved_count
    
    async def collect_and_save_strong_stocks(
        self,
        repository: StockSignalRepository,
        pool_type: str = "strong",
        limit: Optional[int] = None,
        signal_type: str = "strong_stock"
    ) -> Dict[str, Any]:
        """
        收集强势股池数据并保存到数据库（一站式方法）
        
        Args:
            repository: StockSignalRepository实例
            pool_type: 股池类型，"strong"（强势股池）或"limit_up"（涨跌停股池）
            limit: 限制返回数量（可选）
            signal_type: 信号类型（默认"strong_stock"）
        
        Returns:
            Dict: 包含收集和保存的结果
                - collected_count: 收集到的数据数量
                - saved_count: 成功保存的数量
                - errors: 错误列表
        """
        result = {
            "collected_count": 0,
            "saved_count": 0,
            "errors": []
        }
        
        try:
            # 收集股池数据
            pool_data = await self.collect_strong_stocks_pool(pool_type=pool_type, limit=limit)
            result["collected_count"] = len(pool_data)
            
            # 保存到数据库
            if pool_data:
                saved_count = await self.save_stock_signals(
                    repository=repository,
                    pool_data=pool_data,
                    signal_type=signal_type
                )
                result["saved_count"] = saved_count
            
        except Exception as e:
            error_msg = f"收集并保存{pool_type}股池数据失败: {str(e)}"
            logger.error(error_msg, exc_info=True)
            result["errors"].append(error_msg)
        
        return result

