"""
智兔数服API适配器

封装智兔数服API调用，提供统一的接口获取基本面数据。
包括公司基础信息、股东信息、资本运作等信息。

API文档：https://zhituapi.cn/hsstockapi.html
"""
import time
from typing import Optional, List, Dict, Any
from datetime import date, datetime, timedelta
import httpx

from stockainews.core.config import config
from stockainews.core.logger import setup_logger
from stockainews.core.exceptions import StockAiNewsError

logger = setup_logger(__name__)

class DailyQuotaExceededError(StockAiNewsError):
    """智兔/数据服务：当日配额超限"""


class ZhituAdapter:
    """智兔数服API适配器"""
    
    # API基础URL
    BASE_URL = "https://api.zhituapi.com"
    
    def __init__(self, api_key: Optional[str] = None, rate_limit: Optional[float] = None):
        """
        初始化智兔数服适配器
        
        Args:
            api_key: 智兔数服API密钥（token），如果为None则从配置读取
            rate_limit: API调用限流间隔（秒），如果为None则从配置读取，默认0.2秒（1分钟300次请求）
        """
        self.api_key = api_key or config.zhitu_api_key
        if not self.api_key:
            logger.warning(
                "ZHITU_API_KEY is not configured. "
                "Please set it in environment variables or .env file"
            )
            # 不抛出异常，允许在测试环境中使用Mock
        
        # 从配置读取限流参数，如果没有则使用默认值
        self.rate_limit = rate_limit if rate_limit is not None else config.zhitu_rate_limit
        self.last_call_time = 0.0
        
        # 时间窗口限流：记录每分钟的请求时间戳
        self.max_requests_per_minute = config.zhitu_max_requests_per_minute
        self.request_timestamps = []  # 存储最近一分钟内的请求时间戳
        
        self.client = httpx.Client(
            base_url=self.BASE_URL,
            timeout=30.0
        )
        
        logger.info("ZhituAdapter initialized")
    
    def _rate_limit(self) -> None:
        """
        API调用限流控制（基于时间窗口）
        
        确保每分钟不超过300次请求。
        使用滑动窗口算法：记录最近一分钟内的请求时间戳。
        """
        current_time = time.time()
        
        # 清理一分钟之前的请求时间戳
        cutoff_time = current_time - 60.0
        self.request_timestamps = [ts for ts in self.request_timestamps if ts > cutoff_time]
        
        # 如果当前分钟内的请求数已达到上限，等待到下一分钟
        if len(self.request_timestamps) >= self.max_requests_per_minute:
            # 计算需要等待的时间（到最早请求的时间戳+60秒）
            oldest_request_time = min(self.request_timestamps)
            wait_until = oldest_request_time + 60.0
            wait_time = wait_until - current_time
            
            if wait_time > 0:
                logger.warning(
                    f"Rate limit reached ({len(self.request_timestamps)}/{self.max_requests_per_minute} requests/min), "
                    f"waiting {wait_time:.2f}s"
                )
                time.sleep(wait_time)
                # 重新清理时间戳
                current_time = time.time()
                cutoff_time = current_time - 60.0
                self.request_timestamps = [ts for ts in self.request_timestamps if ts > cutoff_time]
        
        # 确保最小间隔（0.2秒）
        elapsed = current_time - self.last_call_time
        if elapsed < self.rate_limit:
            sleep_time = self.rate_limit - elapsed
            time.sleep(sleep_time)
            current_time = time.time()
        
        # 记录本次请求时间戳
        self.request_timestamps.append(current_time)
        self.last_call_time = current_time
    
    def _to_zhitu_symbol(self, stock_code: str) -> str:
        """
        将标准股票代码转换为智兔数服格式
        
        Args:
            stock_code: 标准股票代码（如'000001'或'600000'）
            
        Returns:
            智兔数服格式的股票代码（如'000001.SZ'或'600000.SH'）
        """
        # 如果已经是智兔格式，直接返回
        if '.' in stock_code:
            return stock_code
        
        # 根据股票代码前缀判断交易所
        if stock_code.startswith(('600', '601', '603', '605', '688', '689')):
            return f"{stock_code}.SH"  # 上交所（主板+科创板）
        elif stock_code.startswith(('000', '001', '002', '003', '300', '301')):
            return f"{stock_code}.SZ"  # 深交所（主板+创业板）
        elif stock_code.startswith(('8', '43', '92')):
            return f"{stock_code}.BJ"  # 北交所（30%涨停）
        else:
            logger.warning(f"Unknown exchange for stock code: {stock_code}, using SZ as default")
            return f"{stock_code}.SZ"
    
    def _make_request(
        self,
        endpoint: str,
        method: str = "GET",
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        retry_count: int = 3
    ) -> Any:
        """
        发送API请求（带重试机制）
        
        Args:
            endpoint: API端点路径
            method: HTTP方法（GET/POST）
            params: URL参数（会自动添加token）
            data: 请求体数据
            retry_count: 重试次数
            
        Returns:
            API响应数据（可能是字典或列表）
            
        Raises:
            Exception: 如果所有重试都失败
        """
        # 确保params存在并添加token
        if params is None:
            params = {}
        if self.api_key:
            params['token'] = self.api_key
        
        for attempt in range(retry_count):
            try:
                self._rate_limit()
                
                logger.debug(f"Making {method} request to {endpoint}, attempt {attempt + 1}/{retry_count}")
                
                if method.upper() == "GET":
                    response = self.client.get(endpoint, params=params)
                elif method.upper() == "POST":
                    response = self.client.post(endpoint, json=data, params=params)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                
                response.raise_for_status()
                result = response.json()
                
                # 检查是否有错误信息
                # API可能返回字典格式：{"code": 101, "msg": "..."} 或字符串格式："101:Licence证书当日请求已超限"
                result_str = str(result).lower() if isinstance(result, (dict, str)) else ""
                
                if isinstance(result, dict):
                    # 检查错误码和错误消息（支持多种字段名）
                    error_code = result.get('code') or result.get('error_code') or result.get('err_code')
                    error_msg = (
                        result.get('error') or 
                        result.get('message') or 
                        result.get('msg') or 
                        result.get('err_msg') or
                        ''
                    )
                    
                    # 检查是否是当日请求已超限错误（多种检测方式）
                    is_daily_quota_error = (
                        error_code == 101 or 
                        '101' in str(error_code) or 
                        '当日请求已超限' in str(error_msg) or 
                        'licence证书当日请求已超限' in result_str or
                        'daily limit' in result_str or
                        'daily quota' in result_str or
                        'quota exceeded' in result_str
                    )
                    
                    if is_daily_quota_error:
                        error_str = f"{error_code}:{error_msg}" if error_code else str(error_msg)
                        raise DailyQuotaExceededError(
                            f"智兔API当日请求已超限: {error_str}. "
                            f"请明天再试或联系API提供商增加配额。"
                        )
                    
                    # 如果有错误码或错误消息，但不是配额错误，则抛出异常
                    if error_code or error_msg:
                        error_str = f"{error_code}:{error_msg}" if error_code else str(error_msg)
                        logger.warning(f"API returned error: {error_str}")
                        
                        # 检查是否是认证错误
                        if 'token' in result_str or 'auth' in result_str or '认证' in result_str:
                            raise ValueError(f"API authentication error: {error_str}")
                        
                        # 其他错误，抛出通用异常
                        raise ValueError(f"API error: {error_str}")
                
                # 如果返回的是字符串格式的错误（如 "101:Licence证书当日请求已超限"）
                elif isinstance(result, str):
                    if '101' in result or '当日请求已超限' in result or 'licence证书当日请求已超限' in result.lower():
                        raise DailyQuotaExceededError(
                            f"智兔API当日请求已超限: {result}. "
                            f"请明天再试或联系API提供商增加配额。"
                        )
                    elif 'error' in result.lower() or 'fail' in result.lower():
                        raise ValueError(f"API error: {result}")
                
                logger.debug(f"Successfully fetched data from {endpoint}")
                return result
                
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:  # Rate limit
                    # 遇到429错误，等待更长时间（至少60秒，确保重置限流窗口）
                    wait_time = max(60.0, (2 ** attempt) * 10.0)  # 至少60秒，指数退避
                    logger.warning(
                        f"Rate limited (429), waiting {wait_time:.1f}s before retry. "
                        f"Clearing request history to reset rate limit window."
                    )
                    # 清空请求时间戳，重置限流窗口
                    self.request_timestamps = []
                    time.sleep(wait_time)
                    continue
                elif e.response.status_code >= 500:  # Server error
                    if attempt < retry_count - 1:
                        wait_time = (2 ** attempt) * 1.0
                        logger.warning(f"Server error {e.response.status_code}, retrying in {wait_time}s")
                        time.sleep(wait_time)
                        continue
                logger.error(f"HTTP error {e.response.status_code}: {e.response.text}")
                raise
            except Exception as e:
                if attempt < retry_count - 1:
                    wait_time = (2 ** attempt) * 1.0
                    logger.warning(f"Request failed: {e}, retrying in {wait_time}s")
                    time.sleep(wait_time)
                    continue
                logger.error(f"Failed to make request to {endpoint}: {e}", exc_info=True)
                raise
        
        raise Exception(f"Failed to make request to {endpoint} after {retry_count} attempts")
    
    def _normalize_stock_code(self, stock_code: str) -> str:
        """
        标准化股票代码
        
        Args:
            stock_code: 股票代码（如'000001'或'600000'）
            
        Returns:
            标准化后的股票代码（去除空格，统一格式）
        """
        return stock_code.strip()
    
    def _format_date(self, date_obj: Any) -> str:
        """
        格式化日期为API需要的格式（YYYYMMDD）
        
        Args:
            date_obj: 日期对象（date/datetime/str）
            
        Returns:
            格式化后的日期字符串（YYYYMMDD格式，如'20240101'）
        """
        if isinstance(date_obj, str):
            # 尝试解析不同格式
            for fmt in ['%Y%m%d', '%Y-%m-%d', '%Y/%m/%d']:
                try:
                    dt = datetime.strptime(date_obj, fmt)
                    return dt.strftime('%Y%m%d')
                except ValueError:
                    continue
            return date_obj
        elif isinstance(date_obj, (date, datetime)):
            return date_obj.strftime('%Y%m%d')
        else:
            return str(date_obj)
    
    # ==================== 公司基础信息获取方法 ====================
    
    def get_company_profile(self, stock_code: str) -> Dict[str, Any]:
        """
        获取公司简介
        
        API: /hs/gs/gsjj/{股票代码}
        返回字段：name(公司名称), ename(英文名称), market(交易所), ldate(上市日期),
                 sprice(股价), principal(法定代表人), rdate(成立日期), rprice(注册资本),
                 instype(行业类型), organ(企业性质)
        
        Args:
            stock_code: 股票代码（如'000001'）
            
        Returns:
            公司简介数据字典，包含：
            - stock_code: 股票代码
            - company_name: 公司名称（中文）
            - company_name_en: 公司名称（英文）
            - exchange: 交易所
            - listed_date: 上市日期
            - stock_price: 股价
            - legal_representative: 法定代表人
            - established_date: 成立日期
            - registered_capital: 注册资本
            - industry_type: 行业类型
            - enterprise_nature: 企业性质
        """
        stock_code = self._normalize_stock_code(stock_code)
        zhitu_symbol = self._to_zhitu_symbol(stock_code)
        
        try:
            endpoint = f"/hs/gs/gsjj/{stock_code}"
            response = self._make_request(endpoint)
            
            # 如果返回空对象或错误，返回基础结构
            if not isinstance(response, dict) or not response:
                logger.warning(f"Empty or invalid response for company profile: {stock_code}")
                return {
                    "stock_code": stock_code,
                    "zhitu_symbol": zhitu_symbol,
                    "raw_data": response if isinstance(response, dict) else {}
                }
            
            # 转换字段名（从API字段转为snake_case）
            result = {
                "stock_code": stock_code,
                "zhitu_symbol": zhitu_symbol,
                "company_name": response.get("name", ""),  # 公司名称（中文）
                "company_name_en": response.get("ename", ""),  # 公司名称（英文）
                "exchange": response.get("market", ""),  # 交易所
                "listed_date": response.get("ldate", ""),  # 上市日期
                "stock_price": response.get("sprice", ""),  # 股价
                "legal_representative": response.get("principal", ""),  # 法定代表人
                "established_date": response.get("rdate", ""),  # 成立日期
                "registered_capital": response.get("rprice", ""),  # 注册资本
                "industry_type": response.get("instype", ""),  # 行业类型
                "enterprise_nature": response.get("organ", ""),  # 企业性质
                # 保留原始数据
                "raw_data": response
            }
            
            logger.info(f"Successfully fetched company profile for {stock_code}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to fetch company profile for {stock_code}: {e}", exc_info=True)
            raise
    
    def get_stock_sectors(self, stock_code: str) -> Optional[str]:
        """
        获取股票所属板块（行业）信息
        
        API: /hs/gs/ssbk/{股票代码}
        返回格式：[
            {
                "keyword": "所属板块",
                "content": "食品饮料 上海板块 内贸流通 融资融券 小红书概念 AIGC概念 社口"
            }
        ]
        
        从content字段中提取第一个词作为行业名称（通常是行业，后面是概念板块等）
        
        Args:
            stock_code: 股票代码（如'603777'）
            
        Returns:
            行业名称（如'食品饮料'），如果无法获取则返回None
        """
        stock_code = self._normalize_stock_code(stock_code)
        
        try:
            endpoint = f"/hs/gs/ssbk/{stock_code}"
            response = self._make_request(endpoint)
            
            # API返回列表格式
            if not isinstance(response, list) or len(response) == 0:
                logger.debug(f"No sectors data returned for {stock_code}")
                return None
            
            # 查找"所属板块"关键字的数据
            for item in response:
                if isinstance(item, dict):
                    keyword = item.get("keyword", "")
                    content = item.get("content", "")
                    
                    if keyword == "所属板块" and content:
                        # content格式： "食品饮料 上海板块 内贸流通 融资融券 小红书概念 AIGC概念 社口"
                        # 提取第一个词作为行业名称
                        sectors = content.strip().split()
                        if sectors:
                            industry = sectors[0]
                            logger.info(f"Extracted industry from sectors API for {stock_code}: {industry}")
                            return industry
            
            logger.debug(f"No '所属板块' keyword found in response for {stock_code}")
            return None
            
        except Exception as e:
            logger.warning(f"Failed to fetch stock sectors for {stock_code}: {e}")
            return None
    
    def get_directors_info(self, stock_code: str) -> Dict[str, Any]:
        """
        获取历届董事会成员信息
        
        API: /hs/gs/ljds/{股票代码}
        
        Args:
            stock_code: 股票代码
            
        Returns:
            董事成员信息字典，包含：
            - stock_code: 股票代码
            - board_members: 历届董事会成员列表
        """
        stock_code = self._normalize_stock_code(stock_code)
        zhitu_symbol = self._to_zhitu_symbol(stock_code)
        
        try:
            endpoint = f"/hs/gs/ljds/{stock_code}"
            response = self._make_request(endpoint)
            
            # API返回格式需要根据实际响应调整
            if isinstance(response, list):
                board_members = response
            elif isinstance(response, dict):
                board_members = response.get("board_members", [])
            else:
                logger.warning(f"Unexpected response format for directors info: {stock_code}")
                board_members = []
            
            result = {
                "stock_code": stock_code,
                "zhitu_symbol": zhitu_symbol,
                "board_members": board_members,
                "senior_management": [],  # 高级管理人员可能需要其他接口
                "related_companies": [],  # 关联公司信息可能需要其他接口
            }
            
            logger.info(f"Successfully fetched {len(board_members)} board member records for {stock_code}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to fetch directors info for {stock_code}: {e}", exc_info=True)
            raise
    
    def get_management_changes(
        self,
        stock_code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        获取关键人员变动记录
        
        注意：智兔数服API文档中未找到专门的人员变动接口
        此方法作为占位符，后续可能需要通过其他数据源获取
        
        Args:
            stock_code: 股票代码
            start_date: 起始日期（格式'YYYYMMDD'或'YYYY-MM-DD'）
            end_date: 结束日期（格式'YYYYMMDD'或'YYYY-MM-DD'）
            
        Returns:
            人员变动记录列表（当前返回空列表）
        """
        stock_code = self._normalize_stock_code(stock_code)
        
        logger.warning(f"Management changes API not available in Zhitu API for {stock_code}")
        return []
    
    # ==================== 股东信息获取方法 ====================
    
    def get_top_shareholders(
        self,
        stock_code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        获取十大股东信息
        
        API: /hs/fin/topholder/{股票代码}
        返回字段：plrq(披露日期), jzrq(截止日期), gdmc(股东名称), gdlx(股东类型),
                 cgsl(持股数量), bdyy(变动原因), cgbl(持股比例), gfxz(股份性质), cgpm(持股排名)
        
        Args:
            stock_code: 股票代码（如'000001'）
            start_date: 起始日期（格式'YYYYMMDD'），可选
            end_date: 结束日期（格式'YYYYMMDD'），可选
            
        Returns:
            十大股东信息字典，包含：
            - stock_code: 股票代码
            - top_10_shareholders: 十大股东列表（按时间排序）
            - latest_report_date: 最新报告日期
        """
        stock_code = self._normalize_stock_code(stock_code)
        zhitu_symbol = self._to_zhitu_symbol(stock_code)
        
        try:
            endpoint = f"/hs/fin/topholder/{zhitu_symbol}"
            params = {}
            if start_date:
                params["st"] = self._format_date(start_date)
            if end_date:
                params["et"] = self._format_date(end_date)
            
            response = self._make_request(endpoint, params=params)
            
            # API返回列表格式
            if not isinstance(response, list):
                logger.warning(f"Unexpected response format for top shareholders: {stock_code}")
                return {
                    "stock_code": stock_code,
                    "zhitu_symbol": zhitu_symbol,
                    "top_10_shareholders": [],
                    "latest_report_date": None
                }
            
            # 转换字段名（从拼音缩写转为snake_case）
            shareholders = []
            latest_date = None
            for item in response:
                shareholder = {
                    "disclosure_date": item.get("plrq"),  # 披露日期
                    "report_date": item.get("jzrq"),  # 截止日期
                    "shareholder_name": item.get("gdmc"),  # 股东名称
                    "shareholder_type": item.get("gdlx"),  # 股东类型
                    "share_count": item.get("cgsl"),  # 持股数量
                    "change_reason": item.get("bdyy"),  # 变动原因
                    "share_ratio": item.get("cgbl"),  # 持股比例
                    "share_nature": item.get("gfxz"),  # 股份性质
                    "rank": item.get("cgpm"),  # 持股排名
                }
                shareholders.append(shareholder)
                
                # 记录最新日期
                report_date = item.get("jzrq")
                if report_date and (not latest_date or report_date > latest_date):
                    latest_date = report_date
            
            result = {
                "stock_code": stock_code,
                "zhitu_symbol": zhitu_symbol,
                "top_10_shareholders": shareholders,
                "latest_report_date": latest_date
            }
            
            logger.info(f"Successfully fetched {len(shareholders)} top shareholder records for {stock_code}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to fetch top shareholders for {stock_code}: {e}", exc_info=True)
            raise
    
    def get_top_circulating_shareholders(
        self,
        stock_code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        获取十大流通股东信息
        
        API: /hs/fin/flowholder/{股票代码}
        返回字段与十大股东类似
        
        Args:
            stock_code: 股票代码
            start_date: 起始日期（格式'YYYYMMDD'），可选
            end_date: 结束日期（格式'YYYYMMDD'），可选
            
        Returns:
            十大流通股东信息字典
        """
        stock_code = self._normalize_stock_code(stock_code)
        zhitu_symbol = self._to_zhitu_symbol(stock_code)
        
        try:
            endpoint = f"/hs/fin/flowholder/{zhitu_symbol}"
            params = {}
            if start_date:
                params["st"] = self._format_date(start_date)
            if end_date:
                params["et"] = self._format_date(end_date)
            
            response = self._make_request(endpoint, params=params)
            
            if not isinstance(response, list):
                logger.warning(f"Unexpected response format for top circulating shareholders: {stock_code}")
                return {
                    "stock_code": stock_code,
                    "zhitu_symbol": zhitu_symbol,
                    "top_10_circulating_shareholders": [],
                    "latest_report_date": None
                }
            
            # 转换字段名
            shareholders = []
            latest_date = None
            for item in response:
                shareholder = {
                    "disclosure_date": item.get("plrq"),
                    "report_date": item.get("jzrq"),
                    "shareholder_name": item.get("gdmc"),
                    "shareholder_type": item.get("gdlx"),
                    "share_count": item.get("cgsl"),
                    "change_reason": item.get("bdyy"),
                    "share_ratio": item.get("cgbl"),
                    "share_nature": item.get("gfxz"),
                    "rank": item.get("cgpm"),
                }
                shareholders.append(shareholder)
                
                report_date = item.get("jzrq")
                if report_date and (not latest_date or report_date > latest_date):
                    latest_date = report_date
            
            result = {
                "stock_code": stock_code,
                "zhitu_symbol": zhitu_symbol,
                "top_10_circulating_shareholders": shareholders,
                "latest_report_date": latest_date
            }
            
            logger.info(f"Successfully fetched {len(shareholders)} top circulating shareholder records for {stock_code}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to fetch top circulating shareholders for {stock_code}: {e}", exc_info=True)
            raise
    
    def get_shareholder_count(
        self,
        stock_code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        获取公司流通股东数据
        
        API: /hs/fin/flowholder/{股票代码}
        返回字段：plrq(披露日期), jzrq(截止日期), gdmc(股东名称), gdlx(股东类型),
                 cgsl(持股数量), bdyy(变动原因), cgbl(持股比例), gfxz(股份性质), cgpm(持股排名)
        
        Args:
            stock_code: 股票代码
            start_date: 起始日期（格式'YYYYMMDD'），可选
            end_date: 结束日期（格式'YYYYMMDD'），可选
            
        Returns:
            流通股东记录列表，按报告日期分组
        """
        stock_code = self._normalize_stock_code(stock_code)
        zhitu_symbol = self._to_zhitu_symbol(stock_code)
        
        try:
            endpoint = f"/hs/fin/flowholder/{zhitu_symbol}"
            params = {}
            if start_date:
                params["st"] = self._format_date(start_date)
            if end_date:
                params["et"] = self._format_date(end_date)
            
            response = self._make_request(endpoint, params=params)
            
            if not isinstance(response, list):
                logger.warning(f"Unexpected response format for shareholder count: {stock_code}")
                return []
            
            # 转换字段名并按报告日期分组
            records_dict = {}  # key: (plrq, jzrq), value: list of shareholders
            for item in response:
                plrq = item.get("plrq")  # 披露日期
                jzrq = item.get("jzrq")  # 截止日期
                
                # 使用(披露日期, 截止日期)作为唯一键
                key = (plrq, jzrq)
                if key not in records_dict:
                    records_dict[key] = []
                
                shareholder_record = {
                    "shareholder_name": item.get("gdmc"),  # 股东名称
                    "shareholder_type": item.get("gdlx"),  # 股东类型
                    "shareholding_amount": item.get("cgsl"),  # 持股数量
                    "change_reason": item.get("bdyy"),  # 变动原因
                    "shareholding_ratio": item.get("cgbl"),  # 持股比例
                    "share_nature": item.get("gfxz"),  # 股份性质
                    "shareholding_rank": item.get("cgpm"),  # 持股排名
                }
                records_dict[key].append(shareholder_record)
            
            # 转换为列表格式，每个报告期一条记录
            records = []
            for (plrq, jzrq), shareholders in records_dict.items():
                record = {
                    "disclosure_date": plrq,  # 披露日期
                    "report_date": jzrq,  # 截止日期
                    "shareholders": shareholders,  # 该报告期的所有流通股东列表
                    "shareholder_count": len(shareholders),  # 股东数量
                }
                records.append(record)
            
            # 按报告日期排序（最新的在前）
            records.sort(key=lambda x: x.get("report_date", ""), reverse=True)
            
            logger.info(f"Successfully fetched {len(records)} shareholder count records for {stock_code}")
            return records
            
        except Exception as e:
            logger.error(f"Failed to fetch shareholder count for {stock_code}: {e}", exc_info=True)
            raise
    
    def get_institutional_holdings(
        self,
        stock_code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        获取机构持股变化
        
        注意：此方法通过从十大股东数据中筛选机构类型股东来实现
        智兔数服API文档中未找到专门的机构持股接口
        
        Args:
            stock_code: 股票代码
            start_date: 起始日期
            end_date: 结束日期
            
        Returns:
            机构持股变化记录列表
        """
        stock_code = self._normalize_stock_code(stock_code)
        
        try:
            # 获取十大股东数据，筛选机构类型
            shareholders_data = self.get_top_shareholders(stock_code, start_date, end_date)
            shareholders = shareholders_data.get("top_10_shareholders", [])
            
            # 机构类型关键词（根据实际数据调整）
            institution_keywords = ["基金", "保险", "QFII", "社保", "信托", "资管", "银行"]
            
            institutional_holdings = []
            for shareholder in shareholders:
                shareholder_type = shareholder.get("shareholder_type", "")
                if any(keyword in shareholder_type for keyword in institution_keywords):
                    institutional_holdings.append({
                        "report_date": shareholder.get("report_date"),
                        "institution_type": shareholder_type,
                        "shareholder_name": shareholder.get("shareholder_name"),
                        "share_count": shareholder.get("share_count"),
                        "share_ratio": shareholder.get("share_ratio"),
                    })
            
            logger.info(f"Successfully extracted {len(institutional_holdings)} institutional holdings for {stock_code}")
            return institutional_holdings
            
        except Exception as e:
            logger.warning(f"Failed to fetch institutional holdings for {stock_code}: {e}")
            return []
    
    # ==================== 财务数据获取方法 ====================
    
    def get_financial_ratios(
        self,
        stock_code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        获取财务主要指标
        
        API: /hs/fin/ratios/{股票代码}
        返回大量财务指标字段（如每股收益、ROE、毛利率等）
        
        Args:
            stock_code: 股票代码
            start_date: 起始日期（格式'YYYYMMDD'），可选
            end_date: 结束日期（格式'YYYYMMDD'），可选
            
        Returns:
            财务指标记录列表
        """
        stock_code = self._normalize_stock_code(stock_code)
        zhitu_symbol = self._to_zhitu_symbol(stock_code)
        
        try:
            endpoint = f"/hs/fin/ratios/{zhitu_symbol}"
            params = {}
            if start_date:
                params["st"] = self._format_date(start_date)
            if end_date:
                params["et"] = self._format_date(end_date)
            
            response = self._make_request(endpoint, params=params)
            
            if not isinstance(response, list):
                logger.warning(f"Unexpected response format for financial ratios: {stock_code}")
                return []
            
            # 转换字段名（根据API文档完整映射所有字段）
            records = []
            for item in response:
                record = {
                    # 日期字段
                    "report_date": item.get("jzrq"),  # 截止日期
                    "disclosure_date": item.get("plrq"),  # 披露日期
                    # 每股指标
                    "capital_reserve_per_share": item.get("mgzbgjj"),  # 每股资本公积金
                    "eps_after_deduction": item.get("kfmgsy"),  # 扣非每股收益
                    "eps_operating_cash_flow": item.get("mgjyhdxjl"),  # 每股经营活动现金流量
                    "bps": item.get("mgjzc"),  # 每股净资产
                    "basic_eps": item.get("jbmgsy"),  # 基本每股收益
                    "diluted_eps": item.get("xsmgsy"),  # 稀释每股收益
                    "undistributed_profit_per_share": item.get("mgwfplr"),  # 每股未分配利润
                    # 盈利能力指标
                    "roe": item.get("jzcsyl"),  # 净资产收益率
                    "roe_weighted_avg": item.get("jqjzcsyl"),  # 加权净资产收益率
                    "roe_diluted": item.get("tbjzcsyl"),  # 摊薄净资产收益率
                    "roa_diluted": item.get("tbzzcsyl"),  # 摊薄总资产收益率
                    "gross_margin": item.get("mlv"),  # 毛利率
                    "net_margin": item.get("jlv"),  # 净利率
                    "tax_rate": item.get("sjslv"),  # 实际税率
                    "sales_margin_rate": item.get("xsmlv"),  # 销售毛利率
                    # 成长性指标
                    "operating_revenue_yoy": item.get("zyyrsrzz"),  # 主营收入同比增长
                    "net_profit_yoy": item.get("jlrzz"),  # 净利润同比增长
                    "net_profit_attributable_yoy": item.get("gsmgsyzzdjlrzz"),  # 归属于母公司所有者的净利润同比增长
                    "net_profit_after_deduction_yoy": item.get("kfjlrzz"),  # 扣非净利润同比增长
                    "operating_revenue_qoq": item.get("yyzsrgdhbzz"),  # 营业总收入滚动环比增长
                    "net_profit_qoq": item.get("sljlrjqhbzz"),  # 归属净利润滚动环比增长
                    "net_profit_after_deduction_qoq": item.get("kfjlrgdhbzz"),  # 扣非净利润滚动环比增长
                    # 其他指标
                    "receivables_to_revenue": item.get("yskyysr"),  # 预收款营业收入
                    "cash_to_revenue": item.get("xsxjlyysr"),  # 销售现金流营业收入
                    "asset_liability_ratio": item.get("zcfzl"),  # 资产负债比率
                    "inventory_turnover": item.get("chzzl"),  # 存货周转率
                }
                records.append(record)
            
            logger.info(f"Successfully fetched {len(records)} financial ratio records for {stock_code}")
            return records
            
        except Exception as e:
            logger.error(f"Failed to fetch financial ratios for {stock_code}: {e}", exc_info=True)
            raise
    
    def get_profit_statement(
        self,
        stock_code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        获取利润表数据
        
        API: /hs/fin/income/{股票代码}
        返回利润表各项数据（营业收入、净利润等）
        
        Args:
            stock_code: 股票代码
            start_date: 起始日期（格式'YYYYMMDD'），可选
            end_date: 结束日期（格式'YYYYMMDD'），可选
            
        Returns:
            利润表记录列表
        """
        stock_code = self._normalize_stock_code(stock_code)
        zhitu_symbol = self._to_zhitu_symbol(stock_code)
        
        try:
            endpoint = f"/hs/fin/income/{zhitu_symbol}"
            params = {}
            if start_date:
                params["st"] = self._format_date(start_date)
            if end_date:
                params["et"] = self._format_date(end_date)
            
            response = self._make_request(endpoint, params=params)
            
            if not isinstance(response, list):
                logger.warning(f"Unexpected response format for profit statement: {stock_code}")
                return []
            
            # 转换字段名为标准snake_case格式
            records = []
            for item in response:
                record = {
                    # 日期字段
                    "report_date": item.get("jzrq"),  # 截止日期
                    "disclosure_date": item.get("plrq"),  # 披露日期
                    # 营业收入相关
                    "operating_revenue": item.get("yysr"),  # 营业收入
                    "total_operating_revenue": item.get("yyzsr"),  # 营业总收入
                    "earned_premium": item.get("yzbf"),  # 已赚保费
                    "real_estate_revenue": item.get("fdczssr"),  # 房地产销售收入
                    # 营业成本相关
                    "total_operating_cost": item.get("yyzcb"),  # 营业总成本
                    "operating_cost": item.get("yycb"),  # 营业成本
                    "real_estate_cost": item.get("fdczscb"),  # 房地产销售成本
                    "rd_expenses": item.get("yffy"),  # 研发费用
                    "surrender_value": item.get("tbj"),  # 退保金
                    "net_claims_paid": item.get("pczjje"),  # 赔付支出净额
                    "insurance_reserve": item.get("tqbxhtzbjje"),  # 提取保险合同准备金净额
                    "policy_dividend_expense": item.get("bdhlzc"),  # 保单红利支出
                    "reinsurance_expense": item.get("fbfy"),  # 分保费用
                    "other_operating_cost": item.get("qtywcb"),  # 其他业务成本
                    # 营业税金及附加
                    "tax_and_surcharge": item.get("yysjjfj"),  # 营业税金及附加
                    # 期间费用
                    "sales_expenses": item.get("xsfy"),  # 销售费用
                    "management_expenses": item.get("glfy"),  # 管理费用
                    "financial_expenses": item.get("cwfy"),  # 财务费用
                    # 资产减值
                    "asset_impairment_loss": item.get("zcjzss"),  # 资产减值损失
                    # 投资收益相关
                    "investment_income": item.get("tzsy"),  # 投资收益
                    "equity_investment_income": item.get("lyqyhhhqydtzsy"),  # 联营企业和合营企业的投资收益
                    "fair_value_change_income": item.get("gyjzbdsy"),  # 公允价值变动收益
                    "futures_profit_loss": item.get("qhsy"),  # 期货损益
                    "custody_income": item.get("tgsy"),  # 托管收益
                    "exchange_gain": item.get("hdsy"),  # 汇兑收益
                    "non_current_asset_disposal_income": item.get("fldzcczsy"),  # 非流动资产处置收益
                    "other_income": item.get("qtywsr"),  # 其他业务收入
                    "other_profit": item.get("qtywlr"),  # 其他业务利润
                    "subsidy_income": item.get("btsr"),  # 补贴收入
                    "other_income_total": item.get("qtsy"),  # 其他收益
                    "merged_profit_before_merger": item.get("bhbfzhbqsljlr"),  # 被合并方在合并前实现净利润
                    "interest_income": item.get("lxsr"),  # 利息收入
                    "fee_commission_income": item.get("sxfjyjsr"),  # 手续费及佣金收入
                    "fee_commission_expense": item.get("sxfjyjzc"),  # 手续费及佣金支出
                    "interest_expense": item.get("lxzc"),  # 利息支出
                    # 利润相关
                    "operating_profit": item.get("yylr"),  # 营业利润
                    "non_operating_income": item.get("ywsr"),  # 营业外收入
                    "non_operating_expense": item.get("ywzc"),  # 营业外支出
                    "total_profit": item.get("lrze"),  # 利润总额
                    "income_tax_expense": item.get("sdsfy"),  # 所得税费用
                    "unconfirmed_investment_loss": item.get("wqrtzss"),  # 未确认投资损失
                    # 净利润相关
                    "net_profit": item.get("jlr"),  # 净利润
                    "net_profit_attributable": item.get("gsmgsyzzdjlr"),  # 归属于母公司所有者的净利润
                    "net_profit_after_deduction": item.get("jlrhfcjcx"),  # 净利润(扣除非经常性损益后)
                    "minority_interest": item.get("ssgdsy"),  # 少数股东损益
                    # 每股收益
                    "basic_eps": item.get("jbmgsy"),  # 基本每股收益
                    "diluted_eps": item.get("xsmgsy"),  # 稀释每股收益
                    # 综合收益
                    "total_comprehensive_income": item.get("zhsyz"),  # 综合收益总额
                    "minority_comprehensive_income": item.get("gsssgdzhsyz"),  # 归属于少数股东的综合收益总额
                }
                records.append(record)
            
            logger.info(f"Successfully fetched {len(records)} profit statement records for {stock_code}")
            return records
            
        except Exception as e:
            logger.error(f"Failed to fetch profit statement for {stock_code}: {e}", exc_info=True)
            raise
    
    # ==================== 资本运作信息获取方法 ====================
    
    def get_capital_operations(
        self,
        stock_code: str,
        years: int = 3
    ) -> List[Dict[str, Any]]:
        """
        获取近年增发信息
        
        API: /hs/gs/jnzf/{股票代码}
        按公告日期倒序返回
        
        Args:
            stock_code: 股票代码
            years: 查询年数（默认3年，API返回近年数据）
            
        Returns:
            增发记录列表，每条记录包含增发相关信息
        """
        stock_code = self._normalize_stock_code(stock_code)
        zhitu_symbol = self._to_zhitu_symbol(stock_code)
        
        try:
            endpoint = f"/hs/gs/jnzf/{stock_code}"
            response = self._make_request(endpoint)
            
            # API返回格式需要根据实际响应调整
            if isinstance(response, list):
                operations = response
            elif isinstance(response, dict):
                operations = response.get("operations", [])
            else:
                logger.warning(f"Unexpected response format for capital operations: {stock_code}")
                operations = []
            
            # 转换字段名（根据实际API响应调整）
            result = []
            for item in operations:
                operation = {
                    "stock_code": stock_code,
                    "zhitu_symbol": zhitu_symbol,
                    **item  # 直接展开，后续根据实际字段调整
                }
                result.append(operation)
            
            logger.info(f"Successfully fetched {len(result)} capital operation records for {stock_code}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to fetch capital operations for {stock_code}: {e}", exc_info=True)
            raise
    
    def get_share_unlock_plan(
        self,
        stock_code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        获取解禁限售信息
        
        API: /hs/gs/jjxs/{股票代码}
        按解禁日期倒序返回
        
        Args:
            stock_code: 股票代码
            start_date: 起始日期（可选，API可能不支持日期过滤）
            end_date: 结束日期（可选，API可能不支持日期过滤）
            
        Returns:
            解禁限售记录列表，每条记录包含解禁相关信息
        """
        stock_code = self._normalize_stock_code(stock_code)
        zhitu_symbol = self._to_zhitu_symbol(stock_code)
        
        try:
            endpoint = f"/hs/gs/jjxs/{stock_code}"
            # 注意：API文档未提到日期参数，先不传日期参数
            response = self._make_request(endpoint)
            
            # API返回格式需要根据实际响应调整
            if isinstance(response, list):
                unlocks = response
            elif isinstance(response, dict):
                unlocks = response.get("unlocks", [])
            else:
                logger.warning(f"Unexpected response format for share unlock plan: {stock_code}")
                unlocks = []
            
            # 转换字段名（根据实际API响应调整）
            result = []
            for item in unlocks:
                unlock = {
                    "stock_code": stock_code,
                    "zhitu_symbol": zhitu_symbol,
                    **item  # 直接展开，后续根据实际字段调整
                }
                # 如果提供了日期范围，在客户端过滤
                if start_date or end_date:
                    unlock_date = item.get("unlock_date") or item.get("jjrq") or item.get("date")
                    if unlock_date:
                        unlock_date_str = self._format_date(unlock_date)
                        if start_date and unlock_date_str < self._format_date(start_date):
                            continue
                        if end_date and unlock_date_str > self._format_date(end_date):
                            continue
                
                result.append(unlock)
            
            logger.info(f"Successfully fetched {len(result)} share unlock records for {stock_code}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to fetch share unlock plan for {stock_code}: {e}", exc_info=True)
            raise
    
    # ==================== 异动信号相关方法 ====================
    
    def get_limit_up_pool(self, trade_date: str) -> List[Dict[str, Any]]:
        """
        获取涨停股池
        
        API: /hs/pool/ztgc/{交易日期}
        根据日期（格式yyyy-MM-dd，从2019-11-28开始到现在的每个交易日）作为参数，
        得到每天的涨停股票列表，根据封板时间升序。
        
        更新频率：交易时间段每10分钟
        
        Args:
            trade_date: 交易日期（格式'YYYYMMDD'或'YYYY-MM-DD'）
            
        Returns:
            涨停股票列表，每条记录包含涨停相关信息（字段名根据实际API响应调整）
        """
        try:
            # 转换日期格式为 yyyy-MM-dd
            if len(trade_date) == 8:  # YYYYMMDD格式
                formatted_date = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:8]}"
            elif len(trade_date) == 10 and '-' in trade_date:  # YYYY-MM-DD格式
                formatted_date = trade_date
            else:
                raise ValueError(f"Invalid trade_date format: {trade_date}, expected YYYYMMDD or YYYY-MM-DD")
            
            endpoint = f"/hs/pool/ztgc/{formatted_date}"
            response = self._make_request(endpoint)
            
            # API返回格式需要根据实际响应调整
            if isinstance(response, list):
                limit_up_stocks = response
            elif isinstance(response, dict):
                limit_up_stocks = response.get("stocks", []) or response.get("data", [])
            else:
                logger.warning(f"Unexpected response format for limit up pool: {trade_date}")
                return []
            
            # 转换字段名（根据实际API响应调整）
            result = []
            for item in limit_up_stocks:
                stock = {
                    "trade_date": trade_date,
                    "formatted_date": formatted_date,
                    **item  # 直接展开，后续根据实际字段调整
                }
                result.append(stock)
            
            logger.info(f"Successfully fetched {len(result)} limit up stocks for {trade_date}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to fetch limit up pool for {trade_date}: {e}", exc_info=True)
            raise
    
    def get_realtime_market_data(self, stock_code: str) -> Dict[str, Any]:
        """
        获取实时行情数据
        
        API: /hs/real/ssjy/{stock_code}
        完整API URL示例：https://api.zhituapi.com/hs/real/ssjy/300078
        股票代码格式：直接使用6位股票代码（如'300078'），无需交易所后缀
        
        Args:
            stock_code: 股票代码（6位数字，如'300078'）
            
        Returns:
            实时行情数据字典，包含以下字段（snake_case格式）：
            - stock_code: 股票代码
            - amplitude_pct: 振幅（%），用于更新max_price_change_pct
            - turnover_rate_pct: 换手（%），用于更新turnover_rate
            - volume_ratio_pct: 量比（%），用于量能筛选
            - price_change_pct: 涨跌幅（%），用于收盘涨幅判断
            - high_price: 最高价（元）
            - low_price: 最低价（元）
            - open_price: 开盘价（元）
            - current_price: 当前价格（元）
            - pre_close_price: 昨日收盘价（元）
            - volume: 成交量（手）
            - turnover_amount: 成交额（元）
            - total_market_value: 总市值（元）
            - circulating_market_value: 流通市值（元）
            - pe_ratio: 市盈率（动态）
            - pb_ratio: 市净率
            - five_min_change_pct: 五分钟涨跌幅（%）
            - price_speed_pct: 涨速（%）
            - price_change_amount: 涨跌额（元）
            - change_pct_60d: 60日涨跌幅（%）
            - change_pct_ytd: 年初至今涨跌幅（%）
        """
        stock_code = self._normalize_stock_code(stock_code)
        
        # 确保股票代码是6位数字（去除可能的交易所后缀）
        if '.' in stock_code:
            stock_code = stock_code.split('.')[0]
        if len(stock_code) != 6 or not stock_code.isdigit():
            raise ValueError(f"Invalid stock code format: {stock_code}, expected 6-digit code")
        
        try:
            endpoint = f"/hs/real/ssjy/{stock_code}"
            response = self._make_request(endpoint)
            
            # API返回格式需要根据实际响应调整
            if not isinstance(response, dict):
                logger.warning(f"Unexpected response format for realtime market data: {stock_code}")
                return {
                    "stock_code": stock_code,
                    "raw_data": response if isinstance(response, dict) else {}
                }
            
            # 转换字段名（从API字段转为snake_case）
            result = {
                "stock_code": stock_code,
                # 核心字段（用于信号捕捉）
                "amplitude_pct": response.get("zf"),  # 振幅（%）
                "turnover_rate_pct": response.get("hs"),  # 换手（%）
                "volume_ratio_pct": response.get("lb"),  # 量比（%）
                "price_change_pct": response.get("pc"),  # 涨跌幅（%）
                # 价格字段（用于数据验证和补充）
                "high_price": response.get("h"),  # 最高价（元）
                "low_price": response.get("l"),  # 最低价（元）
                "open_price": response.get("o"),  # 开盘价（元）
                "current_price": response.get("p"),  # 当前价格（元）
                "pre_close_price": response.get("yc"),  # 昨日收盘价（元）
                # 成交数据字段（用于量能分析）
                "volume": response.get("v"),  # 成交量（手）
                "turnover_amount": response.get("cje"),  # 成交额（元）
                # 市值字段（用于估值分析）
                "total_market_value": response.get("sz"),  # 总市值（元）
                "circulating_market_value": response.get("lt"),  # 流通市值（元）
                # 估值指标字段（用于后续分析）
                "pe_ratio": response.get("pe"),  # 市盈率（动态）
                "pb_ratio": response.get("sjl"),  # 市净率
                # 其他指标字段（用于增强分析）
                "five_min_change_pct": response.get("fm"),  # 五分钟涨跌幅（%）
                "price_speed_pct": response.get("zs"),  # 涨速（%）
                "price_change_amount": response.get("ud"),  # 涨跌额（元）
                "change_pct_60d": response.get("zdf60"),  # 60日涨跌幅（%）
                "change_pct_ytd": response.get("zdfnc"),  # 年初至今涨跌幅（%）
                # 保留原始数据
                "raw_data": response
            }
            
            logger.info(f"Successfully fetched realtime market data for {stock_code}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to fetch realtime market data for {stock_code}: {e}", exc_info=True)
            raise
    
    def get_strong_stock_pool(self, trade_date: str) -> List[Dict[str, Any]]:
        """
        获取强势股池
        
        API: /hs/pool/qsgc/{date}
        完整API URL示例：https://api.zhituapi.com/hs/pool/qsgc/2025-11-18
        日期格式：yyyy-MM-dd（从2019-11-28开始到现在的每个交易日）
        日期参数转换：如果输入为YYYYMMDD格式（如'20251118'），需转换为yyyy-MM-dd格式（如'2025-11-18'）
        
        Args:
            trade_date: 交易日期（格式'YYYYMMDD'或'YYYY-MM-DD'）
            
        Returns:
            强势股票列表，每条记录包含以下字段（snake_case格式）：
            - stock_code: 股票代码
            - stock_name: 股票名称
            - current_price: 当前价格（元）
            - limit_up_price: 涨停价（元）
            - price_change_pct: 涨幅（%）
            - price_speed_pct: 涨速（%）
            - turnover_amount: 成交额（元）
            - volume_ratio: 量比
            - turnover_rate: 换手率（%）
            - circulating_market_value: 流通市值（元）
            - total_market_value: 总市值（元）
            - is_new_high: 是否新高（布尔值）
            - limit_up_statistics: 涨停统计（字符串，格式如"3天/2板"）
        """
        try:
            # 转换日期格式为 yyyy-MM-dd
            if len(trade_date) == 8:  # YYYYMMDD格式
                formatted_date = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:8]}"
            elif len(trade_date) == 10 and '-' in trade_date:  # YYYY-MM-DD格式
                formatted_date = trade_date
            else:
                raise ValueError(f"Invalid trade_date format: {trade_date}, expected YYYYMMDD or YYYY-MM-DD")
            
            endpoint = f"/hs/pool/qsgc/{formatted_date}"
            response = self._make_request(endpoint)
            
            # API返回格式需要根据实际响应调整
            if isinstance(response, list):
                strong_stocks = response
            elif isinstance(response, dict):
                strong_stocks = response.get("stocks", []) or response.get("data", [])
            else:
                logger.warning(f"Unexpected response format for strong stock pool: {trade_date}")
                return []
            
            # 转换字段名（从API字段转为snake_case）
            result = []
            for item in strong_stocks:
                stock = {
                    "trade_date": trade_date,
                    "formatted_date": formatted_date,
                    # 基础信息字段
                    "stock_code": item.get("dm"),  # 代码（股票代码）
                    "stock_name": item.get("mc"),  # 名称（股票名称）
                    # 价格字段
                    "current_price": item.get("p"),  # 价格（元）
                    "limit_up_price": item.get("ztp"),  # 涨停价（元）
                    # 涨跌幅字段
                    "price_change_pct": item.get("zf"),  # 涨幅（%）
                    "price_speed_pct": item.get("zs"),  # 涨速（%）
                    # 成交数据字段
                    "turnover_amount": item.get("cje"),  # 成交额（元）
                    "volume_ratio": item.get("lb"),  # 量比
                    "turnover_rate": item.get("hs"),  # 换手率（%）
                    # 市值字段
                    "circulating_market_value": item.get("lt"),  # 流通市值（元）
                    "total_market_value": item.get("zsz"),  # 总市值（元）
                    # 其他字段
                    "is_new_high": bool(item.get("nh", 0)),  # 是否新高（0：否，1：是）
                    "limit_up_statistics": item.get("tj"),  # 涨停统计（x天/y板）
                }
                result.append(stock)
            
            logger.info(f"Successfully fetched {len(result)} strong stocks for {trade_date}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to fetch strong stock pool for {trade_date}: {e}", exc_info=True)
            raise
    
    def get_stock_fundamentals(
        self,
        stock_code: str,
        include_historical: bool = True
    ) -> Dict[str, Any]:
        """
        获取个股基本面数据（整合所有基本面数据接口）
        
        整合以下数据源：
        - 上市公司详情：公司简介、解禁限售、近年增发、财务指标
        - 财务报表：利润表、财务主要指标、公司十大流通股东、公司股东数
        
        Args:
            stock_code: 股票代码（如'000001'）
            include_historical: 是否包含历史数据（如历史财务数据、历史股东信息等），默认True
        
        Returns:
            基本面数据字典，包含以下结构：
            {
                "stock_code": "000001",
                "company_profile": {...},  # 公司简介
                "financial_ratios": [...],  # 财务主要指标（最新或历史）
                "profit_statement": [...],  # 利润表（最新或历史）
                "top_shareholders": [...],  # 十大股东（最新或历史）
                "top_circulating_shareholders": [...],  # 十大流通股东（最新或历史）
                "shareholder_count": [...],  # 公司股东数（最新或历史）
                "capital_operations": [...],  # 近年增发
                "share_unlock_plan": [...],  # 解禁限售
                "directors_info": {...},  # 董事信息
                "institutional_holdings": [...],  # 机构持仓
            }
        """
        stock_code = self._normalize_stock_code(stock_code)
        zhitu_symbol = self._to_zhitu_symbol(stock_code)
        
        logger.info(f"开始获取 {stock_code} 的基本面数据...")
        
        fundamentals = {
            "stock_code": stock_code,
            "zhitu_symbol": zhitu_symbol,
            "data_collected_at": datetime.now().isoformat(),
        }
        
        try:
            # 1. 公司简介（基础信息）
            try:
                fundamentals["company_profile"] = self.get_company_profile(stock_code)
            except Exception as e:
                logger.warning(f"获取公司简介失败: {e}")
                fundamentals["company_profile"] = {}
            
            # 2. 财务主要指标（最新数据）
            try:
                if include_historical:
                    # 获取最近3年的财务指标
                    end_date = datetime.now().strftime("%Y%m%d")
                    start_date = (datetime.now() - timedelta(days=3*365)).strftime("%Y%m%d")
                    fundamentals["financial_ratios"] = self.get_financial_ratios(
                        stock_code, start_date=start_date, end_date=end_date
                    )
                else:
                    # 只获取最新数据（不指定日期范围，获取最新一条）
                    ratios = self.get_financial_ratios(stock_code)
                    fundamentals["financial_ratios"] = ratios[:1] if ratios else []
            except Exception as e:
                logger.warning(f"获取财务主要指标失败: {e}")
                fundamentals["financial_ratios"] = []
            
            # 3. 利润表（最新数据）
            try:
                if include_historical:
                    end_date = datetime.now().strftime("%Y%m%d")
                    start_date = (datetime.now() - timedelta(days=3*365)).strftime("%Y%m%d")
                    fundamentals["profit_statement"] = self.get_profit_statement(
                        stock_code, start_date=start_date, end_date=end_date
                    )
                else:
                    profit = self.get_profit_statement(stock_code)
                    fundamentals["profit_statement"] = profit[:1] if profit else []
            except Exception as e:
                logger.warning(f"获取利润表失败: {e}")
                fundamentals["profit_statement"] = []
            
            # 4. 十大股东（返回字典格式，包含top_10_shareholders列表）
            try:
                if include_historical:
                    end_date = datetime.now().strftime("%Y%m%d")
                    start_date = (datetime.now() - timedelta(days=2*365)).strftime("%Y%m%d")
                    fundamentals["top_shareholders"] = self.get_top_shareholders(
                        stock_code, start_date=start_date, end_date=end_date
                    )
                else:
                    shareholders_data = self.get_top_shareholders(stock_code)
                    # 只保留最新一条记录
                    if shareholders_data.get("top_10_shareholders"):
                        fundamentals["top_shareholders"] = {
                            **shareholders_data,
                            "top_10_shareholders": shareholders_data["top_10_shareholders"][:1]
                        }
                    else:
                        fundamentals["top_shareholders"] = shareholders_data
            except Exception as e:
                logger.warning(f"获取十大股东失败: {e}")
                fundamentals["top_shareholders"] = {"stock_code": stock_code, "top_10_shareholders": [], "latest_report_date": None}
            
            # 5. 十大流通股东（返回字典格式，包含top_10_circulating_shareholders列表）
            try:
                if include_historical:
                    end_date = datetime.now().strftime("%Y%m%d")
                    start_date = (datetime.now() - timedelta(days=2*365)).strftime("%Y%m%d")
                    fundamentals["top_circulating_shareholders"] = self.get_top_circulating_shareholders(
                        stock_code, start_date=start_date, end_date=end_date
                    )
                else:
                    circulating_data = self.get_top_circulating_shareholders(stock_code)
                    # 只保留最新一条记录
                    if circulating_data.get("top_10_circulating_shareholders"):
                        fundamentals["top_circulating_shareholders"] = {
                            **circulating_data,
                            "top_10_circulating_shareholders": circulating_data["top_10_circulating_shareholders"][:1]
                        }
                    else:
                        fundamentals["top_circulating_shareholders"] = circulating_data
            except Exception as e:
                logger.warning(f"获取十大流通股东失败: {e}")
                fundamentals["top_circulating_shareholders"] = {"stock_code": stock_code, "top_10_circulating_shareholders": [], "latest_report_date": None}
            
            # 6. 公司股东数（返回列表格式）
            try:
                if include_historical:
                    end_date = datetime.now().strftime("%Y%m%d")
                    start_date = (datetime.now() - timedelta(days=2*365)).strftime("%Y%m%d")
                    fundamentals["shareholder_count"] = self.get_shareholder_count(
                        stock_code, start_date=start_date, end_date=end_date
                    )
                else:
                    count = self.get_shareholder_count(stock_code)
                    fundamentals["shareholder_count"] = count[:1] if count else []
            except Exception as e:
                logger.warning(f"获取公司股东数失败: {e}")
                fundamentals["shareholder_count"] = []
            
            # 7. 近年增发
            try:
                fundamentals["capital_operations"] = self.get_capital_operations(stock_code)
            except Exception as e:
                logger.warning(f"获取近年增发失败: {e}")
                fundamentals["capital_operations"] = []
            
            # 8. 解禁限售
            try:
                fundamentals["share_unlock_plan"] = self.get_share_unlock_plan(stock_code)
            except Exception as e:
                logger.warning(f"获取解禁限售失败: {e}")
                fundamentals["share_unlock_plan"] = []
            
            # 9. 董事信息
            try:
                fundamentals["directors_info"] = self.get_directors_info(stock_code)
            except Exception as e:
                logger.warning(f"获取董事信息失败: {e}")
                fundamentals["directors_info"] = {}
            
            # 10. 机构持仓
            try:
                fundamentals["institutional_holdings"] = self.get_institutional_holdings(stock_code)
            except Exception as e:
                logger.warning(f"获取机构持仓失败: {e}")
                fundamentals["institutional_holdings"] = []
            
            logger.info(f"成功获取 {stock_code} 的基本面数据")
            return fundamentals
            
        except Exception as e:
            logger.error(f"获取 {stock_code} 基本面数据失败: {e}", exc_info=True)
            # 返回已收集的部分数据
            fundamentals["error"] = str(e)
            return fundamentals
    
    def get_history_kline(
        self,
        stock_code: str,
        period: str = "d",  # 分时级别：5, 15, 30, 60, d, w, m, y
        adjust: str = "f",  # 除权方式：n, f, b, fr, br
        start_date: Optional[str] = None,  # 格式：YYYYMMDD
        end_date: Optional[str] = None,    # 格式：YYYYMMDD
    ) -> List[Dict[str, Any]]:
        """
        获取历史K线数据
        
        API: /hs/history/{股票代码.市场}/{分时级别}/{除权方式}
        完整API URL示例：https://api.zhituapi.com/hs/history/000001.SZ/d/f?token=xxx&st=20240601&et=20250430
        
        Args:
            stock_code: 股票代码（如'000001'）
            period: 分时级别（默认'd'日线）
                - 5, 15, 30, 60: 分钟级别
                - d: 日线
                - w: 周线
                - m: 月线
                - y: 年线
            adjust: 除权方式（默认'f'前复权）
                - n: 不复权（分钟级无除权数据，必须使用n）
                - f: 前复权（推荐用于技术分析）
                - b: 后复权
                - fr: 等比前复权
                - br: 等比后复权
            start_date: 开始日期（格式：YYYYMMDD，如'20240601'）
            end_date: 结束日期（格式：YYYYMMDD，如'20250430'）
        
        Returns:
            K线数据列表，每条记录包含标准OHLCV字段
        """
        stock_code = self._normalize_stock_code(stock_code)
        zhitu_symbol = self._to_zhitu_symbol(stock_code)
        
        try:
            # 构建API路径：/hs/history/{股票代码.市场}/{分时级别}/{除权方式}
            endpoint = f"/hs/history/{zhitu_symbol}/{period}/{adjust}"
            
            # 构建查询参数
            params = {}
            if start_date:
                params["st"] = start_date
            if end_date:
                params["et"] = end_date
            
            response = self._make_request(endpoint, params=params)
            
            # API返回列表格式
            if not isinstance(response, list):
                logger.warning(f"Unexpected response format for history kline: {stock_code}")
                return []
            
            # 转换字段名（从智兔API字段转为标准OHLCV格式）
            kline_data = []
            for item in response:
                # 处理时间格式：智兔API返回的时间格式可能是 "yyyy-MM-dd" 或 "yyyy-MM-ddHH:mm:ss"
                trade_time = item.get("t", "")
                if len(trade_time) == 10:  # yyyy-MM-dd
                    date_str = trade_time
                elif len(trade_time) > 10:  # yyyy-MM-ddHH:mm:ss
                    date_str = trade_time[:10]  # 只取日期部分
                else:
                    logger.warning(f"Invalid time format: {trade_time}")
                    continue
                
                # 计算涨跌幅
                close_price = item.get("c")
                pre_close = item.get("pc")
                change_pct = None
                if close_price and pre_close and pre_close > 0:
                    change_pct = (close_price - pre_close) / pre_close * 100
                
                # 智兔API可能返回小写l或大写L，需要兼容处理
                low_price = item.get("L") or item.get("l")
                
                kline = {
                    "date": date_str,
                    "open": item.get("o"),  # 开盘价
                    "high": item.get("h"),  # 最高价
                    "low": low_price,        # 最低价（兼容L和l）
                    "close": close_price,  # 收盘价
                    "volume": item.get("v"),  # 成交量
                    "amount": item.get("a"),  # 成交额
                    "pre_close": pre_close,   # 前收盘价
                    "change_pct": change_pct, # 涨跌幅（%）
                    "is_suspended": bool(item.get("sf", 0)),  # 停牌标志（1=停牌，0=不停牌）
                }
                kline_data.append(kline)
            
            logger.info(f"Successfully fetched {len(kline_data)} kline records for {stock_code} (period={period}, adjust={adjust})")
            return kline_data
            
        except Exception as e:
            logger.error(f"Failed to fetch history kline for {stock_code}: {e}", exc_info=True)
            raise
    
    def get_history_macd(
        self,
        stock_code: str,
        period: str = "d",  # 分时级别
        adjust: str = "f",  # 除权方式
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None,  # 最新条数（如lt=10）
    ) -> List[Dict[str, Any]]:
        """
        获取历史分时MACD数据
        
        API: /hs/history/macd/{股票代码}/{分时级别}/{除权类型}
        
        Returns:
            MACD数据列表，每条记录包含：date, diff, dea, macd, ema12, ema26
        """
        stock_code = self._normalize_stock_code(stock_code)
        zhitu_symbol = self._to_zhitu_symbol(stock_code)
        
        try:
            endpoint = f"/hs/history/macd/{zhitu_symbol}/{period}/{adjust}"
            params = {}
            if start_date:
                params["st"] = start_date
            if end_date:
                params["et"] = end_date
            if limit:
                params["lt"] = limit
            
            response = self._make_request(endpoint, params=params)
            
            if not isinstance(response, list):
                logger.warning(f"Unexpected response format for history MACD: {stock_code}")
                return []
            
            macd_data = []
            for item in response:
                trade_time = item.get("t", "")
                if len(trade_time) >= 10:
                    date_str = trade_time[:10]
                else:
                    continue
                
                macd_data.append({
                    "date": date_str,
                    "diff": item.get("diff"),
                    "dea": item.get("dea"),
                    "macd": item.get("macd"),
                    "ema12": item.get("ema12"),
                    "ema26": item.get("ema26"),
                })
            
            logger.info(f"Successfully fetched {len(macd_data)} MACD records for {stock_code}")
            return macd_data
            
        except Exception as e:
            logger.error(f"Failed to fetch history MACD for {stock_code}: {e}", exc_info=True)
            raise
    
    def get_history_ma(
        self,
        stock_code: str,
        period: str = "d",
        adjust: str = "f",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        获取历史分时MA（均线）数据
        
        API: /hs/history/ma/{股票代码}/{分时级别}/{除权类型}
        
        Returns:
            MA数据列表，每条记录包含：date, ma3, ma5, ma10, ma15, ma20, ma30, ma60, ma120, ma200, ma250
        """
        stock_code = self._normalize_stock_code(stock_code)
        zhitu_symbol = self._to_zhitu_symbol(stock_code)
        
        try:
            endpoint = f"/hs/history/ma/{zhitu_symbol}/{period}/{adjust}"
            params = {}
            if start_date:
                params["st"] = start_date
            if end_date:
                params["et"] = end_date
            if limit:
                params["lt"] = limit
            
            response = self._make_request(endpoint, params=params)
            
            if not isinstance(response, list):
                logger.warning(f"Unexpected response format for history MA: {stock_code}")
                return []
            
            ma_data = []
            for item in response:
                trade_time = item.get("t", "")
                if len(trade_time) >= 10:
                    date_str = trade_time[:10]
                else:
                    continue
                
                ma_data.append({
                    "date": date_str,
                    "ma3": item.get("ma3"),
                    "ma5": item.get("ma5"),
                    "ma10": item.get("ma10"),
                    "ma15": item.get("ma15"),
                    "ma20": item.get("ma20"),
                    "ma30": item.get("ma30"),
                    "ma60": item.get("ma60"),
                    "ma120": item.get("ma120"),
                    "ma200": item.get("ma200"),
                    "ma250": item.get("ma250"),
                })
            
            logger.info(f"Successfully fetched {len(ma_data)} MA records for {stock_code}")
            return ma_data
            
        except Exception as e:
            logger.error(f"Failed to fetch history MA for {stock_code}: {e}", exc_info=True)
            raise
    
    def get_history_boll(
        self,
        stock_code: str,
        period: str = "d",
        adjust: str = "f",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        获取历史分时BOLL（布林带）数据
        
        API: /hs/history/boll/{股票代码}/{分时级别}/{除权类型}
        
        Returns:
            BOLL数据列表，每条记录包含：date, upper, lower, middle
        """
        stock_code = self._normalize_stock_code(stock_code)
        zhitu_symbol = self._to_zhitu_symbol(stock_code)
        
        try:
            endpoint = f"/hs/history/boll/{zhitu_symbol}/{period}/{adjust}"
            params = {}
            if start_date:
                params["st"] = start_date
            if end_date:
                params["et"] = end_date
            if limit:
                params["lt"] = limit
            
            response = self._make_request(endpoint, params=params)
            
            if not isinstance(response, list):
                logger.warning(f"Unexpected response format for history BOLL: {stock_code}")
                return []
            
            boll_data = []
            for item in response:
                trade_time = item.get("t", "")
                if len(trade_time) >= 10:
                    date_str = trade_time[:10]
                else:
                    continue
                
                boll_data.append({
                    "date": date_str,
                    "upper": item.get("u"),
                    "lower": item.get("d"),
                    "middle": item.get("m"),
                })
            
            logger.info(f"Successfully fetched {len(boll_data)} BOLL records for {stock_code}")
            return boll_data
            
        except Exception as e:
            logger.error(f"Failed to fetch history BOLL for {stock_code}: {e}", exc_info=True)
            raise
    
    def get_history_kdj(
        self,
        stock_code: str,
        period: str = "d",
        adjust: str = "f",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        获取历史分时KDJ数据
        
        API: /hs/history/kdj/{股票代码}/{分时级别}/{除权类型}
        
        Returns:
            KDJ数据列表，每条记录包含：date, k, d, j
        """
        stock_code = self._normalize_stock_code(stock_code)
        zhitu_symbol = self._to_zhitu_symbol(stock_code)
        
        try:
            endpoint = f"/hs/history/kdj/{zhitu_symbol}/{period}/{adjust}"
            params = {}
            if start_date:
                params["st"] = start_date
            if end_date:
                params["et"] = end_date
            if limit:
                params["lt"] = limit
            
            response = self._make_request(endpoint, params=params)
            
            if not isinstance(response, list):
                logger.warning(f"Unexpected response format for history KDJ: {stock_code}")
                return []
            
            kdj_data = []
            for item in response:
                trade_time = item.get("t", "")
                if len(trade_time) >= 10:
                    date_str = trade_time[:10]
                else:
                    continue
                
                kdj_data.append({
                    "date": date_str,
                    "k": item.get("k"),
                    "d": item.get("d"),
                    "j": item.get("j"),
                })
            
            logger.info(f"Successfully fetched {len(kdj_data)} KDJ records for {stock_code}")
            return kdj_data
            
        except Exception as e:
            logger.error(f"Failed to fetch history KDJ for {stock_code}: {e}", exc_info=True)
            raise
    
    def batch_get_stock_fundamentals(
        self,
        stock_codes: List[str],
        include_historical: bool = True,
        max_concurrent: int = 5
    ) -> Dict[str, Dict[str, Any]]:
        """
        批量获取多只股票的基本面数据
        
        Args:
            stock_codes: 股票代码列表（如['000001', '600000']）
            include_historical: 是否包含历史数据，默认True
            max_concurrent: 最大并发数（由于API限流，建议不超过5），默认5
        
        Returns:
            Dict: 以股票代码为key的字典，value为基本面数据
                {
                    "000001": {...基本面数据...},
                    "600000": {...基本面数据...},
                    ...
                }
        """
        import concurrent.futures
        from threading import Lock
        
        results = {}
        results_lock = Lock()
        errors = []
        
        def fetch_single_stock(stock_code: str):
            """获取单只股票的基本面数据"""
            try:
                fundamentals = self.get_stock_fundamentals(
                    stock_code=stock_code,
                    include_historical=include_historical
                )
                with results_lock:
                    results[stock_code] = fundamentals
                logger.debug(f"成功获取 {stock_code} 的基本面数据")
            except Exception as e:
                error_msg = f"获取 {stock_code} 基本面数据失败: {str(e)}"
                logger.error(error_msg, exc_info=True)
                with results_lock:
                    errors.append(error_msg)
                    results[stock_code] = None
        
        # 使用线程池进行并发请求（注意：由于API限流，并发数不宜过高）
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            futures = [executor.submit(fetch_single_stock, code) for code in stock_codes]
            concurrent.futures.wait(futures)
        
        if errors:
            logger.warning(f"批量获取基本面数据完成，成功: {len(results) - len(errors)}, 失败: {len(errors)}")
        else:
            logger.info(f"批量获取基本面数据完成，成功: {len(results)}")
        
        return results
    
    def batch_get_strong_stock_pools(
        self,
        trade_dates: List[str],
        max_concurrent: int = 3
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        批量获取多个交易日的强势股池数据
        
        Args:
            trade_dates: 交易日期列表（格式'YYYYMMDD'或'YYYY-MM-DD'）
            max_concurrent: 最大并发数（由于API限流，建议不超过3），默认3
        
        Returns:
            Dict: 以交易日期为key的字典，value为强势股池列表
                {
                    "20250101": [...强势股票列表...],
                    "20250102": [...强势股票列表...],
                    ...
                }
        """
        import concurrent.futures
        from threading import Lock
        
        results = {}
        results_lock = Lock()
        errors = []
        
        def fetch_single_date(trade_date: str):
            """获取单个交易日的强势股池数据"""
            try:
                pool_data = self.get_strong_stock_pool(trade_date=trade_date)
                with results_lock:
                    results[trade_date] = pool_data
                logger.debug(f"成功获取 {trade_date} 的强势股池数据，共{len(pool_data)}条")
            except Exception as e:
                error_msg = f"获取 {trade_date} 强势股池数据失败: {str(e)}"
                logger.error(error_msg, exc_info=True)
                with results_lock:
                    errors.append(error_msg)
                    results[trade_date] = []
        
        # 使用线程池进行并发请求（注意：由于API限流，并发数不宜过高）
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            futures = [executor.submit(fetch_single_date, date) for date in trade_dates]
            concurrent.futures.wait(futures)
        
        if errors:
            logger.warning(f"批量获取强势股池数据完成，成功: {len(results) - len(errors)}, 失败: {len(errors)}")
        else:
            logger.info(f"批量获取强势股池数据完成，成功: {len(results)}")
        
        return results
    
    def _validate_and_clean_data(
        self,
        data: Any,
        data_type: str = "generic"
    ) -> Any:
        """
        数据验证和清洗（通用方法）
        
        Args:
            data: 原始数据（可能是字典、列表或其他类型）
            data_type: 数据类型（用于特定验证规则）
        
        Returns:
            清洗后的数据
        """
        if data is None:
            return None
        
        if isinstance(data, dict):
            return self._clean_dict(data, data_type)
        elif isinstance(data, list):
            return [self._validate_and_clean_data(item, data_type) for item in data]
        elif isinstance(data, (int, float)):
            # 验证数值范围
            if isinstance(data, float) and (data == float('inf') or data == float('-inf') or data != data):
                return None  # 处理无穷大和NaN
            return data
        elif isinstance(data, str):
            # 清洗字符串
            cleaned = data.strip()
            if cleaned == "" or cleaned.lower() in ["null", "none", "n/a", "-", "--"]:
                return None
            return cleaned
        else:
            return data
    
    def _clean_dict(self, data: Dict[str, Any], data_type: str) -> Dict[str, Any]:
        """
        清洗字典数据
        
        Args:
            data: 原始字典数据
            data_type: 数据类型
        
        Returns:
            清洗后的字典
        """
        cleaned = {}
        
        for key, value in data.items():
            # 转换key为snake_case（如果还不是）
            snake_key = self._to_snake_case(key)
            
            # 清洗value
            cleaned_value = self._validate_and_clean_data(value, data_type)
            
            # 只保留非None的值（可选：也可以保留None值）
            if cleaned_value is not None:
                cleaned[snake_key] = cleaned_value
        
        return cleaned
    
    def _to_snake_case(self, name: str) -> str:
        """
        将字段名转换为snake_case格式
        
        Args:
            name: 原始字段名（可能是camelCase、PascalCase等）
        
        Returns:
            snake_case格式的字段名
        """
        import re
        
        # 如果已经是snake_case，直接返回
        if '_' in name and name.islower():
            return name
        
        # 处理camelCase和PascalCase
        # 在大写字母前插入下划线，然后全部转为小写
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1)
        return s2.lower()
    
    def _validate_stock_code(self, stock_code: str) -> bool:
        """
        验证股票代码格式
        
        Args:
            stock_code: 股票代码
        
        Returns:
            如果格式正确返回True，否则返回False
        """
        if not stock_code or not isinstance(stock_code, str):
            return False
        
        # 标准化股票代码
        normalized = self._normalize_stock_code(stock_code)
        
        # 检查是否为6位数字
        if len(normalized) == 6 and normalized.isdigit():
            return True
        
        # 检查是否为智兔格式（如"000001.SZ"）
        if '.' in normalized:
            parts = normalized.split('.')
            if len(parts) == 2 and len(parts[0]) == 6 and parts[0].isdigit():
                return True
        
        return False
    
    def _validate_date(self, date_str: str) -> bool:
        """
        验证日期格式
        
        Args:
            date_str: 日期字符串
        
        Returns:
            如果格式正确返回True，否则返回False
        """
        if not date_str or not isinstance(date_str, str):
            return False
        
        # 检查YYYYMMDD格式
        if len(date_str) == 8 and date_str.isdigit():
            try:
                year = int(date_str[:4])
                month = int(date_str[4:6])
                day = int(date_str[6:8])
                if 1900 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31:
                    return True
            except ValueError:
                pass
        
        # 检查YYYY-MM-DD格式
        if len(date_str) == 10 and '-' in date_str:
            parts = date_str.split('-')
            if len(parts) == 3:
                try:
                    year = int(parts[0])
                    month = int(parts[1])
                    day = int(parts[2])
                    if 1900 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31:
                        return True
                except ValueError:
                    pass
        
        return False
    
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'client'):
            try:
                self.client.close()
            except Exception:
                pass

