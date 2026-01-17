"""
魔码云服API适配器

封装魔码云服API调用，提供统一的接口获取股票数据。
包括K线数据、涨停数据、技术指标等。

API文档：https://www.momaapi.com/docs-shares.html
"""
import time
from typing import Optional, List, Dict, Any
from datetime import date, datetime, timedelta
import httpx

from stockainews.core.config import config
from stockainews.core.logger import setup_logger

logger = setup_logger(__name__)


class MomaAdapter:
    """魔码云服API适配器"""
    
    # API基础URL
    BASE_URL = "http://api.momaapi.com"
    
    def __init__(self, api_key: Optional[str] = None, rate_limit: Optional[float] = None):
        """
        初始化魔码云服适配器
        
        Args:
            api_key: 魔码云服API密钥（token），如果为None则从配置读取
            rate_limit: API调用限流间隔（秒），如果为None则从配置读取，默认0.2秒（1分钟300次请求）
        """
        self.api_key = api_key or getattr(config, 'moma_api_key', None)
        if not self.api_key:
            logger.warning(
                "MOMA_API_KEY is not configured. "
                "Please set it in environment variables or .env file"
            )
            # 不抛出异常，允许在测试环境中使用Mock
        
        # 从配置读取限流参数，如果没有则使用默认值
        self.rate_limit = rate_limit if rate_limit is not None else (getattr(config, 'moma_rate_limit', 0.2))
        self.last_call_time = 0.0
        
        # 时间窗口限流：记录每分钟的请求时间戳
        self.max_requests_per_minute = getattr(config, 'moma_max_requests_per_minute', 300)
        self.request_timestamps = []  # 存储最近一分钟内的请求时间戳
        
        self.client = httpx.Client(
            base_url=self.BASE_URL,
            timeout=30.0
        )
        
        logger.info("MomaAdapter initialized")
    
    def _rate_limit(self) -> None:
        """
        API调用限流控制（基于时间窗口）
        
        确保每分钟不超过300次请求（默认）。
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
    
    def _to_moma_symbol(self, stock_code: str) -> str:
        """
        将标准股票代码转换为魔码云服格式
        
        Args:
            stock_code: 标准股票代码（如'000001'或'600000'）
            
        Returns:
            魔码云服格式的股票代码（如'000001.SZ'或'600000.SH'）
        """
        # 如果已经是完整格式，直接返回
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
    
    def _normalize_stock_code(self, stock_code: str) -> str:
        """
        标准化股票代码（去除.SZ/.SH后缀）
        
        Args:
            stock_code: 股票代码（可能包含.SZ/.SH后缀）
            
        Returns:
            标准6位数字股票代码
        """
        if '.' in stock_code:
            return stock_code.split('.')[0]
        return stock_code
    
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
            endpoint: API端点路径（不包含Token，Token会自动添加到URL路径中）
            method: HTTP方法（GET/POST）
            params: URL参数
            data: 请求体数据
            retry_count: 重试次数
            
        Returns:
            API响应数据（可能是字典或列表）
            
        Raises:
            Exception: 如果所有重试都失败
        """
        if not self.api_key:
            raise ValueError("MOMA_API_KEY is required but not configured")
        
        # MomaAPI的Token是URL路径的一部分，不是查询参数
        # 例如：http://api.momaapi.com/hslt/list/您的Token
        # 所以endpoint应该已经包含Token，或者我们需要在endpoint末尾添加Token
        if not endpoint.endswith(self.api_key):
            # 如果endpoint不以Token结尾，添加Token
            if endpoint.endswith('/'):
                endpoint = f"{endpoint}{self.api_key}"
            else:
                endpoint = f"{endpoint}/{self.api_key}"
        
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
                if isinstance(result, dict):
                    # 检查是否有错误码
                    if 'code' in result and result.get('code') != 0:
                        error_msg = result.get('msg', 'Unknown error')
                        logger.error(f"API returned error: {error_msg}")
                        raise Exception(f"API error: {error_msg}")
                
                return result
                
            except httpx.HTTPStatusError as e:
                if e.response.status_code in (500, 502, 503, 504):
                    # 服务器错误，重试
                    if attempt < retry_count - 1:
                        wait_time = (attempt + 1) * 1.0  # 递增等待时间
                        logger.warning(f"Server error {e.response.status_code}, retrying in {wait_time:.1f}s")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"HTTP error {e.response.status_code}: {e.response.text}")
                        raise
                else:
                    # 其他HTTP错误，不重试
                    logger.error(f"HTTP error {e.response.status_code}: {e.response.text}")
                    raise
                    
            except Exception as e:
                if attempt < retry_count - 1:
                    wait_time = (attempt + 1) * 1.0
                    logger.warning(f"Request failed: {e}, retrying in {wait_time:.1f}s")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Request failed after {retry_count} attempts: {e}", exc_info=True)
                    raise
        
        raise Exception(f"Request failed after {retry_count} attempts")
    
    def get_stock_list(self) -> List[Dict[str, Any]]:
        """
        获取股票列表
        
        API: /hslt/list/{Token}
        获取基础的股票代码和名称，用于后续接口的参数传入。
        
        更新频率：每日16:20
        请求频率限制：300次/分钟
        
        Returns:
            股票列表，每条记录包含：
            - dm: 股票代码（如"000001"）
            - mc: 股票名称（如"平安银行"）
            - jys: 交易所（"sh"表示上证，"sz"表示深证）
        """
        try:
            endpoint = "/hslt/list"
            response = self._make_request(endpoint)
            
            # API返回列表格式
            if not isinstance(response, list):
                logger.warning(f"Unexpected response format for stock list")
                return []
            
            logger.info(f"Successfully fetched {len(response)} stocks")
            return response
            
        except Exception as e:
            logger.error(f"Failed to fetch stock list: {e}", exc_info=True)
            raise
    
    def get_limit_up_pool(self, trade_date: str) -> List[Dict[str, Any]]:
        """
        获取涨停股池
        
        API: /hslt/ztgc/{日期}/{Token}
        根据日期获取每天的涨停股票列表。
        
        日期格式：YYYY-MM-DD（如：2020-01-15）
        
        Args:
            trade_date: 交易日期（格式'YYYYMMDD'或'YYYY-MM-DD'）
            
        Returns:
            涨停股票列表，每条记录包含涨停相关信息
            字段说明：
            - dm: 代码
            - mc: 名称
            - p: 价格（元）
            - zf: 涨幅（%）
            - cje: 成交额（元）
            - lt: 流通市值（元）
            - zsz: 总市值（元）
            - hs: 换手率（%）
            - lbc: 连板数
            - fbt: 首次封板时间（HH:mm:ss）
            - lbt: 最后封板时间（HH:mm:ss）
            - zj: 封板资金（元）
            - zbc: 炸板次数
            - tj: 涨停统计（x天/y板）
            - hy: 所属行业
        """
        try:
            # 转换日期格式为 YYYY-MM-DD（MomaAPI要求这种格式）
            if len(trade_date) == 8:  # YYYYMMDD格式
                formatted_date = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:8]}"
            elif len(trade_date) == 10 and '-' in trade_date:  # YYYY-MM-DD格式
                formatted_date = trade_date
            else:
                raise ValueError(f"Invalid trade_date format: {trade_date}, expected YYYYMMDD or YYYY-MM-DD")
            
            # 正确的API路径：/hslt/ztgc/{日期}
            endpoint = f"/hslt/ztgc/{formatted_date}"
            response = self._make_request(endpoint)
            
            # API返回格式需要根据实际响应调整
            if isinstance(response, list):
                limit_up_stocks = response
            elif isinstance(response, dict):
                limit_up_stocks = response.get("stocks", []) or response.get("data", []) or response.get("list", [])
            else:
                logger.warning(f"Unexpected response format for limit up pool: {trade_date}")
                return []
            
            # 转换字段名（根据MomaAPI实际响应）
            result = []
            for item in limit_up_stocks:
                # 提取股票代码（dm字段）
                stock_code = item.get('dm', '')
                if not stock_code:
                    continue
                
                # 标准化股票代码（确保是6位数字）
                stock_code = self._normalize_stock_code(stock_code)
                if len(stock_code) != 6 or not stock_code.isdigit():
                    continue
                
                moma_symbol = self._to_moma_symbol(stock_code)
                
                # 提取价格信息（根据MomaAPI字段名）
                current_price = item.get('p', 0)  # 价格（元）
                change_pct = item.get('zf', 0)  # 涨幅（%）
                
                # 涨停价 = 当前价格（因为已经是涨停股票）
                limit_up_price = current_price
                
                stock = {
                    "trade_date": trade_date,
                    "formatted_date": formatted_date,
                    "stock_code": stock_code,
                    "moma_symbol": moma_symbol,
                    "stock_name": item.get('mc', ''),  # 名称
                    "current_price": float(current_price) if current_price else 0,
                    "limit_up_price": float(limit_up_price) if limit_up_price else 0,
                    "price_change_pct": float(change_pct) if change_pct else 0,
                    # MomaAPI原始字段
                    "turnover": float(item.get('cje', 0)),  # 成交额（元）
                    "circulating_market_value": float(item.get('lt', 0)),  # 流通市值（元）
                    "total_market_value": float(item.get('zsz', 0)),  # 总市值（元）
                    "turnover_rate": float(item.get('hs', 0)),  # 换手率（%）
                    "consecutive_boards": int(item.get('lbc', 0)),  # 连板数
                    "first_seal_time": item.get('fbt', ''),  # 首次封板时间
                    "last_seal_time": item.get('lbt', ''),  # 最后封板时间
                    "seal_funds": float(item.get('zj', 0)),  # 封板资金（元）
                    "explosion_count": int(item.get('zbc', 0)),  # 炸板次数
                    "limit_up_statistics": item.get('tj', ''),  # 涨停统计
                    "industry": item.get('hy', ''),  # 所属行业
                    **item  # 保留原始数据
                }
                result.append(stock)
            
            logger.info(f"Successfully fetched {len(result)} limit up stocks for {trade_date}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to fetch limit up pool for {trade_date}: {e}", exc_info=True)
            raise
    
    def get_history_kline(
        self,
        stock_code: str,
        period: str = "d",  # 分时级别：5, 15, 30, 60, d, w, m, y
        adjust: str = "n",  # 除权方式：n, f, b, fr, br
        start_date: Optional[str] = None,  # 格式：YYYYMMDD
        end_date: Optional[str] = None,    # 格式：YYYYMMDD
        limit: Optional[int] = None        # 最新条数
    ) -> List[Dict[str, Any]]:
        """
        获取历史K线数据
        
        API: /hsstock/history/{股票代码}/{分时级别}/{除权类型}/{Token}?st=开始时间&et=结束时间&lt=最新条数
        
        根据《股票列表》得到的股票代码和分时级别获取历史K线数据，交易时间升序。
        目前分时级别支持5分钟、15分钟、30分钟、60分钟、日线、周线、月线、年线，
        对应的请求参数分别为5、15、30、60、d、w、m、y。
        日线以上除权方式有不复权、前复权、后复权、等比前复权、等比后复权，
        对应的参数分别为n、f、b、fr、br，分钟级仅限请求不复权数据，对应的参数为n。
        
        Args:
            stock_code: 股票代码（如'000001'）
            period: 分时级别（默认'd'日线）
                - 5, 15, 30, 60: 分钟级别
                - d: 日线
                - w: 周线
                - m: 月线
                - y: 年线
            adjust: 除权方式（默认'n'不复权）
                - n: 不复权（分钟级必须使用n）
                - f: 前复权（推荐用于技术分析）
                - b: 后复权
                - fr: 等比前复权
                - br: 等比后复权
            start_date: 开始日期（格式：YYYYMMDD，如'20240101'）
            end_date: 结束日期（格式：YYYYMMDD，如'20241231'）
            limit: 最新条数（如指定lt=10，则获取最新的10条数据）
        
        Returns:
            K线数据列表，每条记录包含标准OHLCV字段
        """
        stock_code = self._normalize_stock_code(stock_code)
        moma_symbol = self._to_moma_symbol(stock_code)
        
        try:
            # 构建API路径：/hsstock/history/{股票代码}/{分时级别}/{除权类型}
            # 注意：Token会在_make_request中自动添加到路径末尾
            endpoint = f"/hsstock/history/{moma_symbol}/{period}/{adjust}"
            
            # 构建查询参数
            params = {}
            if start_date:
                params["st"] = start_date
            if end_date:
                params["et"] = end_date
            if limit:
                params["lt"] = limit
            
            logger.debug(f"Requesting K-line data: endpoint={endpoint}, params={params}")
            
            response = self._make_request(endpoint, params=params)
            
            # 调试：检查响应格式
            logger.debug(f"API response type: {type(response)}")
            if isinstance(response, list):
                logger.debug(f"API response: list with {len(response)} items")
                if len(response) > 0:
                    logger.debug(f"First item sample: {response[0]}")
            elif isinstance(response, dict):
                logger.warning(f"API response is dict (not list): {response}")
                # 可能是错误响应
                if 'code' in response or 'msg' in response or 'error' in response:
                    error_msg = response.get('msg') or response.get('error') or str(response)
                    raise Exception(f"API返回错误: {error_msg}")
            else:
                logger.warning(f"Unexpected API response type: {type(response)}, value: {response}")
            
            # API返回列表格式
            if not isinstance(response, list):
                logger.error(f"Unexpected response format for history kline: {stock_code}, response type: {type(response)}")
                logger.error(f"Response content: {response}")
                return []
            
            # 转换字段名（从MomaAPI字段转为标准OHLCV格式）
            kline_data = []
            for item in response:
                # 处理时间格式：MomaAPI返回的时间格式可能是 "yyyy-MM-dd" 或 "yyyy-MM-ddHH:mm:ss"
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
                
                # MomaAPI可能返回小写l或大写L，需要兼容处理
                low_price = item.get("L") or item.get("l")
                
                # 确保所有价格字段都是数值类型
                open_price = item.get("o")
                high_price = item.get("h")
                
                # 验证数据完整性
                if close_price is None or open_price is None or high_price is None or low_price is None:
                    logger.warning(f"K线数据不完整，跳过: {item}")
                    continue
                
                kline = {
                    "date": date_str,
                    "open": float(open_price) if open_price is not None else 0.0,  # 开盘价
                    "high": float(high_price) if high_price is not None else 0.0,  # 最高价
                    "low": float(low_price) if low_price is not None else 0.0,        # 最低价（兼容L和l）
                    "close": float(close_price) if close_price is not None else 0.0,  # 收盘价
                    "volume": float(item.get("v", 0)) if item.get("v") is not None else 0.0,  # 成交量
                    "amount": float(item.get("a", 0)) if item.get("a") is not None else 0.0,  # 成交额
                    "pre_close": float(pre_close) if pre_close is not None else None,   # 前收盘价
                    "change_pct": change_pct, # 涨跌幅（%）
                    "is_suspended": bool(item.get("sf", 0)),  # 停牌标志
                }
                kline_data.append(kline)
            
            logger.info(f"Successfully fetched {len(kline_data)} kline records for {stock_code} (period={period}, adjust={adjust})")
            return kline_data
            
        except Exception as e:
            logger.error(f"Failed to fetch history kline for {stock_code}: {e}", exc_info=True)
            raise

