"""
爬虫适配器基类

封装Playwright操作，集成代理IP池管理器，实现任务拆分和超时控制。
"""
import time
import asyncio
import threading
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

try:
    from playwright.async_api import async_playwright, Browser, Page, BrowserContext
except ImportError:
    raise ImportError(
        "Playwright is required. Please install it with: pip install playwright && playwright install chromium"
    )

from stockainews.core.config import config
from stockainews.core.logger import setup_logger
from stockainews.core.exceptions import CrawlerError
from stockainews.core.proxy_pool import ProxyPoolManager, ProxyIP, get_proxy_pool_manager
from stockainews.crawlers.utils.popup_handler import PopupHandler

logger = setup_logger(__name__)


class BaseCrawler(ABC):
    """爬虫适配器基类"""
    
    def __init__(
        self,
        task_timeout: int = 50,
        headless: bool = None,
        browser_type: str = "chromium"
    ):
        """
        初始化爬虫适配器
        
        Args:
            task_timeout: 任务超时时间（秒），默认50秒
            headless: 是否使用无头模式，默认从配置读取
            browser_type: 浏览器类型，默认chromium
        """
        self.task_timeout = task_timeout
        self.headless = headless if headless is not None else config.crawler_headless
        self.browser_type = browser_type
        
        # 初始化代理池管理器
        self.proxy_pool = get_proxy_pool_manager()
        
        # Playwright实例
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        
        logger.info(
            f"{self.__class__.__name__} initialized: "
            f"task_timeout={self.task_timeout}s, headless={headless}"
        )
    
    
    # 获取浏览器代理配置
    def _get_browser_proxy_config(self, proxy: Optional[ProxyIP] = None) -> Optional[Dict[str, str]]:
        """
        获取浏览器代理配置
        
        Args:
            proxy: 代理IP对象，如果为None则从代理池获取（仅在代理模式启用时）
            
        Returns:
            代理配置字典，如果不需要代理则返回None
        """
        # 如果明确指定不使用代理（proxy=None且代理模式未启用），返回None
        if proxy is None and not self.proxy_pool.proxy_enabled:
            return None
        
        # 如果代理模式已启用，尝试获取代理IP
        if proxy is None and self.proxy_pool.proxy_enabled:
            proxy = self.proxy_pool.get_proxy()
        
        if proxy is None:
            return None  # 不使用代理
        
        if proxy.username and proxy.password:
            return {
                "server": f"http://{proxy.ip}:{proxy.port}",
                "username": proxy.username,
                "password": proxy.password
            }
        else:
            return {
                "server": f"http://{proxy.ip}:{proxy.port}"
            }
    
    # 初始化浏览器实例
    async def _init_browser(self, proxy: Optional[ProxyIP] = None) -> Browser:
        """
        初始化浏览器实例
        
        Args:
            proxy: 代理IP对象，如果为None则从代理池获取
            
        Returns:
            浏览器实例
        """
        if self.playwright is None:
            self.playwright = await async_playwright().start()
        
        proxy_config = self._get_browser_proxy_config(proxy)
        
        browser_options = {
            "headless": self.headless,
            "timeout": self.task_timeout * 1000,  # 转换为毫秒
        }
        
        if proxy_config:
            browser_options["proxy"] = proxy_config
            logger.info(f"Using proxy: {proxy_config['server']}")
        
        if self.browser_type == "chromium":
            self.browser = await self.playwright.chromium.launch(**browser_options)
        elif self.browser_type == "firefox":
            self.browser = await self.playwright.firefox.launch(**browser_options)
        elif self.browser_type == "webkit":
            self.browser = await self.playwright.webkit.launch(**browser_options)
        else:
            raise ValueError(f"Unsupported browser type: {self.browser_type}")
        
        return self.browser
    
    # 创建浏览器上下文
    async def _create_context(self, proxy: Optional[ProxyIP] = None) -> BrowserContext:
        """
        创建浏览器上下文
        
        Args:
            proxy: 代理IP对象，如果为None则从代理池获取
            
        Returns:
            浏览器上下文
        """
        if self.browser is None:
            await self._init_browser(proxy)
        
        context_options = {
            "viewport": {"width": 1920, "height": 1080},
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        
        self.context = await self.browser.new_context(**context_options)
        return self.context
    
    # 获取页面实例
    async def _get_page(self, proxy: Optional[ProxyIP] = None) -> Page:
        """
        获取页面实例
        
        Args:
            proxy: 代理IP对象，如果为None则从代理池获取
            
        Returns:
            页面实例
        """
        # 如果已有页面，先关闭它（避免资源泄漏）
        if self.page:
            try:
                await self.page.close()
            except:
                pass
            self.page = None
        
        if self.context is None:
            await self._create_context(proxy)
        
        self.page = await self.context.new_page()
        # 设置超时时间
        self.page.set_default_timeout(self.task_timeout * 1000)
        
        # 添加弹窗处理钩子：页面加载后自动尝试关闭弹窗
        async def handle_popup_on_load():
            try:
                await PopupHandler.close_popups(self.page)
            except Exception as e:
                logger.debug(f"自动关闭弹窗失败: {e}")
        
        # 监听页面加载事件
        self.page.on("load", handle_popup_on_load)
        
        return self.page
    
    # 执行带超时控制的函数
    def _execute_with_timeout(
        self,
        func,
        *args,
        proxy: Optional[ProxyIP] = None,
        **kwargs
    ) -> Any:
        """
        执行带超时控制的函数
        
        Args:
            func: 要执行的函数
            *args: 函数参数
            proxy: 代理IP对象
            **kwargs: 函数关键字参数
            
        Returns:
            函数执行结果
        """
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            
            if elapsed > self.task_timeout:
                logger.warning(
                    f"Task took {elapsed:.1f}s, exceeded timeout of {self.task_timeout}s"
                )
            else:
                logger.debug(f"Task completed in {elapsed:.1f}s")
            
            return result
            
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(
                f"Task failed after {elapsed:.1f}s: {e}",
                exc_info=True
            )
            raise
    
    # 检测是否IP被限制
    def _is_ip_restricted(self, error: Exception) -> bool:
        """
        检测是否IP被限制
        
        Args:
            error: 异常对象
            
        Returns:
            如果被限制返回True，否则返回False
        """
        error_str = str(error).lower()
        error_type = type(error).__name__
        
        # 检查常见的限制错误
        restriction_indicators = [
            "403",  # Forbidden
            "429",  # Too Many Requests
            "blocked",  # 被阻止
            "forbidden",  # 禁止访问
            "access denied",  # 访问被拒绝
            "rate limit",  # 速率限制
            "too many requests",  # 请求过多
            "ip banned",  # IP被封禁
            "ip blocked",  # IP被阻止
            "验证码",  # 验证码
            "captcha",  # 验证码
            "反爬虫",  # 反爬虫
            "请稍后再试",  # 请稍后再试
        ]
        
        # 检查错误消息中是否包含限制关键词
        for indicator in restriction_indicators:
            if indicator in error_str:
                logger.warning(f"IP restriction detected: {indicator}")
                return True
        
        # 检查HTTP状态码
        if hasattr(error, 'status') or hasattr(error, 'status_code'):
            status = getattr(error, 'status', None) or getattr(error, 'status_code', None)
            if status in [403, 429]:
                logger.warning(f"IP restriction detected: HTTP {status}")
                return True
        
        return False
    
    # 检测是否是代理认证错误
    def _is_proxy_auth_error(self, error: Exception) -> bool:
        """
        检测是否是代理认证错误
        
        Args:
            error: 异常对象
            
        Returns:
            如果是代理认证错误返回True，否则返回False
        """
        error_str = str(error).lower()
        
        # 检查代理认证相关的错误
        proxy_auth_indicators = [
            "err_proxy_auth_unsupported",
            "proxy_auth",
            "proxy authentication",
            "代理认证",
            "代理授权"
        ]
        
        for indicator in proxy_auth_indicators:
            if indicator in error_str:
                logger.warning(f"Proxy authentication error detected: {indicator}")
                return True
        
        return False
    
    # 关闭当前的浏览器上下文和页面
    async def _close_current_context(self) -> None:
        """
        关闭当前的浏览器上下文和页面（用于代理切换）
        """
        try:
            if self.page:
                try:
                    await self.page.close()
                except:
                    pass
                self.page = None
            
            if self.context:
                try:
                    await self.context.close()
                except:
                    pass
                self.context = None
            
            logger.debug("Current browser context closed")
        except Exception as e:
            logger.warning(f"Error closing context: {e}")
    
    # 获取当前直连的IP地址（可能有多个出口IP）
    def _get_current_ip_addresses(self) -> List[str]:
        """
        获取当前直连的IP地址（可能有多个出口IP）
        
        Returns:
            IP地址列表
        """
        import httpx
        
        ip_addresses = []
        
        # 使用多个服务检测IP地址
        ip_check_services = [
            "https://api.ipify.org?format=json",
            "https://httpbin.org/ip",
            "https://api.ip.sb/ip",
        ]
        
        for service_url in ip_check_services:
            try:
                with httpx.Client(timeout=5.0) as client:
                    response = client.get(service_url)
                    if response.status_code == 200:
                        data = response.json()
                        # 不同服务返回格式不同
                        if isinstance(data, dict):
                            ip = data.get("ip") or data.get("origin")
                        else:
                            ip = data.strip() if isinstance(data, str) else None
                        
                        if ip:
                            # 可能有多个IP（逗号分隔）
                            ips = [ip.strip() for ip in ip.split(",")]
                            for ip_addr in ips:
                                if ip_addr and ip_addr not in ip_addresses:
                                    ip_addresses.append(ip_addr)
                        if ip_addresses:
                            break  # 获取到IP后退出
            except Exception:
                continue
        
        return ip_addresses
    
    # 测试代理IP的连接性（使用httpx）
    def _test_proxy_connectivity(self, proxy: ProxyIP) -> None:
        """
        测试代理IP的连接性（使用httpx）
        
        Args:
            proxy: 代理IP对象
        """
        import httpx
        
        try:
            # 构建代理URL
            if proxy.username and proxy.password:
                proxy_url = f"http://{proxy.username}:{proxy.password}@{proxy.ip}:{proxy.port}"
            else:
                proxy_url = f"http://{proxy.ip}:{proxy.port}"
            
            # 测试连接（使用一个简单的测试URL）
            test_url = "http://httpbin.org/ip"
            
            try:
                with httpx.Client(proxy=proxy_url, timeout=5.0) as client:
                    response = client.get(test_url)
                    if response.status_code == 200:
                        logger.debug(f"Proxy connectivity test passed: {proxy.ip}:{proxy.port}")
                        return
            except Exception:
                pass
            
        except Exception:
            pass
    
    # 带重试的页面导航（优先使用直连，被限制后使用代理）
    async def _navigate_with_retry(
        self,
        url: str,
        proxy: Optional[ProxyIP] = None,
        max_retries: int = 3,
        wait_until: str = "networkidle",
        use_proxy_on_restriction: bool = True
    ) -> Page:
        """
        带重试的页面导航（优先使用直连，被限制后使用代理）
        
        Args:
            url: 要访问的URL
            proxy: 代理IP对象（如果为None，优先使用直连）
            max_retries: 最大重试次数
            wait_until: 等待条件，默认networkidle
            use_proxy_on_restriction: 检测到限制后是否使用代理，默认True
            
        Returns:
            页面实例
        """
        # 首先尝试不使用代理（直连）
        if proxy is None and not self.proxy_pool.proxy_enabled:
            try:
                logger.info(f"Attempting direct connection to {url}")
                page = await self._get_page(proxy=None)  # 明确不使用代理
                await page.goto(url, wait_until=wait_until, timeout=self.task_timeout * 1000)
                logger.info("Direct connection successful")
                return page
            except Exception as e:
                # 检测是否被限制
                if self._is_ip_restricted(e) and use_proxy_on_restriction:
                    logger.warning(f"Direct connection failed with restriction: {e}")
                    logger.info("Enabling proxy mode and retrying...")
                    self.proxy_pool.enable_proxy()
                    # 继续使用代理重试
                else:
                    # 如果不是限制错误，直接抛出
                    logger.error(f"Direct connection failed: {e}")
                    raise
        
        # 使用代理重试
        attempt = 0
        max_attempts = max_retries * 2  # 增加最大尝试次数，因为代理切换可能需要多次重试
        
        while attempt < max_attempts:
            try:
                # 如果启用了代理模式，获取代理IP
                if self.proxy_pool.proxy_enabled:
                    if proxy is None or attempt > 0:
                        proxy = self.proxy_pool.get_proxy(force_new=(attempt > 0))
                
                page = await self._get_page(proxy)
                connection_type = "proxy" if proxy else "direct"
                logger.info(
                    f"Navigating to {url} ({connection_type}, attempt {attempt + 1}/{max_attempts})"
                )
                
                await page.goto(url, wait_until=wait_until, timeout=self.task_timeout * 1000)
                
                return page
                
            except Exception as e:
                attempt += 1
                logger.warning(f"Navigation attempt {attempt} failed: {e}")
                
                # 检测是否是代理认证错误（需要清理上下文）
                if self._is_proxy_auth_error(e):
                    # 获取当前直连IP地址
                    current_ips = self._get_current_ip_addresses()
                    current_ip_info = ", ".join(current_ips) if current_ips else "Unable to detect"
                    
                    # 记录详细的代理信息
                    proxy_info = "None"
                    if proxy:
                        proxy_info = (
                            f"IP={proxy.ip}:{proxy.port}, "
                            f"username={proxy.username or 'None'}, "
                            f"password={'***' if proxy.password else 'None'}, "
                            f"expires_at={proxy.expires_at.strftime('%Y-%m-%d %H:%M:%S') if proxy.expires_at else 'N/A'}, "
                            f"remaining={proxy.remaining_seconds:.1f}s"
                        )
                    
                    logger.warning(
                        f"Proxy authentication error detected. "
                        f"Current IP(s): {current_ip_info}, "
                        f"Proxy: {proxy.ip}:{proxy.port if proxy else 'None'}"
                    )
                    
                    # 尝试自动添加IP到白名单
                    whitelist_added = False
                    if current_ips:
                        success = self.proxy_pool.add_ips_to_whitelist(current_ips)
                        if success:
                            whitelist_added = True
                            import time
                            time.sleep(10)  # 等待白名单生效
                    
                    # 测试代理IP的连接性（使用httpx）
                    if proxy:
                        self._test_proxy_connectivity(proxy)
                    
                    # 完全关闭浏览器上下文和浏览器实例，重新创建
                    await self._close_current_context()
                    try:
                        if self.browser:
                            await self.browser.close()
                            self.browser = None
                    except Exception:
                        pass
                    
                    # 如果白名单已添加，不标记代理失败，继续使用同一个代理
                    if not whitelist_added:
                        if proxy:
                            self.proxy_pool.mark_proxy_failed(proxy)
                        import time
                        time.sleep(2)
                        if self.proxy_pool.proxy_enabled:
                            proxy = self.proxy_pool.get_proxy(force_new=True)
                            if not proxy and attempt >= max_retries:
                                self.proxy_pool.disable_proxy()
                                proxy = None
                    else:
                        # 重置浏览器实例，强制重新创建
                        self.browser = None
                    
                    # 继续重试，不增加attempt计数（因为这是代理问题，不是真正的失败）
                    continue
                
                # 检测是否被限制
                if self._is_ip_restricted(e) and use_proxy_on_restriction:
                    if not self.proxy_pool.proxy_enabled:
                        logger.warning("IP restriction detected, enabling proxy mode")
                        self.proxy_pool.enable_proxy()
                    # 标记当前代理失败并获取新IP
                    if proxy:
                        self.proxy_pool.mark_proxy_failed(proxy)
                    # 关闭当前上下文，重新创建
                    await self._close_current_context()
                    import time
                    time.sleep(2)
                    proxy = self.proxy_pool.get_proxy(force_new=True)
                    # 继续重试
                    continue
                
                # 如果是最后一次尝试，抛出异常
                if attempt >= max_retries:
                    logger.error(f"Failed to navigate after {attempt} attempts")
                    raise
        
        raise Exception("Failed to navigate after all retries")
    
    # 清理资源
    async def cleanup(self) -> None:
        """清理资源"""
        try:
            if self.page:
                await self.page.close()
                self.page = None
            if self.context:
                await self.context.close()
                self.context = None
            if self.browser:
                await self.browser.close()
                self.browser = None
            if self.playwright:
                await self.playwright.stop()
                self.playwright = None
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
    
    # 上下文管理器入口
    async def __aenter__(self):
        """异步上下文管理器入口"""
        return self
    
    # 上下文管理器出口
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.cleanup()
    
    # 执行爬虫任务
    @abstractmethod
    async def crawl(self, *args, **kwargs) -> Any:
        """
        执行爬虫任务（子类必须实现）
        
        Args:
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            爬取结果
        """
        pass

