"""
通用弹窗处理工具

从 PaginationHelper 中剥离出的通用弹窗关闭功能，可在任何爬虫中使用。
支持多种弹窗检测和关闭策略。
"""
from typing import Optional, List, Callable
from playwright.async_api import Page
from stockainews.core.logger import setup_logger

logger = setup_logger(__name__)


class PopupHandler:
    """通用弹窗处理工具"""
    
    # 可扩展的弹窗关闭函数列表（按优先级排序）
    _custom_handlers: List[Callable[[Page], bool]] = []
    
    @classmethod
    def register_handler(cls, handler: Callable[[Page], bool]) -> None:
        """
        注册自定义弹窗关闭处理器
        
        Args:
            handler: 处理函数，接收 Page 对象，返回 bool（是否成功关闭）
        """
        cls._custom_handlers.append(handler)
        logger.debug(f"注册自定义弹窗处理器: {handler.__name__}")
    
    @classmethod
    async def close_popups(cls, page: Page, custom_handlers: Optional[List[Callable[[Page], bool]]] = None) -> bool:
        """
        关闭页面上的弹窗（通用方法）
        
        按优先级尝试多种策略：
        1. 自定义处理器（如果提供）
        2. 调用关闭函数（如 tk_tg_zoomin）
        3. 查找并点击关闭按钮
        4. 移除高 z-index 弹窗容器
        
        Args:
            page: Playwright 页面对象
            custom_handlers: 自定义处理器列表（可选，用于特定页面的硬编码处理）
            
        Returns:
            bool: 是否成功关闭弹窗
        """
        try:
            # 策略1：使用自定义处理器（优先级最高）
            handlers_to_try = custom_handlers or cls._custom_handlers
            for handler in handlers_to_try:
                try:
                    if handler(page):
                        logger.debug(f"✓ 通过自定义处理器关闭弹窗: {handler.__name__}")
                        await page.wait_for_timeout(300)
                        return True
                except Exception as e:
                    logger.debug(f"自定义处理器 {handler.__name__} 失败: {e}")
            
            # 策略2：优先调用关闭函数（最可靠）
            try:
                result = await page.evaluate("""
                    () => {
                        // 尝试常见的关闭函数
                        const closeFunctions = ['tk_tg_zoomin', 'closePopup', 'hidePopup', 'closeModal'];
                        for (const funcName of closeFunctions) {
                            if (typeof window[funcName] === 'function') {
                                window[funcName]();
                                return true;
                            }
                        }
                        return false;
                    }
                """)
                if result:
                    logger.debug("✓ 通过关闭函数关闭弹窗")
                    await page.wait_for_timeout(300)
                    return True
            except Exception as e:
                logger.debug(f"调用关闭函数失败: {e}")
            
            # 策略3：查找并点击关闭按钮（通用方式）
            try:
                close_clicked = await page.evaluate("""
                    () => {
                        const allDivs = document.querySelectorAll('div');
                        for (let div of allDivs) {
                            try {
                                const style = window.getComputedStyle(div);
                                if (style.position === 'fixed' || style.position === 'absolute') {
                                    const zIndex = parseInt(style.zIndex) || 0;
                                    if (zIndex >= 100000) {
                                        // 查找关闭按钮（X图标、关闭文字等）
                                        const closeButtons = div.querySelectorAll(
                                            'img[src*="close"], img[src*="ic_close"], ' +
                                            'button:has-text("关闭"), a:has-text("关闭"), ' +
                                            '[class*="close"], [id*="close"], ' +
                                            '[aria-label*="关闭"], [aria-label*="close"]'
                                        );
                                        for (let btn of closeButtons) {
                                            try {
                                                btn.click();
                                                return true;
                                            } catch (e) {
                                                // 忽略错误
                                            }
                                        }
                                    }
                                }
                            } catch (e) {
                                // 忽略错误
                            }
                        }
                        return false;
                    }
                """)
                if close_clicked:
                    logger.debug("✓ 通过关闭按钮关闭弹窗成功")
                    await page.wait_for_timeout(300)
                    return True
            except Exception as e:
                logger.debug(f"通过关闭按钮定位失败: {e}")
            
            # 策略4：查找固定定位的高 z-index 容器中的图片，移除弹窗容器
            try:
                popup_removed = await page.evaluate("""
                    () => {
                        const allDivs = document.querySelectorAll('div');
                        for (let div of allDivs) {
                            try {
                                const style = window.getComputedStyle(div);
                                if (style.position === 'fixed' || style.position === 'absolute') {
                                    const zIndex = parseInt(style.zIndex) || 0;
                                    // 如果 z-index 很高，且包含图片，可能是弹窗
                                    if (zIndex >= 100000) {
                                        const hasImage = div.querySelector('img') !== null;
                                        if (hasImage) {
                                            div.style.display = 'none';
                                            div.style.visibility = 'hidden';
                                            div.style.opacity = '0';
                                            div.style.pointerEvents = 'none';
                                            div.style.zIndex = '-9999';
                                            try {
                                                div.remove();
                                            } catch (e) {
                                                // 忽略错误
                                            }
                                            return true;
                                        }
                                    }
                                }
                            } catch (e) {
                                // 忽略错误
                            }
                        }
                        return false;
                    }
                """)
                if popup_removed:
                    logger.debug("✓ 通过高 z-index 图片容器定位并移除了弹窗")
                    await page.wait_for_timeout(300)
                    return True
            except Exception as e:
                logger.debug(f"通过图片容器定位弹窗失败: {e}")
            
            # 策略5：直接查找 z-index >= 100000 的固定定位元素并移除（最激进）
            try:
                removed_count = await page.evaluate("""
                    () => {
                        let removed = 0;
                        const allDivs = document.querySelectorAll('div');
                        for (let div of allDivs) {
                            try {
                                const style = window.getComputedStyle(div);
                                if (style.position === 'fixed' || style.position === 'absolute') {
                                    const zIndex = parseInt(style.zIndex) || 0;
                                    // z-index >= 100000 是弹窗容器的特征
                                    if (zIndex >= 100000) {
                                        div.style.display = 'none';
                                        div.style.visibility = 'hidden';
                                        div.style.opacity = '0';
                                        div.style.pointerEvents = 'none';
                                        div.style.zIndex = '-9999';
                                        try {
                                            div.remove();
                                        } catch (e) {
                                            // 忽略错误
                                        }
                                        removed++;
                                    }
                                }
                            } catch (e) {
                                // 忽略错误
                            }
                        }
                        return removed;
                    }
                """)
                if removed_count > 0:
                    logger.debug(f"✓ 移除了 {removed_count} 个高 z-index 弹窗容器（z-index >= 100000）")
                    await page.wait_for_timeout(300)
                    return True
            except Exception as e:
                logger.debug(f"移除高 z-index 弹窗失败: {e}")
            
            return False
        except Exception as e:
            logger.debug(f"关闭弹窗失败: {e}")
            return False
    
    @classmethod
    async def close_popup_ads(cls, page: Page) -> bool:
        """
        关闭页面上的移动端营销弹窗（兼容 PaginationHelper 的方法名）
        
        Args:
            page: Playwright 页面对象
            
        Returns:
            bool: 是否成功关闭弹窗
        """
        return await cls.close_popups(page)

