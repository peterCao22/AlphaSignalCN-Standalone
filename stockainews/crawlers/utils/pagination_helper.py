"""
分页辅助工具类
用于处理多页内容的翻页和弹窗关闭
"""
import logging
from typing import List, Optional, Callable
from pathlib import Path
from playwright.sync_api import Page
from stockainews.crawlers.utils.popup_handler import PopupHandler

logger = logging.getLogger(__name__)


class PaginationHelper:
    """
    分页辅助工具类
    
    功能：
    1. 关闭页面弹窗（硬编码方式）
    2. 翻页并收集所有页面的内容
    3. 合并多页内容
    """
    
    @staticmethod
    def close_popup_ads(page: Page) -> bool:
        """
        关闭页面上的移动端营销弹窗（兼容方法，内部调用 PopupHandler）
        
        Args:
            page: Playwright页面对象
        
        Returns:
            bool: 是否成功关闭弹窗
        """
        return PopupHandler.close_popup_ads(page)
    
    @staticmethod
    def crawl_all_pages(
        page: Page,
        content_extractor: Callable[[Page], str],
        max_pages: int = 10,
        merge_content: bool = True
    ) -> List[str]:
        """
        爬取所有页面的内容
        
        Args:
            page: Playwright页面对象
            content_extractor: 内容提取函数，接收Page对象，返回当前页面的内容字符串
            max_pages: 最大翻页数
            merge_content: 是否合并所有页面内容（True返回合并后的字符串列表，False返回每页内容的列表）
        
        Returns:
            如果merge_content=True，返回包含所有页面合并内容的列表（长度为1）
            如果merge_content=False，返回每页内容的列表
        """
        all_contents = []
        
        try:
            # 关闭初始弹窗
            PaginationHelper.close_popup_ads(page)
            page.wait_for_timeout(1000)
            
            # 提取第一页内容
            logger.info("提取第1页内容...")
            first_page_content = content_extractor(page)
            if first_page_content:
                all_contents.append(first_page_content)
                logger.info(f"第1页内容长度: {len(first_page_content)} 字符")
            
            # 查找下一页按钮
            next_page_button = page.get_by_role("link", name="下一页")
            if next_page_button.count() == 0:
                next_page_button = page.get_by_role("button", name="下一页")
            
            if next_page_button.count() == 0:
                logger.info("未找到下一页按钮，只有1页内容")
                return all_contents if not merge_content else [first_page_content] if first_page_content else []
            
            # 翻页循环
            page_count = 1
            while page_count < max_pages:
                # 关闭弹窗
                PaginationHelper.close_popup_ads(page)
                
                # 重新查找下一页按钮
                next_page_button = page.get_by_role("link", name="下一页")
                if next_page_button.count() == 0:
                    next_page_button = page.get_by_role("button", name="下一页")
                
                if next_page_button.count() == 0:
                    logger.info("找不到下一页按钮，停止翻页")
                    break
                
                # 检查disabled属性
                try:
                    disabled_attr = next_page_button.get_attribute("disabled")
                    if disabled_attr is not None:
                        logger.info("下一页按钮已被禁用，已到达最后一页")
                        break
                except:
                    pass
                
                # 检查是否可见
                try:
                    if not next_page_button.is_visible():
                        logger.info("下一页按钮不可见，停止翻页")
                        break
                except:
                    pass
                
                # 获取当前页码和总页数
                try:
                    page_text = page.evaluate("() => document.body.innerText")
                    import re
                    current_match = re.search(r'当前第(\d+)页', page_text)
                    total_match = re.search(r'共(\d+)页', page_text)
                    if current_match and total_match:
                        actual_current = int(current_match.group(1))
                        total = int(total_match.group(1))
                        logger.debug(f"当前页: {actual_current}/{total}")
                        if actual_current >= total:
                            logger.info(f"当前页 {actual_current} 已经是最后一页（共{total}页），停止翻页")
                            break
                except:
                    pass
                
                # 点击下一页前关闭弹窗
                PaginationHelper.close_popup_ads(page)
                page.wait_for_timeout(300)
                
                # 点击下一页
                click_success = False
                try:
                    next_page_button.scroll_into_view_if_needed(timeout=3000)
                    page.wait_for_timeout(200)
                    
                    try:
                        next_page_button.click(timeout=5000)
                        click_success = True
                        logger.debug("正常点击成功")
                    except:
                        try:
                            next_page_button.evaluate("el => el.click()")
                            click_success = True
                            logger.debug("JavaScript点击成功")
                        except:
                            try:
                                next_page_button.click(force=True, timeout=5000)
                                click_success = True
                                logger.debug("强制点击成功")
                            except Exception as force_error:
                                logger.warning(f"所有点击方式都失败: {force_error}")
                    
                    if click_success:
                        page.wait_for_timeout(1500)
                        
                        # 关闭弹窗
                        PaginationHelper.close_popup_ads(page)
                        page.wait_for_timeout(300)
                        
                        # 验证页面是否变化
                        try:
                            current_url = page.url
                            if 'marketing' in current_url.lower() or 'activity' in current_url.lower():
                                logger.warning(f"检测到跳转到营销页面: {current_url}，返回原页面")
                                page.go_back(timeout=5000)
                                page.wait_for_timeout(1000)
                                PaginationHelper.close_popup_ads(page)
                                continue
                        except:
                            pass
                        
                        # 提取当前页内容
                        page_count += 1
                        logger.info(f"提取第{page_count}页内容...")
                        page_content = content_extractor(page)
                        if page_content:
                            all_contents.append(page_content)
                            logger.info(f"第{page_count}页内容长度: {len(page_content)} 字符")
                        else:
                            logger.warning(f"第{page_count}页内容为空")
                    else:
                        logger.warning("点击失败，停止翻页")
                        break
                        
                except Exception as e:
                    logger.error(f"点击下一页时发生错误: {e}")
                    break
            
            logger.info(f"共提取 {len(all_contents)} 页内容")
            
            # 合并内容
            if merge_content and all_contents:
                merged_content = "\n\n---\n\n".join(all_contents)
                logger.info(f"合并后内容总长度: {len(merged_content)} 字符")
                return [merged_content]
            
            return all_contents
            
        except Exception as e:
            logger.error(f"爬取多页内容时发生错误: {e}")
            return all_contents if all_contents else []

